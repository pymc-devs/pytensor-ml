import warnings

from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
import pytensor

from pytensor import Mode
from pytensor.compile import Function, SharedVariable, get_mode
from pytensor.compile.builders import OpFromGraph, SymbolicOp
from pytensor.graph import FunctionGraph, RewriteDatabaseQuery, graph_inputs, rewrite_graph
from pytensor.graph.basic import Apply, Constant, equal_computations
from pytensor.graph.fg import Output
from pytensor.scan.op import Scan
from pytensor.tensor.random.op import RandomVariable, RNGConsumerOp
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.random.variable import RandomGeneratorSharedVariable
from pytensor.tensor.variable import TensorVariable, Variable

SeedSequenceSeed = None | int | Sequence[int] | np.ndarray | np.random.SeedSequence
RandomSeed = None | int | Sequence[int] | np.ndarray


class LayerOp(SymbolicOp):
    """Base class for the library's neural-network ops.

    A ``SymbolicOp`` is an ``OpFromGraph`` whose inner graph is rebuilt from its ``__props__`` and input
    types by :meth:`build_inner_graph`, so equal props with equal inputs yield an identical op. Basing the
    layers on it (rather than plain ``OpFromGraph``) is what lets the numba backend optimize each inner
    graph: ``SymbolicOp`` restores the fgraph-aware ``__eq__``/``__hash__`` that a props-carrying
    ``OpFromGraph`` would otherwise lose, so the ``ofg_inner_graph`` rewrite keeps its optimized inner
    graph instead of discarding it as unchanged.
    """

    __props__: tuple[str, ...] = ()

    def update_map(self) -> dict[int, int]:
        """Return a mapping of output indexes to input indexes"""
        return {}


class UnaryLayerOp(LayerOp):
    """A ``LayerOp`` with exactly one output, typed as such.

    ``SymbolicOp.__call__`` is annotated ``Variable | list[Variable]`` because an op may produce many
    outputs; a unary layer op produces one, so narrow the result to ``TensorVariable``. The ``isinstance``
    guard narrows for the type checker without a cast and asserts the invariant at runtime.
    """

    def __call__(self, *inputs, **kwargs) -> TensorVariable:
        out = super().__call__(*inputs, **kwargs)
        assert isinstance(out, TensorVariable), f"{type(self).__name__} produced multiple outputs"
        return out


def atleast_list(x):
    if not isinstance(x, list | tuple):
        return [x]
    return x


# RNG utilities vendored from pymc.pytensorf to keep pytensor as the only runtime dependency.
# collect_default_updates is the load-bearing one: it threads the next-RNG update for every RandomVariable,
# Scan, and OpFromGraph between inputs and outputs, so a compiled function advances its generators instead
# of repeating draws.
def find_rng_nodes(variables: Iterable[Variable]) -> list[RandomGeneratorSharedVariable]:
    """Return the shared RNG variables in a graph."""
    return [
        node for node in graph_inputs(variables) if isinstance(node, RandomGeneratorSharedVariable)
    ]


def reseed_rngs(rngs: Sequence[SharedVariable], seed: SeedSequenceSeed) -> None:
    """Replace each shared RNG with a fresh generator seeded from ``seed``."""
    bit_generators = [
        np.random.PCG64(sub_seed)
        for sub_seed in np.random.SeedSequence(seed).spawn(len(rngs))  # type: ignore[arg-type]
    ]
    for rng, bit_generator in zip(rngs, bit_generators):
        rng.set_value(np.random.Generator(bit_generator), borrow=True)


def collect_default_updates_inner_fgraph(node: Apply) -> dict[Variable, Variable]:
    """Collect default RNG updates from a node carrying an inner function graph, mapped to outer variables."""
    op = node.op
    inner_updates = collect_default_updates(
        op.inner_outputs, inputs=op.inner_inputs, must_be_shared=False
    )
    updates = {}
    for rng, update in inner_updates.items():
        input_index = op.inner_inputs.index(rng)
        output_index = op.inner_outputs.index(update)
        updates[node.inputs[input_index]] = node.outputs[output_index]
    return updates


def collect_default_updates(
    outputs: Variable | Sequence[Variable],
    *,
    inputs: Sequence[Variable] | None = None,
    must_be_shared: bool = True,
) -> dict[Variable, Variable]:
    """
    Collect the default next-RNG update for every shared RNG used between ``inputs`` and ``outputs``.

    Parameters
    ----------
    outputs : Variable or sequence of Variable
        Graph outputs whose RNG updates to collect.
    inputs : sequence of Variable, optional
        Inputs above which updates are not collected. Defaults to the graph roots.
    must_be_shared : bool
        Whether to collect updates only for shared-variable RNGs. False is used when recursing into the
        inner graph of an op, whose RNG inputs are not shared. Default True.

    Returns
    -------
    dict mapping Variable to Variable
        Each RNG variable to the expression for its next state.
    """

    def find_default_update(clients, rng: Variable) -> None | Variable:
        rng_clients = clients.get(rng, None)

        # Root case, RNG is not used elsewhere
        if not rng_clients:
            return None

        if len(rng_clients) > 1:
            # Multiple clients are fine if they are identical operations with the same default update.
            all_updates = [
                find_default_update(clients | {rng: [rng_client]}, rng)
                for rng_client in rng_clients
            ]
            updates = [update for update in all_updates if update is not None]
            if not updates:
                return None
            if len(updates) == 1:
                return updates[0]
            update, *other_updates = updates
            if all(equal_computations([update], [other_update]) for other_update in other_updates):
                return update
            warnings.warn(
                f"RNG Variable {rng} has multiple distinct clients {rng_clients}, "
                f"likely due to an inconsistent random graph. No default update will be returned.",
                UserWarning,
            )
            return None

        [client, _] = rng_clients[0]
        client_op = client.op

        match client_op:
            case Output():
                return None
            case RandomVariable():
                # A RandomVariable's first output is always the update of its input RNG.
                next_rng = client.outputs[0]
            case RNGConsumerOp():
                # RandomVariable is a subclass of RNGConsumerOp, specialized above for speed.
                next_rng = client_op.update(client).get(rng)
                if next_rng is None:
                    raise ValueError(f"No update found for at least one RNG used in {client_op}")
            case Scan():
                rng_index = client.inputs.index(rng)
                io_map = client_op.get_oinp_iinp_iout_oout_mappings()["outer_out_from_outer_inp"]
                output_index = io_map.get(rng_index, -1)
                if output_index != -1:
                    next_rng = client.outputs[output_index]
                else:
                    raise ValueError(
                        f"No update found for at least one RNG used in Scan Op {client_op}."
                    )
            case OpFromGraph():
                try:
                    next_rng = collect_default_updates_inner_fgraph(client).get(rng)
                    if next_rng is None:
                        return None
                except ValueError as exc:
                    raise ValueError(
                        f"No update found for at least one RNG used in OpFromGraph Op {client_op}."
                    ) from exc
            case _:
                # Unknown consumer; the caller must provide an update manually.
                return None

        nested_next_rng = find_default_update(clients, next_rng)
        return next_rng if nested_next_rng is None else nested_next_rng

    if inputs is None:
        inputs = []

    outs = atleast_list(outputs)
    clients = FunctionGraph(outputs=outs, clone=False).clients

    rng_updates = {}
    for input_rng in (
        inp
        for inp in graph_inputs(outs, blockers=inputs)
        if (not must_be_shared or isinstance(inp, SharedVariable))
        and isinstance(inp.type, RandomType)
    ):
        default_update = find_default_update(clients, input_rng)
        if default_update is not None:
            rng_updates[input_rng] = default_update

    return rng_updates


def function(
    inputs,
    outputs,
    random_seed: SeedSequenceSeed = None,
    mode=None,
    **kwargs,
) -> Function:
    """
    Compile a Pytensor function, including specialized rewrites.

    Parameters
    ----------
    inputs: list of TensorVariables, optional
        Inputs of the compiled PyTensor function
    outputs: list of TensorVariables, optional
        Outputs of the compiled PyTensor function
    random_seed: int, array-like of int or SeedSequence, optional
        Seed used to override any RandomState/Generator shared variables in the graph.
        If not specified, the value of original shared variables will still be overwritten.
    mode: optional
        PyTensor mode used to compile the function

    Returns
    -------
    pytensor_function: Function
        Compiled function
    """
    rng_updates = collect_default_updates(
        inputs=[inp.variable if isinstance(inp, pytensor.In) else inp for inp in inputs],
        outputs=[
            out.variable if isinstance(out, pytensor.Out) else out for out in atleast_list(outputs)
        ],
    )

    if rng_updates:
        rngs = cast(list[SharedVariable], list(rng_updates))
        reseed_rngs(rngs, random_seed)

    mode = get_mode(mode)
    opt_qry = mode.provided_optimizer.including("random_make_inplace")
    mode = Mode(linker=mode.linker, optimizer=opt_qry)
    pytensor_function = pytensor.function(
        inputs,
        outputs,
        updates={**rng_updates, **kwargs.pop("updates", {})},
        mode=mode,
        **kwargs,
    )
    return pytensor_function


def rewrite_pregrad(graph):
    """Apply simplifying or stabilizing rewrites to graph that are safe to use pre-grad."""
    return rewrite_graph(graph, include=("canonicalize", "stabilize"))


def rewrite_for_prediction(graph):
    """Apply rewrites to specialize a graph for forward passes (e.g. removing Dropout layers)"""
    from pytensor_ml.rewriting.layers import predict_db

    if isinstance(graph, FunctionGraph):
        fgraph = graph
    else:
        outputs = [graph] if isinstance(graph, Variable) else graph
        fgraph = FunctionGraph(outputs=outputs, clone=True, copy_inputs=False)

    rewriter = predict_db.query(RewriteDatabaseQuery(include=["basic"]))
    rewriter.rewrite(fgraph)

    if isinstance(graph, FunctionGraph):
        return fgraph
    if isinstance(graph, Variable):
        return fgraph.outputs[0]

    return fgraph.outputs


def compile_predict(
    prediction: Variable,
    *,
    inputs: Sequence[Variable] | None = None,
    compile_kwargs: dict | None = None,
) -> Function:
    """
    Compile a forward-pass function, specialized for inference.

    Applies :func:`rewrite_for_prediction` to the graph before compiling, which drops stochastic training-only
    layers (such as Dropout) and switches batch norm to its running statistics. The data inputs are collected
    from the graph unless given explicitly.

    Parameters
    ----------
    prediction : Variable
        The model output to evaluate.
    inputs : sequence of Variable, optional
        Data inputs of the compiled function, in call order. Collected from the graph (the non-constant,
        non-shared inputs) when omitted; pass them explicitly when call order matters.
    compile_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`function`.

    Returns
    -------
    Function
        The compiled prediction function.
    """
    prediction = rewrite_for_prediction(prediction)
    if inputs is None:
        inputs = [
            variable
            for variable in graph_inputs([prediction])
            if not isinstance(variable, Constant | SharedVariable)
        ]
    return function(list(inputs), prediction, **(compile_kwargs or {}))


__all__ = ["compile_predict", "function", "rewrite_for_prediction", "rewrite_pregrad"]
