from itertools import chain

import pytensor.tensor as pt

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.rewriting.db import EquilibriumDB
from pytensor.graph.traversal import ancestors, applys_between
from pytensor.scan.op import Scan
from pytensor.scan.utils import ScanArgs, safe_new
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable

optimize_db = EquilibriumDB()


def _one_step_of_each_inner_input(args: ScanArgs) -> dict[Variable, Variable]:
    """
    Map every inner input to the outer expression holding one step of it.

    A sequence, a recurrent state and an untraced state all arrive as an outer tensor whose leading axis
    is time, so one step of any of them is its first slice; a non-sequence is already one step's worth.
    Rebuilding an inner expression against this map moves it to the outer graph.

    A multi-tap state reaches the inner graph once per tap and the outer graph once in total, and every
    tap holds the same shape, so they all map to that one buffer's first slice.
    """

    def first_step(outer: Variable) -> Variable:
        assert isinstance(outer, TensorVariable), "a scan's sequences and states are tensors"
        return outer[0]

    one_step = {
        inner: first_step(outer)
        for inners, outers in (
            (args.inner_in_seqs, args.outer_in_seqs),
            (args.inner_in_sit_sot, args.outer_in_sit_sot),
            (args.inner_in_shared, args.outer_in_shared),
        )
        for inner, outer in zip(inners, outers)
    }
    for taps, outer in zip(args.inner_in_mit_sot, args.outer_in_mit_sot):
        one_step.update((tap, first_step(outer)) for tap in taps)
    one_step.update(zip(args.inner_in_non_seqs, args.outer_in_non_seqs))
    return one_step


def _varying_inner_inputs(args: ScanArgs) -> set[Variable]:
    """The inner inputs holding a different value at every step, as against the loop's non-sequences."""
    return {
        *args.inner_in_seqs,
        *args.inner_in_sit_sot,
        *chain.from_iterable(args.inner_in_mit_sot),
        *args.inner_in_shared,
    }


def _draws_from_a_frozen_generator(args: ScanArgs) -> list[Apply]:
    """
    Return the draws whose generator the loop never advances, in topological order.

    Such a draw yields one value that every step reuses, so the loop is not a function of a sequence of
    draws at all, and the graph does not compile because nothing can derive the generator's next state. A
    generator carried as a recurrent state is threaded deliberately and is left alone, as is one shared by
    two draws, which would need a generator each to come apart.

    A draw whose parameters vary from step to step is left alone as well. Its values depend on where the
    loop has got to, so a whole sequence of them cannot be drawn before the loop runs; only the shape may
    depend on the state, since every inner variable holds the same shape at every step.
    """
    inner_nodes = list(applys_between(args.inner_inputs, args.inner_outputs))

    # Every reader counts, not just the draws: taking a generator out of the loop has to leave nothing
    # behind that still reads it.
    read_by: dict[Variable, int] = {}
    for node in inner_nodes:
        for variable in node.inputs:
            read_by[variable] = read_by.get(variable, 0) + 1

    consumed = {variable for node in inner_nodes for variable in node.inputs}
    consumed.update(args.inner_outputs)
    varying = _varying_inner_inputs(args)

    hoistable = []
    for draw in inner_nodes:
        if not isinstance(draw.op, RandomVariable):
            continue
        generator, _size, *parameters = draw.inputs
        if (
            generator in args.inner_in_non_seqs
            and read_by[generator] == 1
            # A next state that something reads is a generator the caller threads themselves.
            and draw.outputs[0] not in consumed
            and varying.isdisjoint(ancestors(parameters))
        ):
            hoistable.append(draw)
    return hoistable


@node_rewriter([Scan])
def hoist_draws_out_of_scan(fgraph: FunctionGraph, node: Apply) -> list[Variable] | None:
    """
    Draw a whole sequence's worth of randomness outside the loop and feed it back in as a sequence.

    A draw written inside a recurrence reads a generator the loop has no way to advance, so it yields one
    value reused at every step, and the graph does not compile because that generator has no update.
    Lifting the draw out replaces it with ``n_steps`` independent values and gives the generator its
    ordinary update. It is also what makes the loop differentiable: a draw inside the differentiated
    region leaves no fixed sample to take a gradient against, which :func:`pytensor.grad` reports as an
    undefined gradient rather than as a number.

    Parameters
    ----------
    fgraph : FunctionGraph
        Graph being rewritten.
    node : Apply
        The ``Scan`` node being rewritten.

    Returns
    -------
    list of Variable or None
        The rebuilt scan's outputs, or None when the loop has no such draw.
    """
    args = ScanArgs.from_node(node, clone=True)
    # Two shapes this does not rewrite, because it cannot reach every place a draw might be used: a
    # mit-mot's outputs are a nested list, and a while loop's condition is an output of its own.
    if args.inner_out_mit_mot or args.as_while:
        return None

    draws = _draws_from_a_frozen_generator(args)
    if not draws:
        return None

    one_step = _one_step_of_each_inner_input(args)
    per_step_draws = {}

    for draw in draws:
        generator, size, *parameters = draw.inputs
        # Rebuilt against one step of every inner input, so a size taken from an intermediate -- the
        # shape of the state at this step, say -- becomes an expression the outer graph can evaluate.
        outer_size, *outer_parameters = clone_replace([size, *parameters], replace=one_step)
        # The op's own inference rather than its `size` input, which is None whenever the shape follows
        # from the parameters, and which says nothing about a multivariate draw's core dimensions. Private
        # to pytensor, and the only place here that reaches past its public surface.
        per_step_shape = draw.op._infer_shape(outer_size, outer_parameters)
        _next_generator, drawn = draw.op(
            *outer_parameters,
            rng=one_step[generator],
            size=pt.stack([args.n_steps, *per_step_shape]),
            return_next_rng=True,
        )

        per_step = safe_new(draw.outputs[1], "_per_step")
        per_step_draws[draw.outputs[1]] = per_step
        args.inner_in_seqs.append(per_step)
        args.outer_in_seqs.append(drawn.astype(draw.outputs[1].type.dtype))

        # Read outside now, so the generator leaves the loop's inputs along with the draw.
        position = args.inner_in_non_seqs.index(generator)
        del args.inner_in_non_seqs[position], args.outer_in_non_seqs[position]

    # Every output field, not just the one an RNN happens to use: a draw left behind in any of them would
    # still reference the generator that just left the loop's inputs.
    for field in (
        "inner_out_mit_sot",
        "inner_out_sit_sot",
        "inner_out_nit_sot",
        "inner_out_shared",
    ):
        setattr(args, field, clone_replace(getattr(args, field), replace=per_step_draws))

    rebuilt = Scan(
        args.inner_inputs,
        args.inner_outputs,
        args.info,
        mode=node.op.mode,
        truncate_gradient=node.op.truncate_gradient,
        name=node.op.name,
        allow_gc=node.op.allow_gc,
    )
    outputs = rebuilt(*args.outer_inputs, return_list=True)
    assert isinstance(outputs, list), "return_list=True yields a list"
    return outputs


optimize_db.register("hoist_draws_out_of_scan", hoist_draws_out_of_scan, "basic", "scan")
