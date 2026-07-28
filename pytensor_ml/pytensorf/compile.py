from collections.abc import Sequence
from typing import cast

import pytensor

from pytensor import Mode
from pytensor.compile import Function, SharedVariable, get_mode
from pytensor.tensor.variable import Variable

from pytensor_ml.pytensorf.collect import collect_graph_inputs
from pytensor_ml.pytensorf.rewrite import rewrite_for_prediction
from pytensor_ml.pytensorf.rng import (
    SeedSequenceSeed,
    atleast_list,
    collect_default_updates,
    reseed_rngs,
)


def function(
    inputs,
    outputs,
    random_seed: SeedSequenceSeed = None,
    mode=None,
    **kwargs,
) -> Function:
    """
    Compile a Pytensor function, including specialized rewrites.

    Threads the default next-RNG update for every shared generator the graph draws from, so repeated calls
    advance their state instead of repeating draws.

    Parameters
    ----------
    inputs : list of Variable
        Inputs of the compiled function.
    outputs : Variable or list of Variable
        Outputs of the compiled function.
    random_seed : int, array-like of int, or SeedSequence, optional
        Seed used to reseed the graph's shared generators. They are replaced whether or not a seed is
        given, so omitting it reseeds from fresh entropy rather than leaving them untouched.
    mode : Mode or str, optional
        PyTensor mode used to compile the function.
    **kwargs
        Forwarded to :func:`pytensor.function`. Any ``updates`` entry is merged after the RNG updates.

    Returns
    -------
    Function
        The compiled function.
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

    base_mode = get_mode(mode)
    mode = Mode(
        linker=base_mode.linker,
        optimizer=base_mode.provided_optimizer.including("random_make_inplace"),
    )

    return pytensor.function(
        inputs,
        outputs,
        updates={**rng_updates, **kwargs.pop("updates", {})},
        mode=mode,
        **kwargs,
    )


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
    specialized = rewrite_for_prediction(prediction)
    if inputs is None:
        inputs = collect_graph_inputs(specialized)
    return function(list(inputs), specialized, **(compile_kwargs or {}))
