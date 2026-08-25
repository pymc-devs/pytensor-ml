import warnings

from collections.abc import Sequence

import pytensor

from pytensor import Mode
from pytensor.compile import Function, get_mode
from pytensor.tensor.variable import Variable

from pytensor_ml.pytensorf.collect import collect_clock_updates, collect_graph_inputs
from pytensor_ml.pytensorf.rewrite import hoist_scan_draws, rewrite_for_prediction
from pytensor_ml.pytensorf.rng import (
    SeedSequenceSeed,
    atleast_list,
    collect_default_updates,
    find_generators_drawn_from,
    find_rng_nodes,
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
    advance their state instead of repeating draws, and the one-step advance for every training clock the
    graph reads, so a schedule moves through time rather than reading step zero on every call.

    Parameters
    ----------
    inputs : list of Variable
        Inputs of the compiled function.
    outputs : Variable or list of Variable
        Outputs of the compiled function.
    random_seed : int, array-like of int, or SeedSequence, optional
        Seed used to replace the graph's shared generators, making the compiled function's draws
        reproducible. The generators are left as they are when omitted, so compiling has no effect on a
        generator the caller seeded, or on one another function is drawing from.
    mode : Mode or str, optional
        PyTensor mode used to compile the function.
    **kwargs
        Forwarded to :func:`pytensor.function`. An ``updates`` entry wins over the threaded RNG and clock
        updates, which is how a clock is pinned: pass ``updates={clock: clock}``.

    Returns
    -------
    compiled_function : Function
        The compiled function.

    Examples
    --------
    Compile like ``pytensor.function``, with the RNG and training-clock updates threaded in, so repeated
    calls advance their state instead of repeating the first draw. Pass ``random_seed`` to make the draws
    reproducible:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Dropout, Input, Linear, Sequential
        from pytensor_ml.pytensorf import function

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            Dropout(p=0.5),
        )
        activations = network(X)

        forward = function([X], activations, random_seed=0)
        first, second = forward(np.zeros((4, 64))), forward(np.zeros((4, 64)))
    """
    updates = dict(kwargs.pop("updates", {}))
    input_variables = [inp.variable if isinstance(inp, pytensor.In) else inp for inp in inputs]
    # Updates count as readers: a generator a rule draws its noise from is read by the update expression
    # and often by nothing else, and one read by both an output and an update is read twice.
    read_variables = [
        *(out.variable if isinstance(out, pytensor.Out) else out for out in atleast_list(outputs)),
        *updates.values(),
    ]

    # Before any generator is accounted for: a draw inside a scan has none the loop can advance, so it
    # has to come out of the loop first or there is nothing for the collection below to find.
    given_outputs = atleast_list(outputs)
    read_variables = hoist_scan_draws(read_variables)
    rewritten = [
        pytensor.Out(variable, borrow=given.borrow) if isinstance(given, pytensor.Out) else variable
        for given, variable in zip(given_outputs, read_variables)
    ]
    updates = dict(zip(updates, read_variables[len(given_outputs) :]))
    outputs = rewritten if isinstance(outputs, list | tuple) else rewritten[0]

    if random_seed is not None:
        reseed_rngs(find_rng_nodes(read_variables), random_seed)

    with warnings.catch_warnings():
        # This warns for a generator with several distinct draws and returns no update for it. The check
        # below reports that better, and only once the caller's own updates are known.
        warnings.filterwarnings(
            "ignore", message="RNG Variable .* multiple distinct clients", category=UserWarning
        )
        rng_updates = collect_default_updates(inputs=input_variables, outputs=read_variables)

    frozen = [
        generator
        for generator in find_generators_drawn_from(read_variables)
        if generator not in rng_updates and generator not in updates
    ]
    if frozen:
        raise ValueError(
            f"The graph draws from {[str(generator.name or generator) for generator in frozen]}, which "
            "nothing advances, so every call would repeat the same values. Two draws off one generator is "
            "the cause: it has no single next state, so none can be derived. Give each draw its own "
            "generator, thread one through with `next_rng, draw = pt.random.normal(rng=rng, "
            "return_next_rng=True)`, or pass an update for it yourself."
        )

    # A clock's advance is always derivable, unlike a generator's next state, so an unwritten one is
    # threaded rather than reported.
    clock_updates = collect_clock_updates(read_variables, already_written=updates)

    base_mode = get_mode(mode)
    mode = Mode(
        linker=base_mode.linker,
        optimizer=base_mode.provided_optimizer.including("random_make_inplace"),
    )

    return pytensor.function(
        inputs,
        outputs,
        updates={**rng_updates, **clock_updates, **updates},
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
    predict_function : Function
        The compiled prediction function.

    Examples
    --------
    Compile the inference pass: dropout is removed and batch norm reads its running statistics, so the
    result is deterministic and independent of the rest of the batch:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import BatchNorm, Dropout, Input, Linear, Sequential
        from pytensor_ml.model import Model
        from pytensor_ml.pytensorf import compile_predict

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            BatchNorm("bn", n_in=32),
            ReLU(),
            Dropout(p=0.5),
        )
        activations = network(X)
        Model(X, activations).initialize(seed=0)

        predict = compile_predict(activations, inputs=[X])
        predictions = predict(np.zeros((4, 64)))
    """
    specialized = rewrite_for_prediction(prediction)
    if inputs is None:
        inputs = collect_graph_inputs(specialized)
    return function(list(inputs), specialized, **(compile_kwargs or {}))
