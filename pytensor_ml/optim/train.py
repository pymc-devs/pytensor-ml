from collections.abc import Sequence

from pytensor.compile import Function
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Variable, equal_computations
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import (
    Gradients,
    Parameter,
    Steps,
    Transform,
    Updates,
    require_unique_state_names,
)
from pytensor_ml.pytensorf import (
    collect_clock_updates,
    collect_data_inputs,
    collect_differentiable_params,
    collect_non_trainable_updates,
    function,
)
from pytensor_ml.pytensorf.rewrite import hoist_scan_draws
from pytensor_ml.pytensorf.rng import collect_generator_updates


def _inconsistent_update(updates: Updates, variable: SharedVariable, new_value: Variable) -> bool:
    """
    Report whether ``updates`` already writes ``variable`` with an expression other than ``new_value``.

    Two components writing one variable is only a problem when they disagree: a plain merge keeps one write
    and drops the other, silently, leaving something that looks configured and is inert. Writers that agree
    structurally are fine, which is how several components share one quantity.
    """
    written = updates.get(variable)
    return written is not None and not equal_computations([written], [new_value])


def build_train_updates(
    loss: TensorVariable,
    rule: Transform | Updates,
    *,
    parameters: Sequence[Parameter] | None = None,
    inputs: Sequence[Variable] | None = None,
    extra_outputs: Sequence[Variable] | None = None,
    extra_updates: dict[SharedVariable, TensorVariable] | None = None,
) -> tuple[TensorVariable, dict[SharedVariable, Variable]]:
    """
    Assemble and check the updates of one training step, without compiling them.

    Does what :func:`compile_train` does up to its compiler call. It differentiates the loss via ``rule``,
    folds in non-trainable state updates and ``extra_updates``, adds the advance of every training clock and
    generator the step reads, and refuses a step whose writes collide or leave a parameter behind. Use it to
    compile with another compiler or mode, or to list what a step writes for a checkpoint.

    A draw inside a scan is lifted out of the loop in the returned graphs, while ``loss`` keeps the original
    loop. Compile the returned loss, because ``loss`` reports different noise than the step. ``extra_outputs``
    are not returned, so a diagnostic with such a draw needs :func:`~pytensor_ml.pytensorf.rewrite_pregrad`
    before compiling. Only :func:`~pytensor_ml.pytensorf.function` carries a batch-norm statistic through a
    scan that writes it, so the returned updates leave that statistic out.

    Parameters
    ----------
    loss : TensorVariable
        Scalar loss to minimize.
    rule : Transform or Updates
        A configured optimizer, or the updates dict one invocation of it returned, as in
        :func:`compile_train`.
    parameters : sequence of shared tensor variable, optional
        Parameters to optimize. Collected from ``loss`` with :func:`collect_differentiable_params` when
        omitted, as in :func:`compile_train`.
    inputs : sequence of Variable, optional
        Data inputs of the compiled function. A generator read only above one of these is not advanced.
        Default is the graph roots.
    extra_outputs : sequence of Variable, optional
        Diagnostics the step will return alongside the loss. The only updates they contribute are the advance
        of a training clock or a generator that only they read.
    extra_updates : dict, optional
        State the step should write that no gradient produces, as in :func:`compile_train`.

    Returns
    -------
    step_loss : TensorVariable
        The loss to compile, reading the same draws as the updates.
    updates : dict
        Mapping from each variable the step writes to its next value. Pass it to any compiler as the
        function's ``updates``. Its keys are the step's state, clocks and generators included.

    Examples
    --------
    Compile the assembled step with PyTensor directly, and keep the list of variables it writes for a
    checkpoint:

    .. code-block:: python

        import numpy as np
        import pytensor

        from pytensor_ml.optim import adam, build_train_updates
        from pytensor_ml.params import trainable

        parameter = trainable(np.array([2.0]), name="w")
        loss = (parameter**2).sum()

        step_loss, updates = build_train_updates(loss, adam(1e-2), parameters=[parameter])
        state = list(updates)
        step = pytensor.function([], step_loss, updates=updates)
    """
    extra_outputs = list(extra_outputs or [])
    extra_updates = dict(extra_updates or {})

    if parameters is None:
        parameters = collect_differentiable_params(loss)

    result = rule(loss, parameters) if callable(rule) else rule
    if isinstance(result, Gradients):
        raise ValueError(
            "The rule returned gradients rather than the steps to take, so every parameter would move "
            "along its gradient -- uphill, away from a minimum. Put an optimizer such as `adam(1e-3)` in "
            "the chain, or `scale(-learning_rate)` to descend along the gradients yourself."
        )
    updates: Updates = Steps(result)

    # Assigned per key rather than merged: SupportsKeysAndGetItem is invariant in its key type, so
    # dict.update rejects the narrower NonTrainableParameter keys.
    for parameter, new_value in collect_non_trainable_updates(loss).items():
        if _inconsistent_update(updates, parameter, new_value):
            raise ValueError(
                f"The model writes {parameter.name!r} from a stateful op, and the rule writes it too, so "
                "the two writes cannot both take effect. A batch-norm statistic is the model's to update; "
                "leave it out of the rule."
            )
        updates[parameter] = new_value

    # Folded in after the rule's and the model's, so a collision with either is caught rather than deciding
    # by insertion order which of the two writes survives.
    for variable, new_value in extra_updates.items():
        if _inconsistent_update(updates, variable, new_value):
            raise ValueError(
                f"The extra update for {variable.name!r} writes a variable the training step already "
                "writes differently, so the two writes cannot both take effect. Optimizer state and "
                "batch-norm statistics are written by the step itself; drop this one, or fold what it does "
                "into the expression that already writes the variable."
            )
        updates[variable] = new_value

    # Collected from the assembled updates rather than from the loss alone: a clock is read by a schedule
    # or a policy, which live in the updates, where an RNG or a running statistic is read by the model.
    # Assigned per key rather than merged, for the same invariance reason as the statistics above.
    for clock, next_count in collect_clock_updates(
        [loss, *extra_outputs, *updates.values()], already_written=updates
    ).items():
        updates[clock] = next_count

    unwritten = [parameter.name for parameter in parameters if parameter not in updates]
    if unwritten:
        raise ValueError(
            f"No update reaches {unwritten}. Every parameter collected from the loss, or given in "
            "`parameters`, needs an entry in the updates, so an updates dict built over a subset of them "
            "would silently freeze the rest. Leave a parameter out of `parameters` to freeze it on purpose."
        )

    require_unique_state_names(updates)

    # A generator a loop cannot advance has no next state until its draw is out of the loop.
    read_variables = hoist_scan_draws([loss, *extra_outputs, *updates.values()])
    assembled = dict(zip(updates, read_variables[1 + len(extra_outputs) :]))

    return read_variables[0], {
        **assembled,
        **collect_generator_updates(read_variables, inputs=inputs, already_written=assembled),
    }


def compile_train(
    loss: TensorVariable,
    rule: Transform | Updates,
    *,
    parameters: Sequence[Parameter] | None = None,
    inputs: Sequence[Variable] | None = None,
    extra_outputs: Sequence[Variable] | None = None,
    extra_updates: dict[SharedVariable, TensorVariable] | None = None,
    compile_kwargs: dict | None = None,
) -> Function:
    """
    Compile a one-step training function from a loss graph and an update rule.

    Differentiates the loss via ``rule``, applies the resulting updates, folds in any non-trainable state
    updates (such as batch-norm running statistics) and any given in ``extra_updates``, advances every
    training clock the step reads, and compiles. The parameters and data inputs are collected from ``loss``
    unless given explicitly. :func:`build_train_updates` is the same assembly without the compilation.

    Parameters
    ----------
    loss : TensorVariable
        Scalar loss to minimize.
    rule : Transform or Updates
        A configured optimizer ``(loss_gradients_or_updates, parameters) -> Updates``, e.g. ``adam(1e-3)``,
        or the updates dict one invocation of it returned. A rule allocates its state afresh on every
        invocation, so two training functions that should share momentum are compiled from one dict:
        ``updates = adam(1e-3)(loss, parameters)``, then ``compile_train(loss, updates, ...)`` twice.
    parameters : sequence of shared tensor variable, optional
        Parameters to optimize. Collected from ``loss`` with :func:`collect_differentiable_params` when
        omitted, so parameters the loss detaches with a stop-gradient are left alone. A detached parameter
        is still initialized and checkpointed, since :func:`collect_trainable_params` reaches it; only the
        optimizer skips it.
    inputs : sequence of Variable, optional
        Data inputs of the compiled function, in call order. Collected from ``loss``, ``extra_outputs`` and
        ``extra_updates`` with :func:`collect_data_inputs` when omitted; pass them explicitly when call order
        matters (e.g. features before targets).
    extra_outputs : sequence of Variable, optional
        Diagnostics to return alongside the loss, such as gradient norms or a batch accuracy. Evaluated in
        the same pass as the gradients, so they see the pre-update parameter values, and they add no
        non-trainable state updates of their own. A random node reached only through an extra output does
        still advance its generator, since :func:`~pytensor_ml.pytensorf.function` threads the next-RNG
        update for every generator the outputs draw from.
    extra_updates : dict, optional
        State the step should write that no gradient produces -- a target-network sync, a Polyak average,
        replay priorities. Mapping from shared variable to its next value, folded in alongside the rule's own
        updates. Raise if the rule or the model already writes one of these variables, since two writes to one
        variable cannot both take effect. An expression here is part of the step like any other, so a clock it
        reads still advances once, and a generator it draws from still advances.
    compile_kwargs : dict, optional
        Extra keyword arguments forwarded to the function compiler. An ``updates`` entry is taken as
        ``extra_updates`` rather than forwarded, since the compiler's own ``updates`` carries the whole
        assembled step. Raise if a variable is given in both.

    Returns
    -------
    step : Function
        The compiled one-step training function, applying every update in place. Returns the loss alone, or
        ``(loss, *extra_outputs)`` when diagnostics were requested.

    Examples
    --------
    Hand it any scalar loss graph and a rule, and it differentiates the loss, applies the rule, and folds in
    every update the graph carries -- optimizer state, batch-norm statistics, RNGs, training clocks. Pass
    ``extra_outputs`` to read a value out alongside the loss:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, compile_train

        X = Input("X", shape=(None, 4))
        prediction = Linear("fc", n_in=4, n_out=1)(X)
        loss, target = supervised_loss(prediction, SquaredError())

        step = compile_train(loss, adam(1e-3), inputs=[X, target], extra_outputs=[prediction])
        loss_value, predictions = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    extra_outputs = list(extra_outputs or [])
    extra_updates = dict(extra_updates or {})
    # Copied rather than mutated, so popping the caller's `updates` out does not empty their dict.
    compile_kwargs = dict(compile_kwargs or {})

    # `updates` is pytensor's own name for this, so a caller who puts one in compile_kwargs is asking for
    # what extra_updates does; take it as one instead of letting it collide with this function's own call.
    for variable, new_value in compile_kwargs.pop("updates", {}).items():
        if variable in extra_updates:
            raise ValueError(
                f"The update for {variable.name!r} is given twice, once in `extra_updates` and once in "
                "`compile_kwargs['updates']`. Both are folded into the training step, so keep whichever one "
                "is right and drop the other."
            )
        extra_updates[variable] = new_value

    if inputs is None:
        inputs = collect_data_inputs([loss, *extra_outputs, *extra_updates.values()])

    step_loss, updates = build_train_updates(
        loss,
        rule,
        parameters=parameters,
        inputs=inputs,
        extra_outputs=extra_outputs,
        extra_updates=extra_updates,
    )
    outputs = [step_loss, *extra_outputs] if extra_outputs else step_loss

    return function(list(inputs), outputs, updates=updates, **compile_kwargs)
