from collections.abc import Sequence

from pytensor.compile import Function, SharedVariable
from pytensor.graph.basic import Variable
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import Parameter, UpdateRule, require_unique_state_names
from pytensor_ml.pytensorf import (
    collect_clock_updates,
    collect_data_inputs,
    collect_differentiable_params,
    collect_non_trainable_updates,
    function,
)


def compile_train(
    loss: TensorVariable,
    rule: UpdateRule,
    *,
    parameters: Sequence[Parameter] | None = None,
    inputs: Sequence[Variable] | None = None,
    extra_outputs: Sequence[Variable] | None = None,
    compile_kwargs: dict | None = None,
) -> Function:
    """
    Compile a one-step training function from a loss graph and an update rule.

    Differentiates the loss via ``rule``, applies the resulting updates, folds in any non-trainable state
    updates (such as batch-norm running statistics), advances every training clock the step reads, and
    compiles. The parameters and data inputs are collected from ``loss`` unless given explicitly.

    Parameters
    ----------
    loss : TensorVariable
        Scalar loss to minimize.
    rule : UpdateRule
        A configured optimizer ``(loss_or_gradients, parameters) -> Updates``, e.g. ``adam(1e-3)``.
    parameters : sequence of shared tensor variable, optional
        Parameters to optimize. Collected from ``loss`` with :func:`collect_differentiable_params` when
        omitted, so parameters the loss detaches with a stop-gradient are left alone. A detached parameter
        is still initialized and checkpointed, since :func:`collect_trainable_params` reaches it; only the
        optimizer skips it.
    inputs : sequence of Variable, optional
        Data inputs of the compiled function, in call order. Collected from ``loss`` and ``extra_outputs``
        with :func:`collect_data_inputs` when omitted; pass them explicitly when call order matters (e.g.
        features before targets).
    extra_outputs : sequence of Variable, optional
        Diagnostics to return alongside the loss, such as gradient norms or a batch accuracy. Evaluated in
        the same pass as the gradients, so they see the pre-update parameter values, and they add no
        non-trainable state updates of their own. A random node reached only through an extra output does
        still advance its generator, since :func:`~pytensor_ml.pytensorf.function` threads the next-RNG
        update for every generator the outputs draw from.
    compile_kwargs : dict, optional
        Extra keyword arguments forwarded to the function compiler.

    Returns
    -------
    step : Function
        The compiled one-step training function, applying every update in place. Returns the loss alone, or
        ``(loss, *extra_outputs)`` when diagnostics were requested.
    """
    extra_outputs = list(extra_outputs or [])

    if parameters is None:
        parameters = collect_differentiable_params(loss)
    if inputs is None:
        inputs = collect_data_inputs([loss, *extra_outputs])

    updates: dict[SharedVariable, TensorVariable] = dict(rule(loss, parameters))

    # Assigned per key rather than merged: SupportsKeysAndGetItem is invariant in its key type, so
    # dict.update rejects the narrower NonTrainableParameter keys.
    for parameter, new_value in collect_non_trainable_updates(loss).items():
        updates[parameter] = new_value

    # Collected from the assembled updates rather than from the loss alone: a clock is read by a schedule
    # or a policy, which live in the updates, where an RNG or a running statistic is read by the model.
    for clock, next_count in collect_clock_updates(
        [loss, *extra_outputs, *updates.values()]
    ).items():
        if clock in updates:
            raise ValueError(
                f"The training clock {clock.name!r} is already updated by the rule. Clocks advance once "
                "per step on their own; drop the explicit update."
            )
        updates[clock] = next_count

    require_unique_state_names(updates)

    outputs = [loss, *extra_outputs] if extra_outputs else loss

    return function(list(inputs), outputs, updates=updates, **(compile_kwargs or {}))
