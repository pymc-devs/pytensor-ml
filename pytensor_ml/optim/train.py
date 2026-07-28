from collections.abc import Sequence

from pytensor.compile import Function, SharedVariable
from pytensor.graph.basic import Variable
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import Parameter, UpdateRule, require_unique_state_names
from pytensor_ml.pytensorf import (
    collect_data_inputs,
    collect_non_trainable_updates,
    collect_trainable_params,
    function,
)


def compile_train(
    loss: TensorVariable,
    rule: UpdateRule,
    *,
    parameters: Sequence[Parameter] | None = None,
    inputs: Sequence[Variable] | None = None,
    compile_kwargs: dict | None = None,
) -> Function:
    """
    Compile a one-step training function from a loss graph and an update rule.

    Differentiates the loss via ``rule``, applies the resulting updates, folds in any non-trainable state
    updates (such as batch-norm running statistics), and compiles. The parameters and data inputs are
    collected from ``loss`` unless given explicitly. Returns a plain compiled function that maps a batch of
    inputs to the loss, applying every update in place.

    Parameters
    ----------
    loss : TensorVariable
        Scalar loss to minimize.
    rule : UpdateRule
        A configured optimizer ``(loss_or_gradients, parameters) -> Updates``, e.g. ``adam(1e-3)``.
    parameters : sequence of shared tensor variable, optional
        Parameters to optimize. Collected from ``loss`` with :func:`collect_trainable_params` when omitted.
    inputs : sequence of Variable, optional
        Data inputs of the compiled function, in call order. Collected from ``loss`` with
        :func:`collect_data_inputs` when omitted; pass them explicitly when call order matters (e.g. features
        before targets).
    compile_kwargs : dict, optional
        Extra keyword arguments forwarded to the function compiler.

    Returns
    -------
    Function
        The compiled one-step training function.
    """
    if parameters is None:
        parameters = collect_trainable_params(loss)
    if inputs is None:
        inputs = collect_data_inputs(loss)

    updates: dict[SharedVariable, TensorVariable] = dict(rule(loss, parameters))
    for parameter, new_value in collect_non_trainable_updates(loss).items():
        updates[parameter] = new_value

    require_unique_state_names(updates)

    return function(list(inputs), loss, updates=updates, **(compile_kwargs or {}))
