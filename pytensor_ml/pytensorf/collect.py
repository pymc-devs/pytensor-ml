from collections.abc import Container, Sequence

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.gradient import DisconnectedGrad, ZeroGrad
from pytensor.graph import graph_inputs
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.op import io_connection_pattern
from pytensor.graph.traversal import ancestors
from pytensor.tensor import TensorVariable

from pytensor_ml.base import StatefulOp
from pytensor_ml.params import NonTrainableParameter, StepCounter, TrainableParameter


def as_output_list(outputs: Variable | Sequence[Variable]) -> list[Variable]:
    """Normalize one output, or a sequence of them, to a list."""
    return [outputs] if isinstance(outputs, Variable) else list(outputs)


def _collect_inputs_of_type[T: Variable](
    outputs: Variable | Sequence[Variable],
    variable_type: type[T],
    blockers: Sequence[Variable] | None = None,
) -> list[T]:
    return [
        variable
        for variable in graph_inputs(as_output_list(outputs), blockers=blockers)
        if isinstance(variable, variable_type)
    ]


def collect_graph_inputs(outputs: Variable | Sequence[Variable]) -> list[Variable]:
    """Collect the graph inputs that carry data -- everything that is neither a Constant nor shared."""
    return [
        variable
        for variable in graph_inputs(as_output_list(outputs))
        if not isinstance(variable, Constant | SharedVariable)
    ]


# Same set, named for how the training and serialization boundaries talk about it.
collect_data_inputs = collect_graph_inputs


def collect_shared_variables(outputs: Variable | Sequence[Variable]) -> list[SharedVariable]:
    """Collect every SharedVariable the graph reads, parameters and RNGs alike."""
    return _collect_inputs_of_type(outputs, SharedVariable)


def collect_trainable_params(outputs: Variable | Sequence[Variable]) -> list[TrainableParameter]:
    """Collect the parameters an optimizer should update."""
    return _collect_inputs_of_type(outputs, TrainableParameter)


def collect_differentiable_params(
    outputs: Variable | Sequence[Variable],
) -> list[TrainableParameter]:
    """
    Collect the trainable parameters an optimizer can take gradients of ``outputs`` with respect to.

    Two things disqualify a parameter, and neither subsumes the other. A stop-gradient marker on the only
    path to it -- :func:`pytensor.gradient.disconnected_grad` or :func:`pytensor.gradient.zero_grad` --
    though a parameter reached on both a detached and a live path stays, since the live path still carries
    gradient. Or no connection to the gradient at all, which no marker expresses: an additive constant
    survives in ``outputs`` as written and vanishes once they differentiate, the shape a physics-informed
    loss hits when an output bias is gone by the second derivative.

    Both checks are needed because ``zero_grad`` yields a gradient that is connected and zero, which the
    connection pattern reports as present.

    Parameters
    ----------
    outputs : Variable or sequence of Variable
        One or more graph outputs to trace back from, typically a scalar loss.

    Returns
    -------
    list of TrainableParameter
        The differentiable parameters, in graph-input order.
    """
    output_list = as_output_list(outputs)
    stop_gradient_outputs = [
        variable
        for variable in ancestors(output_list)
        if variable.owner is not None and isinstance(variable.owner.op, DisconnectedGrad | ZeroGrad)
    ]
    undetached = _collect_inputs_of_type(
        outputs, TrainableParameter, blockers=stop_gradient_outputs
    )
    if not undetached:
        return undetached

    connection_pattern = io_connection_pattern(list(undetached), output_list)
    return [
        parameter
        for parameter, to_outputs in zip(undetached, connection_pattern)
        if any(to_outputs)
    ]


def collect_non_trainable_params(
    outputs: Variable | Sequence[Variable],
) -> list[NonTrainableParameter]:
    """Collect the state that training updates without gradients, such as batch-norm running statistics."""
    return _collect_inputs_of_type(outputs, NonTrainableParameter)


def collect_step_counters(outputs: Variable | Sequence[Variable]) -> list[StepCounter]:
    """Collect the training clocks the graph reads."""
    return _collect_inputs_of_type(outputs, StepCounter)


def collect_clock_updates(
    outputs: Variable | Sequence[Variable],
    already_written: Container[SharedVariable] = (),
) -> dict[StepCounter, TensorVariable]:
    """
    Collect the advance for every training clock the graph reads, ready to pass as a function's ``updates``.

    A clock advances once per step however many schedules, policies, and diagnostics read it, because
    readers share a clock by referring to the same variable rather than by name.

    Parameters
    ----------
    outputs
        One or more graph outputs to trace back from.
    already_written : container of shared variable, optional
        The variables the caller writes themselves, typically the ``updates`` mapping itself. Any clock
        among them is left out of the result and exempt from the rule that every clock agrees on its step
        count, since a clock this step does not advance is not counting its steps. Default is empty, which
        collects and checks every clock the graph reads.

    Returns
    -------
    clock_updates : dict
        Mapping from each clock the graph reads to the expression for its next value.
    """
    counters = [
        counter for counter in collect_step_counters(outputs) if counter not in already_written
    ]
    step_counts = {int(counter.get_value()) for counter in counters}
    if len(step_counts) > 1:
        raise ValueError(
            f"Training clocks {sorted(str(counter.name) for counter in counters)} hold different step "
            f"counts {sorted(step_counts)}. They all count training steps, so this usually means a "
            "checkpoint restored some of them and not the others. Restore all of them, or set them to the "
            "same count before compiling."
        )
    return {counter: counter.advance() for counter in counters}


def collect_non_trainable_updates(
    outputs: Variable | Sequence[Variable],
) -> dict[NonTrainableParameter, TensorVariable]:
    """
    Collect the write-backs that stateful ops declare, ready to pass as a function's ``updates``.

    Parameters
    ----------
    outputs
        One or more graph outputs to trace back from.

    Returns
    -------
    non_trainable_updates : dict
        Mapping from each NonTrainableParameter to its new value, for every update an op declares through
        :meth:`~pytensor_ml.base.StatefulOp.update_map`.
    """
    updates: dict[NonTrainableParameter, TensorVariable] = {}
    for ancestor in ancestors(as_output_list(outputs)):
        node = ancestor.owner
        if node is not None and isinstance(node.op, StatefulOp):
            for output_index, input_index in node.op.update_map().items():
                old_value = node.inputs[input_index]
                if isinstance(old_value, NonTrainableParameter):
                    updates[old_value] = node.outputs[output_index]

    return updates


__all__ = [
    "as_output_list",
    "collect_clock_updates",
    "collect_data_inputs",
    "collect_differentiable_params",
    "collect_graph_inputs",
    "collect_non_trainable_params",
    "collect_non_trainable_updates",
    "collect_shared_variables",
    "collect_step_counters",
    "collect_trainable_params",
]
