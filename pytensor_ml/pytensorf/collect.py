from collections.abc import Sequence

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph import graph_inputs
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.traversal import ancestors
from pytensor.tensor import TensorVariable

from pytensor_ml.base import StatefulOp
from pytensor_ml.params import NonTrainableParameter, TrainableParameter


def as_output_list(outputs: Variable | Sequence[Variable]) -> list[Variable]:
    """Normalize one output, or a sequence of them, to a list."""
    return [outputs] if isinstance(outputs, Variable) else list(outputs)


def _collect_inputs_of_type[T: Variable](
    outputs: Variable | Sequence[Variable], variable_type: type[T]
) -> list[T]:
    return [
        variable
        for variable in graph_inputs(as_output_list(outputs))
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


def collect_non_trainable_params(
    outputs: Variable | Sequence[Variable],
) -> list[NonTrainableParameter]:
    """Collect the state that training updates without gradients, such as batch-norm running statistics."""
    return _collect_inputs_of_type(outputs, NonTrainableParameter)


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
    "collect_data_inputs",
    "collect_graph_inputs",
    "collect_non_trainable_params",
    "collect_non_trainable_updates",
    "collect_shared_variables",
    "collect_trainable_params",
]
