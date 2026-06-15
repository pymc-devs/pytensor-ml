from collections.abc import Sequence

import numpy as np

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph import graph_inputs
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.traversal import ancestors
from pytensor.tensor import TensorVariable
from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.type import TensorType

from pytensor_ml.pytensorf import LayerOp


class TrainableParameter(TensorSharedVariable):
    """Marker class for trainable parameters (weights, biases)."""


class NonTrainableParameter(TensorSharedVariable):
    """Marker class for non-trainable state (running mean/var in BatchNorm)."""


def trainable(value, name=None, shape=None, strict=False, **kwargs) -> TrainableParameter:
    """
    Create a trainable parameter SharedVariable.

    These are simply pytensor SharedVariables that are marked with a different class for easy identification during
    graph traversal. They do not do anything special on their own.

    Parameters
    ----------
    value : array-like
        Initial value for the parameter.
    name : str, optional
        Name for the parameter.
    shape : tuple, optional
        Static shape for the variable. If None, uses the concrete shape from value. Pass a tuple with None elements
        for dynamic dimensions, e.g., shape=(None, None) for fully dynamic 2D tensor.
    strict : bool, optional
        If True, the value must exactly match the dtype.
    **kwargs
        Additional arguments passed to the SharedVariable constructor.
    """
    value = np.asarray(value)
    if shape is None:
        shape = value.shape
    ttype = TensorType(dtype=str(value.dtype), shape=shape)
    return TrainableParameter(name=name, type=ttype, value=value, strict=strict, **kwargs)


def non_trainable(value, name=None, shape=None, strict=False, **kwargs) -> NonTrainableParameter:
    """
    Create a non-trainable parameter SharedVariable.

    These are simply pytensor SharedVariables that are marked with a different class for easy identification during
    graph traversal. They do not do anything special on their own.

    Parameters
    ----------
    value : array-like
        Initial value for the parameter.
    name : str, optional
        Name for the parameter.
    shape : tuple, optional
        Static shape for the variable. If None, uses the concrete shape from value. Pass a tuple with None elements
        for dynamic dimensions, e.g., shape=(None, None) for fully dynamic 2D tensor.
    strict : bool, optional
        If True, the value must exactly match the dtype.
    **kwargs
        Additional arguments passed to the SharedVariable constructor.
    """
    value = np.asarray(value)
    if shape is None:
        shape = value.shape
    ttype = TensorType(dtype=str(value.dtype), shape=shape)
    return NonTrainableParameter(name=name, type=ttype, value=value, strict=strict, **kwargs)


def collect_graph_inputs(
    outputs: TensorVariable | Sequence[TensorVariable],
) -> list[TensorVariable]:
    """
    Collect all non-constant, non-shared graph inputs.

    Parameters
    ----------
    outputs : TensorVariable or Sequence of TensorVariable
        One or more graph outputs to trace back from.

    Returns
    -------
    graph_inputs : list of TensorVariable
        All graph inputs that are not Constants or SharedVariables.
    """
    if isinstance(outputs, Variable):
        outputs = [outputs]

    return list(
        filter(
            lambda var: not isinstance(var, Constant | SharedVariable),
            graph_inputs(list(outputs)),
        )
    )


def collect_shared_variables(
    outputs: Variable | Sequence[Variable],
) -> list[SharedVariable]:
    """
    Collect all SharedVariables from a computation graph.

    Parameters
    ----------
    outputs : TensorVariable or Sequence of TensorVariable
        One or more graph outputs to trace back from.

    Returns
    -------
    shared_vars : list of SharedVariable
        All SharedVariables in the graph.
    """
    if isinstance(outputs, Variable):
        outputs = [outputs]

    return [var for var in graph_inputs(list(outputs)) if isinstance(var, SharedVariable)]


def collect_trainable_params(
    outputs: TensorVariable | Sequence[TensorVariable],
) -> list[TrainableParameter]:
    """
    Extract trainable parameters from a computation graph.

    Parameters
    ----------
    outputs : TensorVariable or Sequence of TensorVariable
        One or more graph outputs to trace back from.

    Returns
    -------
    trainable_params : list of TrainableParameter
        All TrainableParameter SharedVariables in the graph.
    """
    if isinstance(outputs, Variable):
        outputs = [outputs]

    result: list[TrainableParameter] = [
        var for var in graph_inputs(list(outputs)) if isinstance(var, TrainableParameter)
    ]
    return result


def collect_non_trainable_params(
    outputs: TensorVariable | Sequence[TensorVariable],
) -> list[NonTrainableParameter]:
    """
    Extract non-trainable parameters from a computation graph.

    Parameters
    ----------
    outputs : TensorVariable or Sequence of TensorVariable
        One or more graph outputs to trace back from.

    Returns
    -------
    non_trainable_params : list of NonTrainableParameter
        All NonTrainableParameter SharedVariables in the graph.
    """
    if isinstance(outputs, Variable):
        outputs = [outputs]

    return [var for var in graph_inputs(list(outputs)) if isinstance(var, NonTrainableParameter)]


def collect_non_trainable_updates(
    outputs: TensorVariable | Sequence[TensorVariable],
) -> dict[NonTrainableParameter, TensorVariable]:
    """
    Extract non-trainable update pairs from LayerOps.

    These are state variables that need to be updated during training but are not subject to gradient-based
    optimization (e.g., running mean/variance in batch normalization).

    Parameters
    ----------
    outputs
        One or more graph outputs to trace back from.

    Returns
    -------
    non_trainable_updates : dict
        Mapping from NonTrainableParameter to its new value for all updates declared by LayerOp.update_map().
    """
    if isinstance(outputs, Variable):
        outputs = [outputs]

    updates: dict[NonTrainableParameter, TensorVariable] = {}
    for ancestor in ancestors(list(outputs)):
        node = ancestor.owner
        if node is not None and isinstance(node.op, LayerOp):
            for output_idx, input_idx in node.op.update_map().items():
                old_value = node.inputs[input_idx]
                new_value = node.outputs[output_idx]
                if isinstance(old_value, NonTrainableParameter):
                    updates[old_value] = new_value

    return updates


def collect_data_inputs(
    outputs: Variable | Sequence[Variable],
) -> list[TensorVariable]:
    """
    Extract data inputs from a graph (inputs that are not SharedVariables).

    Parameters
    ----------
    outputs : TensorVariable or Sequence of TensorVariable
        One or more graph outputs to trace back from.

    Returns
    -------
    data_inputs : list of TensorVariable
        Graph inputs that are not SharedVariables (i.e., data like X, y_true).
    """
    return collect_graph_inputs(outputs)
