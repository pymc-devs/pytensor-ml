import numpy as np

from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.type import TensorType


class TrainableParameter(TensorSharedVariable):
    """Marker class for trainable parameters (weights, biases)."""


class NonTrainableParameter(TensorSharedVariable):
    """Marker class for non-trainable state (running mean/var in BatchNorm)."""


def _make_parameter[T: TensorSharedVariable](
    parameter_type: type[T], value, name, shape, strict, **kwargs
) -> T:
    value = np.asarray(value)
    if shape is None:
        shape = value.shape
    ttype = TensorType(dtype=str(value.dtype), shape=shape)
    return parameter_type(name=name, type=ttype, value=value, strict=strict, **kwargs)


def trainable(value, name=None, shape=None, strict=False, **kwargs) -> TrainableParameter:
    """
    Create a shared variable marked as a trainable parameter.

    The marker class is the only difference from a plain pytensor shared variable. It exists so that graph
    traversal can tell parameters apart from other shared state; it adds no behavior of its own.

    Parameters
    ----------
    value : array-like
        Initial value for the parameter.
    name : str, optional
        Name for the parameter. Optimizer state and checkpoints are matched by name, so prefer giving one.
    shape : tuple, optional
        Static shape for the variable. Defaults to the concrete shape of ``value``; pass a tuple with None
        entries for dynamic dimensions, e.g. ``(None, None)`` for a fully dynamic matrix.
    strict : bool, optional
        If True, the value must exactly match the dtype.
    **kwargs
        Additional arguments passed to the SharedVariable constructor.
    """
    return _make_parameter(TrainableParameter, value, name, shape, strict, **kwargs)


def non_trainable(value, name=None, shape=None, strict=False, **kwargs) -> NonTrainableParameter:
    """
    Create a shared variable marked as non-trainable state, such as batch norm's running statistics.

    Takes the same arguments as :func:`trainable`; only the marker class differs, which is what keeps these
    out of the set an optimizer updates.
    """
    return _make_parameter(NonTrainableParameter, value, name, shape, strict, **kwargs)
