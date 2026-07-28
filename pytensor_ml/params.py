import numpy as np

from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.type import TensorType


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
