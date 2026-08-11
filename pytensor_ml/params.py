from typing import TYPE_CHECKING

import numpy as np

from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorVariable

if TYPE_CHECKING:
    from pytensor_ml.state import Initializer


class TrainableParameter(TensorSharedVariable):
    """
    Marker class for trainable parameters (weights, biases).

    Attributes
    ----------
    initializer : Initializer or None
        The law this parameter's value is drawn from, which :func:`~pytensor_ml.state.initialize_params`
        redraws it from. Every layer declares one for each parameter it builds. None means a redraw has
        nothing to go on and raises.
    """

    initializer: "Initializer | None" = None


class NonTrainableParameter(TensorSharedVariable):
    """Marker class for non-trainable state (running mean/var in BatchNorm)."""


class StepCounter(TensorSharedVariable):
    """Training time: an integer scalar counting training steps, whose transition is :meth:`advance`."""

    def advance(self) -> TensorVariable:
        """Return the expression for this counter's value on the next training step."""
        return self + 1


def _make_parameter[T: TensorSharedVariable](
    parameter_type: type[T], value, name, shape, strict, **kwargs
) -> T:
    value = np.asarray(value)
    if shape is None:
        shape = value.shape
    ttype = TensorType(dtype=str(value.dtype), shape=shape)
    return parameter_type(name=name, type=ttype, value=value, strict=strict, **kwargs)


def trainable(
    value,
    name=None,
    shape=None,
    strict=False,
    initializer: "Initializer | None" = None,
    **kwargs,
) -> TrainableParameter:
    """
    Create a shared variable marked as a trainable parameter.

    The marker class lets graph traversal tell parameters apart from other shared state, so that an
    optimizer updates exactly these. A parameter also declares the initializer it is drawn from, which is
    what lets one seed reproduce every value in a network.

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
    initializer : Initializer, optional
        The law to redraw this parameter from, whether that is a unit scale, a zero bias, or a fan-scaled
        draw. Default None, which leaves the parameter with no law and raises on a redraw.
    **kwargs
        Additional arguments passed to the SharedVariable constructor.
    """
    parameter = _make_parameter(TrainableParameter, value, name, shape, strict, **kwargs)
    parameter.initializer = initializer
    return parameter


def non_trainable(value, name=None, shape=None, strict=False, **kwargs) -> NonTrainableParameter:
    """
    Create a shared variable marked as non-trainable state, such as batch norm's running statistics.

    Takes the same arguments as :func:`trainable`; only the marker class differs, which is what keeps these
    out of the set an optimizer updates.
    """
    return _make_parameter(NonTrainableParameter, value, name, shape, strict, **kwargs)


def step_counter(name: str = "step_count") -> StepCounter:
    """
    Create a training clock: an integer scalar counting training steps, starting at zero.

    Schedules and policies read the clock to place themselves in time, and
    :func:`~pytensor_ml.pytensorf.collect_clock_updates` advances it once per step however many of them read
    it. Hold the returned object and pass it where it is needed: readers share a clock by referring to the
    same variable, never by name.

    Parameters
    ----------
    name : str, optional
        Name for the counter. Training state is matched by name at serialization boundaries. Default
        'step_count'.
    """
    return _make_parameter(
        StepCounter, np.asarray(0, dtype="int64"), name, shape=None, strict=False
    )
