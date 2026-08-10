from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from pytensor.compile.sharedvalue import SharedVariable

from pytensor_ml.params import TrainableParameter
from pytensor_ml.pytensorf import RandomSeed

RandomState = RandomSeed | np.random.RandomState | np.random.Generator

InitializationScheme = Literal["zeros", "ones", "xavier_uniform", "xavier_normal", "unit_uniform"]

SamplingFunction = Callable[[tuple[int, ...], str, np.random.Generator], np.ndarray]


class Initializer(ABC):
    """
    Base class for parameter initializers.

    Subclasses implement :meth:`sample`. Calling an instance assigns a freshly sampled value to a
    parameter in place, while :func:`initialize_params` calls :meth:`sample` directly and leaves the
    assignment to its caller.
    """

    def __call__(self, param: SharedVariable, rng: RandomState | None = None) -> SharedVariable:
        param.set_value(self._sample_like(param, rng))
        return param

    @abstractmethod
    def sample(
        self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator
    ) -> np.ndarray: ...

    def _sample_like(self, param: SharedVariable, rng: RandomState | None = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        value = param.get_value()
        return self.sample(value.shape, str(value.dtype), rng)


class ZeroInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)


class OneInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return np.ones(shape, dtype=dtype)


class UnitUniformInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=shape).astype(dtype)


def fans(shape: tuple[int, ...]) -> tuple[int, int]:
    r"""
    Return the number of units feeding into and out of one position of a parameter of ``shape``.

    Weights here are laid out input dimension first, as :class:`~pytensor_ml.layers.Linear` builds
    ``(n_in, n_out)`` for ``X @ W`` and :class:`~pytensor_ml.layers.Embedding` builds
    ``(vocabulary, features)``, so the leading dimension is the fan-in and the second the fan-out. This is
    the transpose of torch's convention, where the output dimension leads. Any dimension past the second is
    a receptive field: every input reaches an output at each of its offsets, so both fans carry a factor of
    :math:`\prod \text{kernel}`.

    Only the sum of the two matters to a Xavier draw, which is why the orientation is invisible there and
    load-bearing for anything scaling by fan-in alone.

    Parameters
    ----------
    shape : tuple of int
        Shape of the parameter, with at least two dimensions.

    Returns
    -------
    fan_in : int
        Units feeding one output position.
    fan_out : int
        Output positions one input feeds.
    """
    if len(shape) < 2:
        raise ValueError(
            f"A fan-scaled initializer needs a parameter of at least two dimensions to size its draws, but "
            f"got shape {shape}. A bias or a norm scale has no fans; give it an initializer of its own -- "
            "`trainable(value, name, initializer=ZeroInitializer())` -- or initialize it with the 'zeros' "
            "or 'ones' scheme."
        )
    receptive_field = int(np.prod(shape[2:]))
    return shape[0] * receptive_field, shape[1] * receptive_field


class XavierUniformInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        fan_in, fan_out = fans(shape)
        scale = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-scale, scale, size=shape).astype(dtype)


class XavierNormalInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        fan_in, fan_out = fans(shape)
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return rng.normal(0, scale, size=shape).astype(dtype)


class CustomInitializer(Initializer):
    """
    Initializer built from a sampling function.

    Parameters
    ----------
    sample_fn : callable
        ``(shape, dtype, rng) -> ndarray``, returning the initial value for one parameter.
    """

    def __init__(self, sample_fn: SamplingFunction):
        self._sample_fn = sample_fn

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return self._sample_fn(shape, dtype, rng)


_INITIALIZERS: dict[str, type[Initializer]] = {
    "zeros": ZeroInitializer,
    "ones": OneInitializer,
    "xavier_uniform": XavierUniformInitializer,
    "xavier_normal": XavierNormalInitializer,
    "unit_uniform": UnitUniformInitializer,
}

InitializationSchemeLike = InitializationScheme | Initializer


def _declared_initializer(param: SharedVariable, default: Initializer) -> Initializer:
    """The parameter's own initializer, or ``default`` when it does not declare one."""
    declared = param.initializer if isinstance(param, TrainableParameter) else None
    return default if declared is None else declared


def initialize_params(
    params: Sequence[SharedVariable],
    scheme: InitializationSchemeLike = "xavier_normal",
    rng: RandomState | None = None,
) -> list[np.ndarray]:
    """
    Initialize parameter values using the specified scheme.

    A :class:`~pytensor_ml.params.TrainableParameter` that declares its own ``initializer`` uses it
    instead of ``scheme``, leaving batch norm at its unit scale while the weight matrices around it are
    drawn from the requested scheme. Call an :class:`Initializer` on a parameter directly to overwrite a
    declared value anyway.

    Parameters
    ----------
    params
        SharedVariables to initialize values for.
    scheme
        Initialization scheme for parameters that do not declare one: the name of a built-in scheme, or
        any :class:`Initializer` instance (including a :class:`CustomInitializer` wrapping your own
        sampling function).
    rng
        Random number generator. If None, a new one is created.

    Returns
    -------
    list of np.ndarray
        Initialized values matching the shapes and dtypes of params.
    """
    # Resolve once and share: a seed handed to each _sample_like call would repeat draws across parameters.
    rng = np.random.default_rng(rng)

    initializer = scheme if isinstance(scheme, Initializer) else _INITIALIZERS[scheme]()
    return [
        _declared_initializer(param, default=initializer)._sample_like(param, rng)
        for param in params
    ]
