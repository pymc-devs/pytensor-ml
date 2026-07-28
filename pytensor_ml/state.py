from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from pytensor.compile.sharedvalue import SharedVariable

from pytensor_ml.pytensorf import RandomSeed

RandomState = RandomSeed | np.random.RandomState | np.random.Generator

InitializationScheme = Literal["zeros", "xavier_uniform", "xavier_normal", "unit_uniform"]

SamplingFunction = Callable[[tuple[int, ...], str, np.random.Generator], np.ndarray]


class Initializer(ABC):
    """
    Base class for parameter initializers.

    Can be used in two ways:
    - As a class: `XavierNormalInitializer(param, rng)` - directly initializes
    - As an instance: `init = XavierNormalInitializer(); init(param, rng)`
    """

    def __new__(cls, param: SharedVariable | None = None, rng: RandomState | None = None):
        # If called with a param, act as a function and initialize directly
        if param is not None:
            instance = object.__new__(cls)
            cls.__init__(instance)
            return instance(param, rng)
        # Otherwise, return an instance for later use
        return object.__new__(cls)

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


class UnitUniformInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=shape).astype(dtype)


class XavierUniformInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        scale = np.sqrt(6.0 / np.sum([x for x in shape if x is not None]))
        return rng.uniform(-scale, scale, size=shape).astype(dtype)


class XavierNormalInitializer(Initializer):
    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        scale = np.sqrt(2.0 / np.sum([x for x in shape if x is not None]))
        return rng.normal(0, scale, size=shape).astype(dtype)


class CustomInitializer(Initializer):
    def __new__(
        cls,
        sample_fn: SamplingFunction | None = None,
        param: SharedVariable | None = None,
        rng: RandomState | None = None,
    ):
        instance = object.__new__(cls)
        if sample_fn is not None:
            instance._sample_fn = sample_fn
        if param is not None:
            return instance(param, rng)
        return instance

    def __init__(self, sample_fn: SamplingFunction):
        self._sample_fn = sample_fn

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return self._sample_fn(shape, dtype, rng)


_INITIALIZERS: dict[str, type[Initializer]] = {
    "zeros": ZeroInitializer,
    "xavier_uniform": XavierUniformInitializer,
    "xavier_normal": XavierNormalInitializer,
    "unit_uniform": UnitUniformInitializer,
}


def initialize_params(
    params: Sequence[SharedVariable],
    scheme: InitializationScheme = "xavier_normal",
    rng: RandomState | None = None,
) -> list[np.ndarray]:
    """
    Initialize parameter values using the specified scheme.

    Parameters
    ----------
    params
        SharedVariables to initialize values for.
    scheme
        Initialization scheme to use.
    rng
        Random number generator. If None, a new one is created.

    Returns
    -------
    list of np.ndarray
        Initialized values matching the shapes and dtypes of params.
    """
    rng = np.random.default_rng(rng)

    initializer = _INITIALIZERS[scheme]()
    results = []
    for var in params:
        value = var.get_value()
        results.append(initializer.sample(value.shape, str(value.dtype), rng))
    return results
