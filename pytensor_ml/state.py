from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np

from pytensor import config
from pytensor.compile.sharedvalue import SharedVariable

from pytensor_ml.params import TrainableParameter
from pytensor_ml.pytensorf import RandomSeed

RandomState = RandomSeed | np.random.RandomState | np.random.Generator

InitializationScheme = Literal[
    "zeros", "ones", "xavier_uniform", "xavier_normal", "unit_uniform", "normal"
]

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

    def initial_value(self, shape: tuple[int, ...]) -> np.ndarray:
        """
        Draw the value a parameter of ``shape`` is born holding, at the current ``floatX``.

        Uses fresh entropy: reproducibility comes from :meth:`~pytensor_ml.model.Model.initialize`, which
        redraws every parameter from one seed and discards whatever this produced.

        Parameters
        ----------
        shape : tuple of int
            Shape of the parameter to draw.
        """
        return self.sample(shape, config.floatX, np.random.default_rng())

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


class NormalInitializer(Initializer):
    r"""
    Draw every element from :math:`\mathcal{N}(\mu, \sigma^2)`, independent of the parameter's shape.

    The fan-scaled initializers derive their spread from the shape; this one is told it, which is what a
    reference implementation quoting a specific standard deviation needs -- GPT-2 initializes its embeddings
    and weights from ``NormalInitializer(0.0, 0.02)`` whatever their fans work out to.

    Parameters
    ----------
    mean : float
        Center of the distribution :math:`\mu`. Default 0.0.
    std : float
        Standard deviation :math:`\sigma`. Default 0.01.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.std, size=shape).astype(dtype)


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
            "`trainable(value, name, initializer=ZeroInitializer())`."
        )
    receptive_field = int(np.prod(shape[2:]))
    return shape[0] * receptive_field, shape[1] * receptive_field


class XavierUniformInitializer(Initializer):
    r"""
    Draw from :math:`\mathcal{U}(\pm\sqrt{6 / (\text{fan\_in} + \text{fan\_out})})`.

    The bound is chosen so the variance of the activations, and of the gradients flowing back, stays roughly
    constant through depth. Also called Glorot initialization.

    References
    ----------
    .. [1] Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward
           neural networks. Proceedings of AISTATS, 249-256.
    """

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        fan_in, fan_out = fans(shape)
        scale = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-scale, scale, size=shape).astype(dtype)


class XavierNormalInitializer(Initializer):
    r"""
    Draw from :math:`\mathcal{N}(0, 2 / (\text{fan\_in} + \text{fan\_out}))`.

    The normal counterpart of :class:`XavierUniformInitializer`, targeting the same variance.

    References
    ----------
    .. [1] Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward
           neural networks. Proceedings of AISTATS, 249-256.
    """

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


# The name of each initializer, for the config a network is serialized to: an initializer crosses that
# boundary as a name, since a class cannot. Every entry must construct with no arguments to come back.
_INITIALIZERS: dict[str, type[Initializer]] = {
    "zeros": ZeroInitializer,
    "ones": OneInitializer,
    "xavier_uniform": XavierUniformInitializer,
    "xavier_normal": XavierNormalInitializer,
    "unit_uniform": UnitUniformInitializer,
    "normal": NormalInitializer,
}


def _initializer_for(
    param: SharedVariable, initializers: Mapping[SharedVariable, Initializer]
) -> Initializer:
    """The initializer to draw ``param`` with: one named for it, else the one it declares."""
    if param in initializers:
        return initializers[param]

    declared = param.initializer if isinstance(param, TrainableParameter) else None
    if declared is None:
        raise ValueError(
            f"{param.name!r} declares no initializer, so there is no law to redraw it from. Every layer in "
            "this library declares one for each parameter it builds; a parameter built by hand needs the "
            "same -- `trainable(value, name, initializer=XavierNormalInitializer())` -- or an entry naming "
            "it in `initializers={parameter: XavierNormalInitializer()}`."
        )
    return declared


def initialize_params(
    params: Sequence[SharedVariable],
    rng: RandomState | None = None,
    initializers: Mapping[SharedVariable, Initializer] | None = None,
) -> list[np.ndarray]:
    """
    Redraw each parameter from its own initializer, or from the one ``initializers`` names for it.

    Every parameter carries the law its value comes from, so there is nothing to decide here beyond the
    seed: a batch norm layer redraws to its unit scale, a bias to zero, and a weight matrix to a fresh
    Xavier draw. Pass ``initializers`` to draw a particular parameter some other way.

    Parameters
    ----------
    params : sequence of SharedVariable
        Parameters to draw values for. Each must declare an initializer or be named in ``initializers``.
    rng : int or numpy Generator, optional
        Random number generator. A fresh one is created when omitted.
    initializers : dict mapping parameter to Initializer, optional
        How to draw specific parameters, taking precedence over what they declare. Keyed by the parameter
        itself rather than its name, so nothing here depends on how a layer names what it builds.

    Returns
    -------
    list of ndarray
        Drawn values, matching the shapes and dtypes of ``params``.
    """
    # Resolve once and share: a seed handed to each _sample_like call would repeat draws across parameters.
    rng = np.random.default_rng(rng)
    if initializers is None:
        initializers = {}

    return [_initializer_for(param, initializers)._sample_like(param, rng) for param in params]
