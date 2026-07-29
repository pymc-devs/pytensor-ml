import numpy as np
import pytensor.tensor as pt

from pytensor import config

from pytensor_ml.base import Layer


def _constant_like(value: float, x: pt.TensorVariable) -> pt.TensorVariable:
    """
    Wrap a scalar so that combining it with ``x`` cannot widen ``x``'s dtype.

    PyTensor's autocaster types a bare Python float by value, so whether a literal widens its operand
    depends on that value: against a float32 input ``0.5 * x`` stays float32, while ``0.01 * x``
    promotes to float64. Pinning the constant to ``x``'s dtype removes the dependence.
    """
    dtype = np.dtype(x.dtype)
    # np.finfo maps complex64 -> float32, keeping complex inputs at their own precision.
    dtype = np.finfo(dtype).dtype if np.issubdtype(dtype, np.inexact) else np.dtype(config.floatX)
    return pt.constant(np.asarray(value, dtype=dtype))


class Activation(Layer): ...


class ReLU(Activation):
    r"""
    Rectified Linear Unit.

    Compute the positive part of the input:

    .. math::

        \mathrm{ReLU}(x) = \max(0, x).
    """

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.maximum(0, x)
        out.name = "ReLU"
        return out


class LeakyReLU(Activation):
    r"""
    Leaky Rectified Linear Unit.

    Replace ReLU's flat negative branch with a small negative slope, so negative inputs keep a
    nonzero gradient:

    .. math::

        \mathrm{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \le 0 \end{cases}

    Parameters
    ----------
    negative_slope : float, optional
        The slope :math:`\alpha` applied to negative inputs. Default is 0.01.
    """

    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        x = pt.as_tensor(x)
        out = pt.switch(x > 0, x, _constant_like(self.negative_slope, x) * x)
        out.name = "LeakyReLU"
        return out


class Tanh(Activation):
    r"""
    Hyperbolic tangent.

    Squash the input to :math:`(-1, 1)`:

    .. math::

        \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}.
    """

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.tanh(x)
        out.name = "TanH"
        return out


class Sigmoid(Activation):
    r"""
    Logistic sigmoid.

    Squash the input to :math:`(0, 1)`:

    .. math::

        \sigma(x) = \frac{1}{1 + e^{-x}}.
    """

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.sigmoid(x)
        out.name = "Sigmoid"
        return out


class SoftPlus(Activation):
    r"""
    Softplus activation.

    Compute a smooth approximation to ReLU:

    .. math::

        \mathrm{softplus}(x) = \log(1 + e^x).
    """

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.softplus(x)
        out.name = "SoftPlus"
        return out


class GELU(Activation):
    r"""
    Gaussian Error Linear Unit.

    Compute :math:`\mathrm{GELU}(x) = x \, \Phi(x)`, where :math:`\Phi` is the standard normal
    cumulative distribution function:

    .. math::

        \mathrm{GELU}(x) = \frac{x}{2} \left(1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right).

    Parameters
    ----------
    approximate : bool, optional
        Use the tanh approximation

        .. math::

            \mathrm{GELU}(x) \approx \frac{x}{2}
            \left(1 + \tanh\!\left[\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right]\right)

        This is the variant HuggingFace calls ``"gelu_new"`` / ``"gelu_pytorch_tanh"``, PyTorch exposes
        as ``nn.GELU(approximate="tanh")``, and Flax as ``gelu(approximate=True)``; GPT-2 uses it. It is
        cheaper to evaluate than the exact :math:`\operatorname{erf}` form. Default is True.
    """

    def __init__(self, approximate: bool = True):
        self.approximate = approximate

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        x = pt.as_tensor(x)
        half = _constant_like(0.5, x)
        one = _constant_like(1.0, x)
        if self.approximate:
            sqrt_2_over_pi = _constant_like(np.sqrt(2.0 / np.pi), x)
            cubic_coef = _constant_like(0.044715, x)
            out = half * x * (one + pt.tanh(sqrt_2_over_pi * (x + cubic_coef * x**3)))
        else:
            sqrt2 = _constant_like(np.sqrt(2.0), x)
            out = half * x * (one + pt.erf(x / sqrt2))
        out.name = "GELU"
        return out


class Swish(Activation):
    r"""
    Swish activation, also known as SiLU.

    Compute :math:`\mathrm{Swish}(x) = x \, \sigma(\beta x)`, where :math:`\sigma` is the logistic
    sigmoid. With :math:`\beta = 1` this is the Sigmoid Linear Unit (PyTorch ``nn.SiLU``, HuggingFace
    ``"silu"`` / ``"swish"``).

    Parameters
    ----------
    beta : float, optional
        Slope of the sigmoid gate. Larger :math:`\beta` sharpens the gate toward a ReLU; :math:`\beta
        \to 0` collapses it toward the linear map :math:`x/2`. Default is 1.0.
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        x = pt.as_tensor(x)
        out = x * pt.sigmoid(_constant_like(self.beta, x) * x)
        out.name = "Swish"
        return out


class Softmax(Activation):
    r"""
    Softmax activation.

    Normalize the input along one axis into a probability distribution:

    .. math::

        \mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}.

    Parameters
    ----------
    axis : int, optional
        The axis along which the values sum to one. Default is -1.
    """

    def __init__(self, axis: int = -1):
        self.axis = axis

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.special.softmax(x, axis=self.axis)
        out.name = "Softmax"
        return out


__all__ = [
    "GELU",
    "Activation",
    "LeakyReLU",
    "ReLU",
    "Sigmoid",
    "SoftPlus",
    "Softmax",
    "Swish",
    "Tanh",
]
