import numpy as np
import pytensor.tensor as pt

from pytensor_ml.layers import Layer


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
        out = pt.switch(x > 0, x, -self.negative_slope * x)
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
        # float(...) keeps these Python floats: numpy's float64 scalars would upcast a float32 input to
        # float64, but a Python float casts by value and preserves the input dtype.
        if self.approximate:
            out = 0.5 * x * (1 + pt.tanh(float(np.sqrt(2.0 / np.pi)) * (x + 0.044715 * x**3)))
        else:
            out = 0.5 * x * (1 + pt.erf(x / float(np.sqrt(2.0))))
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
        out = x * pt.sigmoid(self.beta * x)
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


__all__ = ["GELU", "LeakyReLU", "ReLU", "Sigmoid", "SoftPlus", "Softmax", "Swish", "Tanh"]
