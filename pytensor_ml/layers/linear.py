import numpy as np
import pytensor.tensor as pt

from pytensor import config

from pytensor_ml.base import Layer, UnaryLayerOp
from pytensor_ml.params import trainable
from pytensor_ml.state import ZeroInitializer


def shape_to_str(shape):
    inner = ",".join([str(st_dim) if st_dim is not None else "?" for st_dim in shape])
    return f"({inner})"


class LinearLayer(UnaryLayerOp):
    __props__ = ("n_in", "n_out", "bias")

    def build_inner_graph(self, X, W, b=None):
        res = X @ W
        if self.bias:
            res = res + b
        return [res]


class Linear(Layer):
    r"""
    Affine map :math:`y = x W + b`.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "Linear" when None.
    n_in : int
        Size of the input feature axis.
    n_out : int
        Size of the output feature axis.
    bias : bool, optional
        Add the learned shift :math:`b`, which starts at zero and stays there under a network-wide
        initialization scheme. Default is True.

    Notes
    -----
    The weight matrix :math:`W` starts at zero, so in a stack every activation below the first layer is
    zero and every weight matrix receives a zero gradient: an uninitialized network can fit only its
    output bias, and predicts a constant. Call :meth:`~pytensor_ml.model.Model.initialize`, or assign a
    value yourself, before training.
    """

    def __init__(self, name: str | None, n_in: int, n_out: int, bias: bool = True):
        self.name = name if name else "Linear"
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias

        W_value = np.zeros((n_in, n_out), dtype=config.floatX)
        self.W = trainable(W_value, f"{self.name}_W")

        if self.bias:
            b_value = np.zeros(n_out, dtype=config.floatX)
            self.b = trainable(b_value, f"{self.name}_b", initializer=ZeroInitializer())

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)

        inputs = [X, self.W]
        if self.bias:
            inputs.append(self.b)

        input_shape = shape_to_str(X.type.shape)
        output_shape = shape_to_str((X @ self.W).type.shape)

        ofg = LinearLayer(
            name=f"{self.name}[{input_shape} -> {output_shape}]",
            n_in=self.n_in,
            n_out=self.n_out,
            bias=self.bias,
        )
        out = ofg(*inputs)
        out.name = f"{self.name}_output"

        return out
