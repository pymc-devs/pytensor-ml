from collections.abc import Callable

import pytensor.tensor as pt


def Input(name: str, shape: tuple[int | None, ...], dtype: str | None = None) -> pt.TensorVariable:
    """
    Create a named symbolic input tensor.

    Parameters
    ----------
    name : str
        Name of the input variable.
    shape : tuple of int or None
        Size of each dimension. Use None wherever the size varies between calls, such as a batch axis.
    dtype : str, optional
        Data type of the input. Default ``floatX``.
    """
    return pt.tensor(name=name, shape=shape, dtype=dtype)


def Sequential(*layers: Callable) -> Callable:
    """Compose layers left to right into a single callable that threads its input through each in turn."""

    def forward(x: pt.TensorLike) -> pt.TensorLike:
        for layer in layers:
            x = layer(x)
        return x

    return forward


def Flatten(X: pt.TensorLike) -> pt.TensorVariable:
    """
    Collapse everything after the batch axis into one, so a convolution stack can reach a dense head.

    Parameters
    ----------
    X : TensorLike
        An activation of any rank, whose first axis is the batch.

    Returns
    -------
    TensorVariable
        Shape ``(batch, features)``, with ``features`` the product of every remaining axis.
    """
    return pt.join_dims(X, start_axis=1)


Squeeze = pt.squeeze
Concatenate = pt.concatenate
