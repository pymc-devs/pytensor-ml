from collections.abc import Callable

import pytensor.tensor as pt


def Input(name: str, shape: tuple[int, ...], dtype: str | None = None) -> pt.TensorVariable:
    """
    Create a named symbolic input tensor with a fully static shape.

    Parameters
    ----------
    name : str
        Name of the input variable.
    shape : tuple of int
        Static size of each dimension. Raise ``ValueError`` if any entry is not an integer.
    dtype : str or None
        Data type of the input. Defaults to ``floatX`` when None.
    """
    if not all(isinstance(dim, int) for dim in shape):
        raise ValueError("All dimensions must be integers")

    return pt.tensor(name=name, shape=shape, dtype=dtype)


def Sequential(*layers: Callable) -> Callable:
    def forward(x: pt.TensorLike) -> pt.TensorLike:
        for layer in layers:
            x = layer(x)
        return x

    return forward


Squeeze = pt.squeeze
Concatenate = pt.concatenate
