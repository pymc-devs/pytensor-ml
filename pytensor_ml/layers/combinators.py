from collections.abc import Callable, Sequence

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
    flattened : TensorVariable
        Shape ``(batch, features)``, with ``features`` the product of every remaining axis.
    """
    return pt.join_dims(X, start_axis=1)


def Squeeze(X: pt.TensorLike, axis: int | Sequence[int] | None = None) -> pt.TensorVariable:
    """
    Drop length-1 axes, so a layer that emits a singleton axis feeds one that does not expect it.

    Parameters
    ----------
    X : TensorLike
        Tensor to squeeze.
    axis : int or sequence of int, optional
        Axes to drop. An axis whose length is statically known to be anything but 1 is rejected as the
        graph is built; an axis of unknown length is accepted and checked when the function runs.
        Default None, which drops every axis already known to have length 1 and leaves the rest.

    Returns
    -------
    squeezed : TensorVariable
        ``X`` with the selected axes removed.
    """
    return pt.squeeze(X, axis=axis)


def Concatenate(tensors: Sequence[pt.TensorLike], axis: int = 0) -> pt.TensorVariable:
    """
    Join tensors end to end along one axis.

    Parameters
    ----------
    tensors : sequence of TensorLike
        Tensors to join. Every one must agree in rank, and in size on every axis but ``axis``.
    axis : int, optional
        Axis to join along; negative values count from the right. Default 0, which for a batched
        activation is the batch axis -- merging two ``(batch, features)`` branches feature-wise wants
        ``axis=-1``.

    Returns
    -------
    joined : TensorVariable
        The inputs joined, with extent along ``axis`` equal to the sum of the inputs' extents.
    """
    return pt.concatenate(tensors, axis=axis)
