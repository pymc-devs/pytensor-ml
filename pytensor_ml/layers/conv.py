from collections.abc import Sequence

import pytensor.tensor as pt

from pytensor.tensor.variable import TensorVariable


def _extract_patches(
    X: TensorVariable,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> TensorVariable:
    """
    Gather every window a kernel visits, for an input with any number of spatial axes.

    Parameters
    ----------
    X : TensorVariable
        Input of shape ``(batch, *spatial, channels)``.
    kernel_size : sequence of int
        Window extent along each spatial axis. Its length is the number of spatial axes.
    stride : sequence of int
        Step between windows along each spatial axis.
    dilation : sequence of int
        Spacing between the taps of one window along each spatial axis.

    Returns
    -------
    TensorVariable
        Shape ``(batch, *out_spatial, *kernel_size, channels)``, where ``out_spatial`` counts the
        windows that fit.
    """
    n_spatial = len(kernel_size)
    windows = []
    for axis, (extent, step, spacing) in enumerate(zip(kernel_size, stride, dilation)):
        span = spacing * (extent - 1) + 1
        starts = pt.arange(0, X.shape[1 + axis] - span + 1, step)
        window = starts[:, None] + pt.arange(extent)[None, :] * spacing

        # One advanced index per spatial axis, each carrying its own window axis and its own tap axis
        # and broadcasting against the others, so the gathered result is windows-then-taps in order.
        pattern: list[int | str] = ["x"] * (2 * n_spatial)
        pattern[axis] = 0
        pattern[n_spatial + axis] = 1
        windows.append(window.dimshuffle(*pattern))

    return X[(slice(None), *windows, slice(None))]
