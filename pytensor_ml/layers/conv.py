from collections.abc import Sequence

import pytensor.tensor as pt

from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import UnaryLayerOp


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


class ConvLayer(UnaryLayerOp):
    """
    Cross-correlate an input with a kernel, over any number of spatial axes.

    Marks the convolution as one node so a backend with a real kernel can be dispatched to it, the way
    :class:`~pytensor_ml.layers.attention.AttentionLayer` is. The graph inside gathers every window and
    contracts them against the kernel with a single ``Dot``, whose reduction runs over
    ``prod(kernel_size) * in_channels`` -- deep enough to reach BLAS, which is what a convolution over
    single-channel signal ops cannot offer.

    Correlation, not convolution: the kernel is not flipped, matching torch, keras and flax.
    """

    __props__ = ("kernel_size", "stride", "dilation")

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
        **kwargs,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        super().__init__(**kwargs)

    def build_inner_graph(self, X, W, *bias):
        """
        Correlate ``X`` of shape ``(batch, *spatial, in_channels)`` with ``W`` of shape
        ``(in_channels, out_channels, *kernel_size)``, optionally adding ``bias``.
        """
        patches = _extract_patches(X, self.kernel_size, self.stride, self.dilation)

        # Windows first, then taps, then channels: flattening the trailing taps-and-channels axes into
        # one is what turns the correlation into a matmul, and it matches how W is flattened below.
        n_spatial = len(self.kernel_size)
        taps = patches.shape[1 + n_spatial :]
        windows = patches.shape[: 1 + n_spatial]
        flat = patches.reshape((*windows, pt.prod(taps)))

        # The kernel is stored input-dimension-first, which is what `fans` reads to size a draw; the
        # contraction wants taps-then-input-channel to match how the patches flattened. The transpose is
        # over the kernel alone, which is negligible beside the matmul it feeds.
        kernel = W.transpose(*range(2, 2 + n_spatial), 0, 1)
        out = flat @ kernel.reshape((-1, kernel.shape[-1]))
        if bias:
            out = out + bias[0]
        return [out]
