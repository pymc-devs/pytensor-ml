from collections.abc import Sequence

import pytensor.tensor as pt

from pytensor.tensor.pad import PadMode
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import Layer, UnaryLayerOp
from pytensor_ml.params import trainable_parameter
from pytensor_ml.state import Initializer, XavierNormalInitializer, ZeroInitializer


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
        ``(*kernel_size, in_channels, out_channels)``, optionally adding ``bias``.
        """
        patches = _extract_patches(X, self.kernel_size, self.stride, self.dilation)

        # Patches come out as batch and windows, then taps, then channels. Flattening the trailing taps
        # and channels into one axis is what turns the correlation into a matmul, and the kernel's own
        # leading axes are already in that order, so both reshapes are contiguous and nothing is
        # permuted.
        n_spatial = len(self.kernel_size)
        kept = patches.shape[: 1 + n_spatial]
        contracted = patches.shape[1 + n_spatial :]
        flat = patches.reshape((*kept, pt.prod(contracted)))

        out = flat @ W.reshape((-1, W.shape[-1]))
        if bias:
            out = out + bias[0]
        return [out]


def _resolve_padding(
    padding: str | int | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """
    Turn a padding argument into an explicit ``(before, after)`` pair per spatial axis.

    ``"same"`` pads by the kernel's span less one, which leaves the output spatial size at
    ``ceil(input / stride)`` -- the input's own size at unit stride. That total is odd whenever the span
    is even, and the extra element goes after rather than before, matching torch and keras.

    Parameters
    ----------
    padding : {"valid", "same"}, int, or sequence of int
        No padding, enough to leave the output at ``ceil(input / stride)``, or an explicit amount
        applied to both sides of every axis or of each axis in turn.
    kernel_size : sequence of int
        Window extent along each spatial axis.
    dilation : sequence of int
        Spacing between taps, which stretches the span the padding has to cover.
    """
    if padding == "valid":
        return tuple((0, 0) for _ in kernel_size)
    if padding == "same":
        pads = []
        for extent, spacing in zip(kernel_size, dilation):
            total = spacing * (extent - 1)
            pads.append((total // 2, total - total // 2))
        return tuple(pads)
    if isinstance(padding, str):
        raise ValueError(
            f"padding must be 'valid', 'same', or an explicit number of elements, but got {padding!r}."
        )

    amounts = [padding] * len(kernel_size) if isinstance(padding, int) else list(padding)
    if any(amount < 0 for amount in amounts):
        raise ValueError(
            f"Padding adds elements, so it cannot be negative; got {padding!r}. To use less of the "
            "input than it has, take a slice of it before convolving."
        )
    if len(amounts) != len(kernel_size):
        raise ValueError(
            f"A convolution over {len(kernel_size)} spatial axes needs one padding amount per axis, but "
            f"got {len(amounts)}."
        )
    return tuple((amount, amount) for amount in amounts)


def _pad_spatial(
    X: TensorVariable, padding: tuple[tuple[int, int], ...], mode: PadMode
) -> TensorVariable:
    """Pad the spatial axes of ``(batch, *spatial, channels)``, leaving batch and channels alone."""
    if not any(before or after for before, after in padding):
        return X
    return pt.pad(X, [(0, 0), *padding, (0, 0)], mode=mode)


def _as_spatial_tuple(value: int | Sequence[int], n_spatial: int, name: str) -> tuple[int, ...]:
    """Broadcast a scalar argument across the spatial axes, or check one already given per axis."""
    if isinstance(value, int):
        return (value,) * n_spatial
    given = tuple(value)
    if len(given) != n_spatial:
        raise ValueError(
            f"{name} must be an int or one value per spatial axis, but got {len(given)} values for "
            f"{n_spatial} spatial axes."
        )
    return given


class _ConvNd(Layer):
    """
    Everything a convolution does that does not depend on how many spatial axes it has.

    Subclasses set :attr:`n_spatial`; see :class:`Conv1D` for the arguments, which are shared.
    """

    n_spatial: int

    def __init__(
        self,
        name: str | None,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] = "valid",
        *,
        padding_mode: PadMode = "constant",
        bias: bool = True,
        weight_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        self.name = name if name else type(self).__name__
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _as_spatial_tuple(kernel_size, self.n_spatial, "kernel_size")
        self.stride = _as_spatial_tuple(stride, self.n_spatial, "stride")
        self.dilation = _as_spatial_tuple(dilation, self.n_spatial, "dilation")
        self.padding = _resolve_padding(padding, self.kernel_size, self.dilation)
        self.padding_mode: PadMode = padding_mode
        self.bias = bias

        # Receptive field, then input channels, then output: the layout flax and keras use, and the one
        # `fans` reads, since it takes the two trailing dimensions as the features.
        self.W = trainable_parameter(
            f"{self.name}_W",
            (*self.kernel_size, in_channels, out_channels),
            weight_initializer,
            XavierNormalInitializer(),
        )
        if bias:
            self.b = trainable_parameter(
                f"{self.name}_b", (out_channels,), bias_initializer, ZeroInitializer()
            )

    def _check_input_covers_a_window(self, X: TensorVariable) -> None:
        """
        Reject a padded input too short for one window, where its length is known at build time.

        Such an input yields no windows at all, so every downstream shape has a zero axis and the graph
        computes an empty answer rather than failing. Only a statically known spatial size can be
        checked; a symbolic one still reaches the loop and comes back empty.
        """
        for axis, (extent, spacing, (before, after)) in enumerate(
            zip(self.kernel_size, self.dilation, self.padding)
        ):
            length = X.type.shape[1 + axis]
            if length is None:
                continue
            span = spacing * (extent - 1) + 1
            if length + before + after < span:
                raise ValueError(
                    f"{self.name} needs at least {span} elements along spatial axis {axis} to place one "
                    f"window, but the input has {length} there and the padding adds {before + after}. "
                    "Shorten the kernel, lower the dilation, or pad more."
                )

    def __call__(self, X: pt.TensorLike) -> TensorVariable:
        """
        Correlate ``X`` of shape ``(batch, *spatial, in_channels)`` with the layer's kernel.

        Returns
        -------
        TensorVariable
            Shape ``(batch, *out_spatial, out_channels)``.
        """
        X = pt.as_tensor(X)
        if X.ndim != self.n_spatial + 2:
            raise ValueError(
                f"{self.name} takes an input of shape (batch, "
                f"{', '.join(['spatial'] * self.n_spatial)}, channels), so a {self.n_spatial}-spatial "
                f"convolution needs a {self.n_spatial + 2}-dimensional input; got a "
                f"{X.ndim}-dimensional one."
            )

        self._check_input_covers_a_window(X)

        padded = _pad_spatial(X, self.padding, self.padding_mode)
        op = ConvLayer(self.kernel_size, self.stride, self.dilation)
        parameters = (self.W, self.b) if self.bias else (self.W,)

        out = op(padded, *parameters)
        out.name = f"{self.name}_output"
        return out


class Conv1D(_ConvNd):
    r"""
    Cross-correlate a sequence with a learned kernel, over one spatial axis.

    Takes ``(batch, time, in_channels)`` and returns ``(batch, out_time, out_channels)``, so it stacks
    with the recurrent layers without a transpose between them. Each output position is the kernel
    contracted against the window of the input beneath it:

    .. math::

        y_{t,o} = b_o + \sum_{c} \sum_{j} x_{t s + j d,\,c} \, W_{j,c,o},

    with :math:`s` the stride and :math:`d` the dilation. The kernel is not flipped, so this is
    correlation in the signal-processing sense and convolution in the sense every ML framework means.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to the class name when None.
    in_channels : int
        Size of the input's channel axis.
    out_channels : int
        Size of the output's channel axis, one per kernel.
    kernel_size : int
        Window extent along the time axis.
    stride : int, optional
        Step between windows. Default is 1.
    dilation : int, optional
        Spacing between the kernel's taps, which widens the receptive field without adding parameters.
        Default is 1.
    padding : {"valid", "same"} or int, optional
        No padding, enough to leave the output length equal to :math:`\lceil \text{time} / s \rceil`, or
        an explicit number of elements on each side. Default is "valid".
    padding_mode : str, optional
        How padded elements are filled, passed to :func:`pytensor.tensor.pad`. Default is "constant",
        which pads with zeros; "reflect", "symmetric" and "edge" are the other useful ones.
    bias : bool, optional
        Add the learned shift :math:`b`, one per output channel. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W` is drawn. Xavier normal when omitted, whose fans count the receptive field.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted.
    """

    n_spatial = 1
