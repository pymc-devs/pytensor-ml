import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from scipy.signal import correlate

from pytensor_ml.layers.conv import ConvLayer, _extract_patches

floatX = pytensor.config.floatX

# The reference sums channel contributions in a different order than the Dot, so the gap tracks
# the precision.
ATOL = 1e-6 if floatX == "float64" else 1e-4


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "pytensor_ml conv")))


def test_patches_over_one_spatial_axis_are_the_windows_a_kernel_visits():
    """The windows written out by hand. A sequence 0..5 with width 3 and unit stride visits four of
    them, and each row of the result is the three consecutive values the kernel would multiply."""
    X = pt.tensor("X", shape=(None, None, 1))
    patches = _extract_patches(X, kernel_size=(3,), stride=(1,), dilation=(1,))

    X_np = np.arange(6, dtype=floatX).reshape(1, 6, 1)
    got = patches.eval({X: X_np})

    assert got.shape == (1, 4, 3, 1)
    expected = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=floatX)
    np.testing.assert_array_equal(got[0, :, :, 0], expected)


def test_patches_step_by_the_stride_and_skip_by_the_dilation():
    """Stride moves the window's start, dilation spreads its taps, and the two are independent. Read
    off 0..7: stride 2 starts at 0, 2, 4; dilation 2 takes every other element within a window."""
    X = pt.tensor("X", shape=(None, None, 1))
    X_np = np.arange(8, dtype=floatX).reshape(1, 8, 1)

    strided = _extract_patches(X, kernel_size=(2,), stride=(3,), dilation=(1,)).eval({X: X_np})
    np.testing.assert_array_equal(strided[0, :, :, 0], np.array([[0, 1], [3, 4], [6, 7]]))

    dilated = _extract_patches(X, kernel_size=(3,), stride=(1,), dilation=(2,)).eval({X: X_np})
    np.testing.assert_array_equal(
        dilated[0, :, :, 0], np.array([[0, 2, 4], [1, 3, 5], [2, 4, 6], [3, 5, 7]])
    )


def test_patches_over_two_spatial_axes_pair_every_row_window_with_every_column_window():
    """The 2-D case is the 1-D case on each axis and the product across them, which is what the
    broadcast between the two index arrays has to produce. A 3x3 image with a 2x2 window visits four
    positions, and the corner ones are the top-left and bottom-right 2x2 blocks."""
    X = pt.tensor("X", shape=(None, None, None, 1))
    patches = _extract_patches(X, kernel_size=(2, 2), stride=(1, 1), dilation=(1, 1))

    X_np = np.arange(9, dtype=floatX).reshape(1, 3, 3, 1)
    got = patches.eval({X: X_np})

    assert got.shape == (1, 2, 2, 2, 2, 1)
    np.testing.assert_array_equal(got[0, 0, 0, :, :, 0], np.array([[0, 1], [3, 4]]))
    np.testing.assert_array_equal(got[0, 1, 1, :, :, 0], np.array([[4, 5], [7, 8]]))


def test_patches_take_a_different_extent_per_spatial_axis(rng):
    """Nothing forces the window to be square, and an implementation that broadcast one axis's index
    array over both would agree with a square kernel and disagree here."""
    X = pt.tensor("X", shape=(None, None, None, None))
    patches = _extract_patches(X, kernel_size=(2, 3), stride=(1, 1), dilation=(1, 1))

    X_np = rng.normal(size=(2, 5, 7, 3)).astype(floatX)
    got = patches.eval({X: X_np})

    assert got.shape == (2, 4, 5, 2, 3, 3)
    np.testing.assert_allclose(got[1, 2, 3], X_np[1, 2:4, 3:6], atol=1e-12)


def test_patches_carry_every_batch_and_channel_axis_through(rng):
    """The gather touches the spatial axes alone, so a batch element and a channel are along for the
    ride. Checked against numpy slicing at a window chosen off the diagonal."""
    X = pt.tensor("X", shape=(None, None, None, None))
    patches = _extract_patches(X, kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))

    X_np = rng.normal(size=(4, 9, 9, 5)).astype(floatX)
    got = patches.eval({X: X_np})

    assert got.shape == (4, 4, 4, 3, 3, 5)
    for batch, row, col in ((0, 0, 3), (3, 2, 1)):
        np.testing.assert_allclose(
            got[batch, row, col],
            X_np[batch, 2 * row : 2 * row + 3, 2 * col : 2 * col + 3],
            atol=1e-12,
        )


def correlate_reference(X_np, W_np, stride=1, dilation=1):
    """A convolution written with scipy, one channel pair at a time, as the independent reference.

    ``scipy.signal.correlate`` is not another path through pytensor, so it cannot agree with a bug the
    implementation and a pytensor-based reference would share. It has no notion of stride or dilation,
    so dilation is spelled by zero-stuffing the kernel and stride by subsampling the result -- both
    definitions rather than reimplementations of what the layer does.
    """
    in_channels, out_channels, *kernel = W_np.shape
    spans = tuple(dilation * (extent - 1) + 1 for extent in kernel)
    if dilation != 1:
        stuffed = np.zeros((in_channels, out_channels, *spans), dtype=W_np.dtype)
        stuffed[(slice(None), slice(None), *(slice(None, None, dilation) for _ in kernel))] = W_np
        W_np = stuffed
    outputs = []
    for image in X_np:
        planes = []
        for out_channel in range(out_channels):
            total = None
            for in_channel in range(in_channels):
                term = correlate(
                    image[..., in_channel], W_np[in_channel, out_channel], mode="valid"
                )
                total = term if total is None else total + term
            planes.append(total)
        outputs.append(np.stack(planes, axis=-1))
    assert outputs[0].ndim == len(kernel) + 1
    stacked = np.stack(outputs)
    return stacked[(slice(None), *(slice(None, None, stride) for _ in kernel), slice(None))]


@pytest.mark.parametrize(
    "spatial, kernel_size",
    [((9,), (3,)), ((9, 9), (3, 3)), ((7, 9), (2, 4))],
    ids=["1d", "2d_square", "2d_rectangular"],
)
def test_the_conv_op_correlates_like_scipy(spatial, kernel_size, rng):
    """The load-bearing correctness test. Every channel pair, summed over input channels, against
    scipy's correlation rather than against another pytensor graph."""
    in_channels, out_channels = 3, 4
    X = pt.tensor("X", shape=(None, *(None for _ in spatial), in_channels))
    W = pt.tensor("W", shape=(in_channels, out_channels, *kernel_size))
    op = ConvLayer(
        kernel_size=kernel_size, stride=(1,) * len(spatial), dilation=(1,) * len(spatial)
    )

    X_np = rng.normal(size=(2, *spatial, in_channels)).astype(floatX)
    W_np = rng.normal(size=(in_channels, out_channels, *kernel_size)).astype(floatX)

    np.testing.assert_allclose(
        op(X, W).eval({X: X_np, W: W_np}), correlate_reference(X_np, W_np), atol=ATOL
    )


def test_the_conv_op_does_not_flip_its_kernel(rng):
    """Correlation, not convolution. An asymmetric kernel is the only thing that tells them apart, and
    a flipped implementation would still agree with scipy's ``convolve``, so pin the convention."""
    X = pt.tensor("X", shape=(None, None, 1))
    W = pt.tensor("W", shape=(1, 1, 2))
    op = ConvLayer(kernel_size=(2,), stride=(1,), dilation=(1,))

    X_np = np.array([[[1.0], [0.0], [0.0]]], dtype=floatX)
    W_np = np.array([[[1.0, 10.0]]], dtype=floatX)

    # The kernel's first tap lands on the input's first element; flipped, the 10 would land there.
    np.testing.assert_allclose(op(X, W).eval({X: X_np, W: W_np})[0, :, 0], [1.0, 0.0], atol=ATOL)


def test_the_conv_op_adds_an_optional_bias(rng):
    """The bias is a third input rather than a separate Elemwise outside the op, so that a backend
    kernel that fuses it has something to fuse."""
    X = pt.tensor("X", shape=(None, None, 2))
    W = pt.tensor("W", shape=(2, 4, 3))
    b = pt.tensor("b", shape=(4,))
    op = ConvLayer(kernel_size=(3,), stride=(1,), dilation=(1,))

    X_np = rng.normal(size=(2, 8, 2)).astype(floatX)
    W_np = rng.normal(size=(2, 4, 3)).astype(floatX)
    b_np = rng.normal(size=(4,)).astype(floatX)

    without = op(X, W).eval({X: X_np, W: W_np})
    with_bias = op(X, W, b).eval({X: X_np, W: W_np, b: b_np})
    np.testing.assert_allclose(with_bias, without + b_np, atol=ATOL)


def test_the_conv_op_takes_a_gradient_matching_finite_differences(rng):
    """Overlapping windows make the patch gather's pullback a scatter-add, and pytensor will produce a
    gradient for an advanced-indexing expression whether or not the accumulation is right. Checked
    against finite differences in float64, where the step size is meaningful."""
    with pytensor.config.change_flags(floatX="float64"):
        X = pt.tensor("X", shape=(None, None, None, 2), dtype="float64")
        W = pt.tensor("W", shape=(2, 3, 2, 2), dtype="float64")
        op = ConvLayer(kernel_size=(2, 2), stride=(1, 1), dilation=(1, 1))

        X_np = rng.normal(size=(2, 5, 5, 2))
        W_np = rng.normal(size=(2, 3, 2, 2))
        cost = (op(X, W) ** 2).sum()

        for wrt, value in ((X, X_np), (W, W_np)):
            analytic = pt.grad(cost, wrt).eval({X: X_np, W: W_np})
            numeric = np.zeros_like(value)
            step = 1e-6
            flat = numeric.reshape(-1)
            for index in range(flat.size):
                nudged = value.copy().reshape(-1)
                nudged[index] += step
                up = cost.eval({X: X_np, W: W_np} | {wrt: nudged.reshape(value.shape)})
                nudged[index] -= 2 * step
                down = cost.eval({X: X_np, W: W_np} | {wrt: nudged.reshape(value.shape)})
                flat[index] = (up - down) / (2 * step)
            np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)
