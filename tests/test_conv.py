import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.layers.conv import _extract_patches

floatX = pytensor.config.floatX


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
