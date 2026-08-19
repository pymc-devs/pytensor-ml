import numpy as np

from numba import prange
from numba.np.unsafe.ndarray import to_fixed_tuple
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch import numba_funcify

from pytensor_ml.layers.conv import Im2Col


@numba_funcify.register(Im2Col)
def numba_funcify_Im2Col(op, node=None, **kwargs):
    """
    Copy one contiguous channel row per window and tap, rather than gathering element by element.

    The equivalent advanced-indexing graph computes an index per element and reaches a few GB/s; this
    reaches memory bandwidth, which is all the operation can do. Flattening the spatial axes into one
    keeps the loop nest independent of how many there are, at the cost of two small offset tables built
    per call.
    """
    kernel_size = np.asarray(op.kernel_size, dtype=np.int64)
    stride = np.asarray(op.stride, dtype=np.int64)
    dilation = np.asarray(op.dilation, dtype=np.int64)
    n_spatial = len(op.kernel_size)
    n_taps = int(np.prod(op.kernel_size))
    # A closed-over tuple stays a compile-time constant, which is what lets the output shape be built
    # without unpacking an array into one -- numba has no variable-length tuples.
    kernel_tuple = tuple(int(extent) for extent in op.kernel_size)

    @numba_basic.numba_njit
    def flat_offsets(counts, steps, spatial_strides):
        """Flat spatial offset of every position in a grid of ``counts``, stepping by ``steps``."""
        total = 1
        for axis in range(n_spatial):
            total *= counts[axis]
        offsets = np.zeros(total, dtype=np.int64)
        for flat in range(total):
            remainder, offset = flat, 0
            for axis in range(n_spatial - 1, -1, -1):
                offset += (remainder % counts[axis]) * steps[axis] * spatial_strides[axis]
                remainder //= counts[axis]
            offsets[flat] = offset
        return offsets

    @numba_basic.numba_njit(parallel=True)
    def im2col(X):
        batch, channels = X.shape[0], X.shape[-1]
        spatial = np.empty(n_spatial, dtype=np.int64)
        for axis in range(n_spatial):
            spatial[axis] = X.shape[1 + axis]

        # Row-major strides over the spatial axes alone, so an offset indexes the flattened middle.
        spatial_strides = np.ones(n_spatial, dtype=np.int64)
        for axis in range(n_spatial - 2, -1, -1):
            spatial_strides[axis] = spatial_strides[axis + 1] * spatial[axis + 1]

        windows = np.empty(n_spatial, dtype=np.int64)
        for axis in range(n_spatial):
            span = dilation[axis] * (kernel_size[axis] - 1) + 1
            windows[axis] = (spatial[axis] - span) // stride[axis] + 1

        window_offsets = flat_offsets(windows, stride, spatial_strides)
        tap_offsets = flat_offsets(kernel_size, dilation, spatial_strides)

        flat_X = X.reshape(batch, -1, channels)
        out = np.empty((batch, len(window_offsets), n_taps, channels), dtype=X.dtype)
        for b in prange(batch):
            for w in range(len(window_offsets)):
                base = window_offsets[w]
                for tap in range(n_taps):
                    out[b, w, tap, :] = flat_X[b, base + tap_offsets[tap], :]

        return out.reshape((batch, *to_fixed_tuple(windows, n_spatial), *kernel_tuple, channels))

    return im2col
