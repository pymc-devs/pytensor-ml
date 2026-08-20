import numpy as np

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.link.numba.dispatch.string_codegen import (
    CODE_TOKEN,
    build_source_code,
    create_tuple_string,
)

from pytensor_ml.layers.conv import Col2Im, Im2Col

# The generated kernels are compiled against this namespace rather than the module's, so the only
# name they can reach is spelled out rather than being whatever the module happens to import.
_KERNEL_GLOBALS = {"np": np}


def _window_position(op, axis: int) -> str:
    """Where tap ``t{axis}`` of window ``w{axis}`` sits along one spatial axis."""
    return f"w{axis} * {op.stride[axis]} + t{axis} * {op.dilation[axis]}"


def _loop_nest(op, extent_names: list[str]) -> list:
    """Open the batch, window and tap loops, with every extent written in as a literal."""
    lines: list = ["for b in range(batch):", CODE_TOKEN.INDENT]
    for axis, extent in enumerate(extent_names):
        lines += [f"for w{axis} in range({extent}):", CODE_TOKEN.INDENT]
    for axis, taps in enumerate(op.kernel_size):
        lines += [f"for t{axis} in range({taps}):", CODE_TOKEN.INDENT]
    return lines


def _window_counts(op, source: list, extents: list[str]) -> list[str]:
    """Emit the window count per spatial axis, and name each one."""
    names = []
    for axis, (taps, spacing) in enumerate(zip(op.kernel_size, op.dilation)):
        span = spacing * (taps - 1) + 1
        source.append(f"windows_{axis} = ({extents[axis]} - {span}) // {op.stride[axis]} + 1")
        names.append(f"windows_{axis}")
    return names


@register_funcify_default_op_cache_key(Im2Col)
def numba_funcify_Im2Col(op, node=None, **kwargs):
    """
    Copy one contiguous channel row per window and tap, rather than gathering element by element.

    The loop nest is generated rather than walked, because the spatial rank is a property of the op:
    every window count, stride and dilation reaches the compiler as a constant it can fold into the
    index arithmetic, instead of an offset table rebuilt on every call.

    Channels are copied one at a time rather than a row at a time. Copying the row instead would need a
    memcpy over ``in_channels`` elements, which is short enough at realistic channel counts that the
    call costs more than the loop it replaces, and it measures 1.3-1.7x slower even once the array is
    typed contiguous.
    """
    n_spatial = len(op.kernel_size)
    extents = [f"X.shape[{1 + axis}]" for axis in range(n_spatial)]
    windows = [f"w{axis}" for axis in range(n_spatial)]
    taps = [f"t{axis}" for axis in range(n_spatial)]

    source: list = [
        "def im2col(X):",
        CODE_TOKEN.INDENT,
        "batch = X.shape[0]",
        f"channels = X.shape[{1 + n_spatial}]",
    ]
    counts = _window_counts(op, source, extents)
    out_shape = create_tuple_string(
        ["batch", *counts, *(str(extent) for extent in op.kernel_size), "channels"]
    )
    source.append(f"out = np.empty({out_shape}, dtype=X.dtype)")
    source += _loop_nest(op, counts)
    source += ["for c in range(channels):", CODE_TOKEN.INDENT]
    read = ", ".join(_window_position(op, axis) for axis in range(n_spatial))
    source.append(f"out[b, {', '.join(windows)}, {', '.join(taps)}, c] = X[b, {read}, c]")
    source += [CODE_TOKEN.DEDENT] * (2 * n_spatial + 2)
    source.append("return out")

    kernel = numba_basic.numba_njit(
        compile_numba_function_src(
            build_source_code(source), function_name="im2col", global_env=_KERNEL_GLOBALS
        )
    )
    # The default cache key covers the op's type and props but not this source, so a change here has
    # to be signalled or numba loads the previous kernel for an unchanged op.
    cache_version = 1
    return kernel, cache_version


@register_funcify_default_op_cache_key(Col2Im)
def numba_funcify_Col2Im(op, node=None, **kwargs):
    """
    Accumulate one channel row per window and tap, mirroring the gather.

    Written as a scalar loop over channels rather than a slice ``+=``, which would allocate a temporary
    per visit. Windows overlap, so a position several of them reach is accumulated in loop order.
    """
    n_spatial = len(op.kernel_size)
    arguments = [f"extent_{axis}" for axis in range(n_spatial)]
    extents = [f"{name}_item" for name in arguments]
    windows = [f"w{axis}" for axis in range(n_spatial)]
    taps = [f"t{axis}" for axis in range(n_spatial)]

    source: list = [
        f"def col2im(patches, {', '.join(arguments)}):",
        CODE_TOKEN.INDENT,
        "batch = patches.shape[0]",
        f"channels = patches.shape[{1 + 2 * n_spatial}]",
        # Shape inputs arrive as zero-dimensional arrays, as they do everywhere else in numba-land
        *(f"{item} = {name}.item()" for item, name in zip(extents, arguments)),
    ]
    counts = _window_counts(op, source, extents)
    out_shape = create_tuple_string(["batch", *extents, "channels"])
    source.append(f"out = np.zeros({out_shape}, dtype=patches.dtype)")
    source += _loop_nest(op, counts)
    source += ["for c in range(channels):", CODE_TOKEN.INDENT]
    write = ", ".join(_window_position(op, axis) for axis in range(n_spatial))
    source.append(f"out[b, {write}, c] += patches[b, {', '.join(windows)}, {', '.join(taps)}, c]")
    source += [CODE_TOKEN.DEDENT] * (2 * n_spatial + 2)
    source.append("return out")

    kernel = numba_basic.numba_njit(
        compile_numba_function_src(
            build_source_code(source), function_name="col2im", global_env=_KERNEL_GLOBALS
        )
    )
    # The default cache key covers the op's type and props but not this source, so a change here has
    # to be signalled or numba loads the previous kernel for an unchanged op.
    cache_version = 1
    return kernel, cache_version
