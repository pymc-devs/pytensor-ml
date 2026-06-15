from collections.abc import Callable, Iterable
from functools import partial

import numpy as np
import pytensor
import pytest

from pytensor.compile.mode import MLX, Mode
from pytensor.graph.basic import Variable
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.link.mlx.linker import MLXLinker

mx = pytest.importorskip("mlx.core")

optimizer = RewriteDatabaseQuery(include=["mlx"], exclude=MLX._optimizer.exclude)
mlx_mode = Mode(linker=MLXLinker(), optimizer=optimizer)
py_mode = Mode(linker="py", optimizer=None)


def compare_mlx_and_py(
    graph_inputs: Iterable[Variable],
    graph_outputs: Variable | Iterable[Variable],
    test_inputs: Iterable,
    *,
    assert_fn: Callable | None = None,
    must_be_device_array: bool = True,
    mlx_mode=mlx_mode,
    py_mode=py_mode,
):
    """Compile a graph on MLX and on the python linker, run both, and assert they agree.

    Parameters
    ----------
    graph_inputs : iterable of Variable
        Symbolic root inputs to the graph.
    graph_outputs : Variable or iterable of Variable
        Symbolic outputs of the graph.
    test_inputs : iterable
        Numerical inputs to evaluate the function on.
    assert_fn : callable, optional
        Equality check between the MLX and python results. Defaults to
        ``np.testing.assert_allclose`` with ``rtol=1e-4``.
    must_be_device_array : bool
        Assert the MLX result is an ``mx.array``, confirming it was computed by MLX. Default True.

    Returns
    -------
    tuple
        The compiled MLX function and its result.
    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4)

    if any(inp.owner is not None for inp in graph_inputs):
        raise ValueError("Inputs must be root variables")

    pytensor_mlx_fn = pytensor.function(graph_inputs, graph_outputs, mode=mlx_mode)
    mlx_res = pytensor_mlx_fn(*test_inputs)

    if must_be_device_array:
        if isinstance(mlx_res, list):
            assert all(isinstance(res, mx.array) for res in mlx_res)
        else:
            assert isinstance(mlx_res, mx.array)

    pytensor_py_fn = pytensor.function(graph_inputs, graph_outputs, mode=py_mode)
    py_res = pytensor_py_fn(*test_inputs)

    if isinstance(graph_outputs, list | tuple):
        for m, p in zip(mlx_res, py_res, strict=True):
            assert_fn(np.asarray(m), p)
    else:
        assert_fn(np.asarray(mlx_res), py_res)

    return pytensor_mlx_fn, mlx_res
