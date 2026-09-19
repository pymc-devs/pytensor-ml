import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("mlx.core")

from pytensor.compile.mode import Mode
from pytensor.link.mlx.linker import MLXLinker

from pytensor_ml.optim import lbfgs_updates
from pytensor_ml.optim.lbfgs import LBFGSDirection
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import function
from tests.dispatch.mlx.test_basic import compare_mlx_and_py
from tests.optim.test_lbfgs import dense_inverse_hessian, ring_stacks

floatX = pytensor.config.floatX


@pytest.mark.parametrize("n_pairs, count", [(2, 2), (4, 6)], ids=["not_yet_wrapped", "wrapped"])
def test_direction_matches_py(n_pairs, count):
    rng = np.random.default_rng(sum(map(ord, "MLX LBFGS")))
    shapes = [(3, 2), (4,)]
    size = sum(int(np.prod(shape)) for shape in shapes)
    memory_size, gamma = 4, 0.7
    gradient = rng.normal(size=size).astype(floatX)
    pairs = []
    for _ in range(n_pairs):
        s = rng.normal(size=size).astype(floatX)
        pairs.append((s, rng.normal(size=size).astype(floatX) + 0.5 * s))
    S, Y = ring_stacks(pairs, memory_size, count, shapes)
    splits = np.cumsum([int(np.prod(shape)) for shape in shapes])[:-1]
    gradient_pieces = [
        piece.reshape(shape) for piece, shape in zip(np.split(gradient, splits), shapes)
    ]

    gradients = [pt.tensor(f"g{i}", shape=shape) for i, shape in enumerate(shapes)]
    S_in = [pt.tensor(f"S{i}", shape=(memory_size, *shape)) for i, shape in enumerate(shapes)]
    Y_in = [pt.tensor(f"Y{i}", shape=(memory_size, *shape)) for i, shape in enumerate(shapes)]
    op = LBFGSDirection(n_parameters=2, memory_size=memory_size)
    outputs = op(count, gamma, *gradients, *S_in, *Y_in, return_list=True)

    _, got = compare_mlx_and_py(
        [*gradients, *S_in, *Y_in],
        outputs,
        [*gradient_pieces, *S, *Y],
        assert_fn=lambda got, want: np.testing.assert_allclose(got, want, rtol=1e-4),
    )
    want = dense_inverse_hessian(gamma, pairs, size) @ gradient
    np.testing.assert_allclose(
        np.concatenate([np.asarray(d).ravel() for d in got]), want, rtol=1e-4
    )


def test_a_single_parameter_returns_one_array():
    g = np.arange(5, dtype=floatX)
    S = np.zeros((3, 5), dtype=floatX)
    Y = np.zeros((3, 5), dtype=floatX)
    g_in = pt.tensor("g", shape=(5,))
    S_in = pt.tensor("S", shape=(3, 5))
    Y_in = pt.tensor("Y", shape=(3, 5))

    d = LBFGSDirection(n_parameters=1, memory_size=3)(0, 0.5, g_in, S_in, Y_in)

    compare_mlx_and_py([g_in, S_in, Y_in], d, [g, S, Y])


@pytest.mark.parametrize("use_compile", [True, False], ids=["compiled", "eager"])
def test_the_rule_reaches_the_minimum_of_a_quadratic(use_compile):
    # The rule reads and writes its ring with a traced slot, which mlx traces only as advanced indexing
    # (pymc-devs/pytensor#2422); this is the end-to-end check that the whole step compiles and runs.
    A = np.array([[3.0, 0.5], [0.5, 1.0]])
    b = np.array([1.0, -2.0])
    u = trainable(np.array([5.0], dtype=floatX), name="u")
    v = trainable(np.array([-3.0], dtype=floatX), name="v")
    x = pt.concatenate([u, v])
    loss = 0.5 * x @ pt.constant(A, dtype=floatX) @ x - pt.constant(b, dtype=floatX) @ x
    mode = Mode(linker=MLXLinker(use_compile=use_compile), optimizer="fast_run")
    step = function(
        [], loss, updates=lbfgs_updates(loss, [u, v], learning_rate=1.0, memory_size=2), mode=mode
    )

    for _ in range(12):
        step()

    np.testing.assert_allclose(
        np.concatenate([np.asarray(u.get_value()), np.asarray(v.get_value())]),
        np.linalg.solve(A, b),
        rtol=1e-4,
    )
