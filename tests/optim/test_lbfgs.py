from functools import cache

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.optim.lbfgs import LBFGSDirection
from pytensor_ml.pytensorf import function
from tests.optim.lbfgs_reference import dense_inverse_hessian, ring_stacks

floatX = pytensor.config.floatX
RTOL = 1e-6 if floatX == "float64" else 1e-4


@pytest.mark.parametrize("n_pairs, count", [(2, 2), (4, 6)], ids=["not_yet_wrapped", "wrapped"])
def test_direction_matches_the_two_loop_recursion_over_a_ring(n_pairs, count):
    # The reference is the dense matrix the recursion is an algorithm for, built from the textbook update
    # on a flat vector, so it shares neither the loop nor the ring or reshape arithmetic with the op. Two
    # parameters of different rank exercise the cross-parameter dot products. Before the ring wraps its
    # empty slots lead the order; after, the newest pair sits mid-ring.
    rng = np.random.default_rng(0)
    shapes = [(3, 2), (4,)]
    size = sum(int(np.prod(shape)) for shape in shapes)
    memory_size, gamma = 4, 0.7
    gradient = rng.normal(size=size).astype(floatX)
    pairs = []
    for _ in range(n_pairs):
        s = rng.normal(size=size).astype(floatX)
        noise = rng.normal(size=size).astype(floatX)
        pairs.append((s, noise - (noise @ s) / (s @ s) * s + 0.5 * s))  # y . s = 0.5 s . s > 0
    S, Y, rho = ring_stacks(pairs, memory_size, count, shapes)

    op = LBFGSDirection(n_parameters=2, memory_size=memory_size)
    gradients = [pt.tensor(f"g{i}", shape=shape) for i, shape in enumerate(shapes)]
    S_in = [pt.tensor(f"S{i}", shape=(memory_size, *shape)) for i, shape in enumerate(shapes)]
    Y_in = [pt.tensor(f"Y{i}", shape=(memory_size, *shape)) for i, shape in enumerate(shapes)]
    direction = function(
        [*gradients, *S_in, *Y_in],
        op(count, gamma, rho, *gradients, *S_in, *Y_in, return_list=True),
    )

    splits = np.cumsum([int(np.prod(shape)) for shape in shapes])[:-1]
    gradient_pieces = [
        piece.reshape(shape) for piece, shape in zip(np.split(gradient, splits), shapes)
    ]
    got = np.concatenate([d.ravel() for d in direction(*gradient_pieces, *S, *Y)])
    want = dense_inverse_hessian(gamma, pairs, size) @ gradient
    np.testing.assert_allclose(got, want, rtol=RTOL)


def test_parameters_of_different_dtypes_keep_their_own():
    # The cross-parameter dot products upcast to the widest dtype; each carried vector has to be cast
    # back or the scan refuses the narrower parameter's recurrence.
    g_wide = pt.tensor("g_wide", shape=(3,), dtype="float64")
    g_narrow = pt.tensor("g_narrow", shape=(2,), dtype="float32")
    S_wide, Y_wide = (pt.tensor(name, shape=(2, 3), dtype="float64") for name in "SY")
    S_narrow, Y_narrow = (pt.tensor(name, shape=(2, 2), dtype="float32") for name in ("s", "y"))

    wide, narrow = LBFGSDirection(n_parameters=2, memory_size=2)(
        1, 0.5, np.zeros(2), g_wide, g_narrow, S_wide, S_narrow, Y_wide, Y_narrow, return_list=True
    )

    assert (wide.dtype, narrow.dtype) == ("float64", "float32")


def test_a_scalar_parameter_has_vector_stacks():
    g = pt.scalar("g", dtype=floatX)
    S = pt.tensor("S", shape=(3,), dtype=floatX)
    Y = pt.tensor("Y", shape=(3,), dtype=floatX)

    d = LBFGSDirection(n_parameters=1, memory_size=3)(1, 1.0, [0.0, 0.0, 1 / (1.5 * 3.0)], g, S, Y)

    # One pair (s, y) with y = 2 s: H y = s, so H maps g onto g / 2.
    np.testing.assert_allclose(
        d.eval(
            {g: 4.0, S: np.array([0, 0, 1.5], dtype=floatX), Y: np.array([0, 0, 3.0], dtype=floatX)}
        ),
        2.0,
        rtol=RTOL,
    )


@cache
def one_vector_direction():
    """One compiled direction over a single vector parameter and three slots, shared by every test that
    needs no other shape, so they pay for one scan compile between them."""
    count = pt.scalar("count", dtype="int64")
    gamma = pt.scalar("gamma", dtype=floatX)
    rho = pt.vector("rho", shape=(3,), dtype=floatX)
    g = pt.vector("g", dtype=floatX)
    S = pt.tensor("S", shape=(3, None), dtype=floatX)
    Y = pt.tensor("Y", shape=(3, None), dtype=floatX)
    direction = LBFGSDirection(n_parameters=1, memory_size=3)(count, gamma, rho, g, S, Y)
    return function([count, gamma, rho, g, S, Y], direction)


def test_an_empty_memory_keeps_a_narrower_dtype():
    # Built at float32 whatever floatX is, so the Python-float gamma has a narrower dtype to upcast.
    g = np.zeros(5, dtype="float32")
    S = np.zeros((3, 5), dtype="float32")

    d = LBFGSDirection(n_parameters=1, memory_size=3)(0, 0.25, np.zeros(3), g, S, S)

    assert d.dtype == "float32"


def test_an_empty_memory_scales_the_gradient():
    g = np.random.default_rng(1).normal(size=5).astype(floatX)
    S = np.zeros((3, 5), dtype=floatX)

    d = one_vector_direction()(0, 0.25, np.zeros(3, dtype=floatX), g, S, S)

    np.testing.assert_allclose(d, 0.25 * g, rtol=RTOL)


def test_a_zero_curvature_retires_its_slot_whatever_the_stacks_hold():
    # The op applies the curvatures it is given and never measures them from the stacks, so a slot whose
    # curvature is zero drops out of the recursion even with a pair still written in it.
    rng = np.random.default_rng(2)
    g = rng.normal(size=4).astype(floatX)
    s = rng.normal(size=4).astype(floatX)
    y = (s + 0.5 * rng.normal(size=4)).astype(floatX)
    S = np.stack([s, np.zeros_like(s), np.zeros_like(s)])
    Y = np.stack([y, np.zeros_like(y), np.zeros_like(y)])

    d = one_vector_direction()(1, 0.5, np.zeros(3, dtype=floatX), g, S, Y)

    np.testing.assert_allclose(d, 0.5 * g, rtol=RTOL)


@pytest.mark.parametrize(
    "props, tensors, message",
    [
        ({"n_parameters": 0, "memory_size": 3}, (), "n_parameters must be at least 1"),
        (
            {"n_parameters": 1, "memory_size": 0},
            (np.ones(0), np.ones(2), np.ones((0, 2)), np.ones((0, 2))),
            "memory_size must be at least 1",
        ),
        (
            {"n_parameters": 1, "memory_size": 3},
            (np.ones(3), np.ones(2), np.ones((3, 2)), np.ones((3, 2)), np.ones((3, 2))),
            "takes 3 tensors",
        ),
        (
            {"n_parameters": 1, "memory_size": 3},
            (np.ones(3), np.ones(2), np.ones((4, 2)), np.ones((3, 2))),
            "memory_size=3",
        ),
        (
            {"n_parameters": 1, "memory_size": 3},
            (np.ones(3), np.ones(2), np.ones((3, 2, 1)), np.ones((3, 2))),
            "memory_size=3",
        ),
        (
            {"n_parameters": 1, "memory_size": 3},
            (np.ones(3), np.ones(2, dtype="float32"), np.ones((3, 2)), np.ones((3, 2))),
            "dtype",
        ),
        (
            {"n_parameters": 1, "memory_size": 3},
            (np.ones(4), np.ones(2), np.ones((3, 2)), np.ones((3, 2))),
            "one curvature per slot",
        ),
        (
            {"n_parameters": 1, "memory_size": 3},
            (np.ones(3), np.ones(2), pt.tensor("S", shape=(None, 2)), np.ones((3, 2))),
            "slot count static",
        ),
    ],
    ids=[
        "no_parameters",
        "no_memory",
        "extra_tensor",
        "wrong_slots",
        "wrong_rank",
        "wrong_dtype",
        "wrong_rho_length",
        "dynamic_slots",
    ],
)
def test_malformed_inputs_are_refused_at_build_time(props, tensors, message):
    with pytest.raises(ValueError, match=message):
        LBFGSDirection(**props)(0, 1.0, *tensors)
