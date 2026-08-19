from functools import partial

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("jax")

from pytensor_ml.layers import Conv1D
from pytensor_ml.layers.conv import ConvLayer
from tests.dispatch.jax.test_basic import compare_jax_and_py

floatX = pytensor.config.floatX
assert_close = partial(np.testing.assert_allclose, atol=1e-5, rtol=1e-4)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(sum(map(ord, "jax conv")))


@pytest.mark.parametrize(
    "stride, dilation",
    [(1, 1), (2, 1), (1, 2), (3, 2)],
    ids=["plain", "strided", "dilated", "both"],
)
def test_the_conv_op_matches_the_graph_it_replaces(stride, dilation, rng):
    """The dispatch has to agree with the gather-and-Dot it stands in for, at every combination of
    stride and dilation the op carries in its props."""
    X_np = rng.normal(size=(2, 16, 3)).astype(floatX)
    W_np = rng.normal(size=(3, 3, 4)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)
    W = pt.tensor("W", shape=W_np.shape)

    out = ConvLayer((3,), (stride,), (dilation,))(X, W)
    compare_jax_and_py([X, W], out, [X_np, W_np], assert_fn=assert_close)


def test_the_conv_op_adds_its_bias(rng):
    """The bias is a third input to the op rather than an Elemwise outside it, so the dispatch is what
    has to add it; dropping it would go unnoticed by a test that never passes one."""
    X_np = rng.normal(size=(2, 12, 3)).astype(floatX)
    W_np = rng.normal(size=(4, 3, 5)).astype(floatX)
    b_np = rng.normal(size=(5,)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)
    W = pt.tensor("W", shape=W_np.shape)
    b = pt.tensor("b", shape=b_np.shape)

    out = ConvLayer((4,), (1,), (1,))(X, W, b)
    compare_jax_and_py([X, W, b], out, [X_np, W_np, b_np], assert_fn=assert_close)


def test_a_conv1d_layer_matches_end_to_end(rng):
    """Through the layer rather than the op, so the padding the layer applies ahead of the op is in the
    graph too -- the dispatch assumes what reaches it is already padded."""
    X_np = rng.normal(size=(2, 10, 3)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)
    layer = Conv1D("conv", in_channels=3, out_channels=6, kernel_size=3, padding="same")
    layer.W.set_value(rng.normal(size=(3, 3, 6)).astype(floatX))
    layer.b.set_value(rng.normal(size=(6,)).astype(floatX))

    compare_jax_and_py([X], layer(X), [X_np], assert_fn=assert_close)


@pytest.mark.parametrize("spatial", [32, None], ids=["static_length", "dynamic_length"])
def test_an_undeclared_length_reaches_jax_only_through_the_dispatch(spatial, rng):
    """The gather builds its window indices from ``X.shape``, and JAX refuses a non-constant ``arange``.
    A declared spatial extent lets `fast_run` fold the shape away, so the graph alone would compile; an
    undeclared one does not fold, and the layer reaches JAX only because the forward and the pullback are
    both ops with dispatches of their own.

    Written against ``mode="JAX"`` rather than the stripped query the other tests here use, since that
    query drops `fast_run` and would hide the folding half of this entirely."""
    X = pt.tensor("X", shape=(8, spatial, 3), dtype=floatX)
    layer = Conv1D("conv", in_channels=3, out_channels=4, kernel_size=3)
    layer.W.set_value(rng.normal(size=(3, 3, 4)).astype(floatX))
    layer.b.set_value(rng.normal(size=(4,)).astype(floatX))
    out = layer(X)
    X_np = rng.normal(size=(8, 32, 3)).astype(floatX)

    assert_close(np.asarray(pytensor.function([X], out, mode="JAX")(X_np)), out.eval({X: X_np}))

    gradient = pt.grad((out**2).sum(), layer.W)
    assert_close(
        np.asarray(pytensor.function([X], gradient, mode="JAX")(X_np)),
        np.asarray(gradient.eval({X: X_np})),
    )


@pytest.mark.parametrize("spatial", [24, None], ids=["static_length", "dynamic_length"])
def test_the_gradient_dispatch_matches_the_graph(spatial, rng):
    """The pullback is its own op so a backend can differentiate its own convolution instead of running
    the gather's scatter-add. Checked against the same gradient on the default backend, which computes
    it from the graph -- two engines differentiating the same forward."""
    X = pt.tensor("X", shape=(4, spatial, 3), dtype=floatX)
    layer = Conv1D("conv", in_channels=3, out_channels=5, kernel_size=3)
    layer.W.set_value(rng.normal(size=(3, 3, 5)).astype(floatX))
    layer.b.set_value(rng.normal(size=(5,)).astype(floatX))
    cost = (layer(X) ** 2).sum()
    X_np = rng.normal(size=(4, 24, 3)).astype(floatX)

    for wrt in (layer.W, layer.b, X):
        gradient = pt.grad(cost, wrt)
        reference = pytensor.function([X], gradient)(X_np)
        assert_close(
            np.asarray(pytensor.function([X], gradient, mode="JAX")(X_np)), np.asarray(reference)
        )


def test_the_input_gradient_is_dropped_when_nothing_reads_it(rng):
    """`ConvLayer.pullback` always asks for both gradients; the rewrite lowers `compute_dX` where the
    input gradient has no clients, so the backend is asked for one gradient rather than two."""
    X = pt.tensor("X", shape=(4, 24, 3), dtype=floatX)
    layer = Conv1D("conv", in_channels=3, out_channels=5, kernel_size=3)
    cost = layer(X).sum()

    def grad_op(targets):
        fn = pytensor.function([X], targets, mode="JAX")
        return next(
            n.op for n in fn.maker.fgraph.apply_nodes if type(n.op).__name__ == "ConvLayerGrad"
        )

    assert grad_op([pt.grad(cost, layer.W)]).compute_dX is False
    assert grad_op(pt.grad(cost, [layer.W, X])).compute_dX is True
