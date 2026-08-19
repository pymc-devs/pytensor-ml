from functools import partial

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("torch")

from pytensor_ml.layers import Conv1D
from pytensor_ml.layers.conv import ConvLayer

floatX = pytensor.config.floatX
assert_close = partial(np.testing.assert_allclose, atol=1e-4, rtol=1e-3)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(sum(map(ord, "pytorch conv")))


@pytest.mark.parametrize(
    "stride, dilation",
    [(1, 1), (2, 1), (1, 2), (3, 2)],
    ids=["plain", "strided", "dilated", "both"],
)
def test_the_conv_op_matches_the_graph_it_replaces(stride, dilation, rng):
    """Torch is the one backend whose layouts disagree with ours at both ends, so the transposes are
    what this checks: an activation moved to channels-first and a kernel to output-channels-first."""
    X_np = rng.normal(size=(2, 16, 3)).astype(floatX)
    W_np = rng.normal(size=(3, 3, 4)).astype(floatX)
    X = pt.tensor("X", shape=X_np.shape)
    W = pt.tensor("W", shape=W_np.shape)

    out = ConvLayer((3,), (stride,), (dilation,))(X, W)
    assert_close(
        np.asarray(pytensor.function([X, W], out, mode="PYTORCH")(X_np, W_np)),
        out.eval({X: X_np, W: W_np}),
    )


def test_the_gradient_dispatch_matches_the_graph(rng):
    """`torch.autograd.grad` over torch's own convolution, against the same gradient computed from the
    graph on the default backend -- two engines differentiating one forward."""
    X = pt.tensor("X", shape=(4, 24, 3), dtype=floatX)
    layer = Conv1D("conv", in_channels=3, out_channels=5, kernel_size=3)
    layer.W.set_value(rng.normal(size=(3, 3, 5)).astype(floatX))
    layer.b.set_value(rng.normal(size=(5,)).astype(floatX))
    cost = (layer(X) ** 2).sum()
    X_np = rng.normal(size=(4, 24, 3)).astype(floatX)

    for wrt in (layer.W, layer.b, X):
        gradient = pt.grad(cost, wrt)
        assert_close(
            np.asarray(pytensor.function([X], gradient, mode="PYTORCH")(X_np)),
            np.asarray(pytensor.function([X], gradient)(X_np)),
        )


def test_the_input_gradient_is_dropped_when_nothing_reads_it(rng):
    """The rewrite lowers `compute_dX` where the input gradient has no clients, so torch is asked to
    differentiate with respect to the kernel alone."""
    X = pt.tensor("X", shape=(4, 24, 3), dtype=floatX)
    layer = Conv1D("conv", in_channels=3, out_channels=5, kernel_size=3)
    cost = layer(X).sum()

    def grad_op(targets):
        fn = pytensor.function([X], targets, mode="PYTORCH")
        return next(
            n.op for n in fn.maker.fgraph.apply_nodes if type(n.op).__name__ == "ConvLayerGrad"
        )

    assert grad_op([pt.grad(cost, layer.W)]).compute_dX is False
    assert grad_op(pt.grad(cost, [layer.W, X])).compute_dX is True
