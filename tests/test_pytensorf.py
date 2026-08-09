import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor import config, shared
from pytensor.gradient import (
    DisconnectedInputError,
    disconnected_grad,
    grad,
    grad_clip,
    zero_grad,
)
from pytensor.graph.traversal import ancestors

from pytensor_ml.layers import Dropout, DropoutLayer, Linear, Sequential
from pytensor_ml.pytensorf import (
    collect_trainable_params,
    compile_predict,
    rewrite_for_prediction,
    rewrite_pregrad,
)
from pytensor_ml.state import initialize_params


def contains_op(graph, op_type):
    return any(isinstance(var.owner.op, op_type) for var in ancestors([graph]) if var.owner)


def test_compile_predict_removes_dropout():
    # The inference rewrite drops Dropout, so repeated calls are deterministic; without it they would differ.
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc", n_in=4, n_out=4), Dropout(p=0.5))(X)
    parameters = collect_trainable_params(prediction)
    for parameter, value in zip(
        parameters, initialize_params(parameters, rng=np.random.default_rng(0))
    ):
        parameter.set_value(value)

    predict = compile_predict(prediction, inputs=[X])
    X_values = np.random.default_rng(0).normal(size=(8, 4)).astype(config.floatX)
    first, second = predict(X_values), predict(X_values)

    # Guards the test itself: at the zero initialization the output is identically zero, and so stable
    # across calls whether or not dropout was removed.
    assert np.any(first != 0)
    np.testing.assert_allclose(first, second)


def test_rewrite_for_prediction_leaves_the_original_graph_intact():
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc", n_in=4, n_out=4), Dropout(p=0.5))(X)

    assert not contains_op(rewrite_for_prediction(prediction), DropoutLayer)
    assert contains_op(prediction, DropoutLayer)


@pytest.mark.parametrize(
    "marker, expected_gradient",
    [
        (zero_grad, [0.0, 0.0, 0.0]),
        (lambda W: grad_clip(W, -1.0, 1.0), [1.0, 1.0, 1.0]),
    ],
    ids=["zero_grad", "grad_clip"],
)
def test_rewrite_pregrad_keeps_gradient_markers(marker, expected_gradient):
    # Every marker is a ViewOp, which canonicalize is otherwise free to splice out. Unmarked, the gradient
    # of this loss is 2 * W, so both expectations only hold while the marker survives into grad.
    W = shared(np.array([1.0, 2.0, 3.0], dtype=config.floatX), name="W")
    loss = (marker(W) ** 2).sum()

    gradient = grad(rewrite_pregrad(loss), W)

    np.testing.assert_allclose(gradient.eval(), expected_gradient)


def test_rewrite_pregrad_keeps_a_stop_gradient_out_of_the_gradient():
    W = shared(np.array([1.0, 2.0, 3.0], dtype=config.floatX), name="W")
    # Differentiating this reads the detached factor as a constant, so the gradient is W itself; with the
    # marker gone the graph is W ** 2 and the gradient comes out twice as large.
    loss = (W * disconnected_grad(W)).sum()

    gradient = grad(rewrite_pregrad(loss), W)

    np.testing.assert_allclose(gradient.eval(), W.get_value())


def test_rewrite_pregrad_leaves_a_fully_detached_parameter_disconnected():
    W = shared(np.array([1.0, 2.0, 3.0], dtype=config.floatX), name="W")
    loss = (disconnected_grad(W) ** 2).sum()

    with pytest.raises(DisconnectedInputError):
        grad(rewrite_pregrad(loss), W)
