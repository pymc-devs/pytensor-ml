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
from pytensor.tensor import random as ptr

from pytensor_ml.layers import Dropout, DropoutLayer, Linear, Sequential
from pytensor_ml.pytensorf import (
    collect_trainable_params,
    compile_predict,
    function,
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


def test_compiling_leaves_a_generator_another_function_is_drawing_from_alone():
    """Compiling used to replace every generator it touched, so building a second function mid-training
    jumped the first one's noise stream."""

    def four_draws(with_intervening_compile):
        rng = shared(np.random.default_rng(0), name="rng")
        _, noise = ptr.normal(rng=rng, return_next_rng=True)
        draw = function([], noise)
        drawn = [float(draw()) for _ in range(2)]
        if with_intervening_compile:
            _, other_noise = ptr.normal(rng=rng, return_next_rng=True)
            function([], other_noise)
        return drawn + [float(draw()) for _ in range(2)]

    assert four_draws(with_intervening_compile=True) == four_draws(with_intervening_compile=False)


@pytest.mark.parametrize("applications", [1, 2], ids=["applied_once", "applied_twice"])
def test_a_seeded_dropout_reproduces_across_identical_runs(applications):
    """The seed a caller puts on a layer has to survive compilation, or `random_state=` means nothing. It has
    to hold however many times the layer is applied, since each application draws off its own generator."""

    def dropout_masks():
        X = pt.tensor("X", shape=(None, 3))
        dropout = Dropout(p=0.5, random_state=0)
        layers = [Linear("fc", n_in=3, n_out=3)]
        for _ in range(applications):
            layers.append(dropout)
        prediction = Sequential(*layers)(X)
        for parameter in collect_trainable_params(prediction):
            parameter.set_value(np.ones_like(parameter.get_value()))
        forward = function([X], prediction)
        features = np.ones((4, 3), dtype=config.floatX)
        return [forward(features) for _ in range(3)]

    for first, second in zip(dropout_masks(), dropout_masks()):
        np.testing.assert_allclose(first, second)


def test_random_seed_makes_an_unseeded_graph_reproducible():
    def two_draws():
        rng = shared(np.random.default_rng(), name="rng")  # deliberately unseeded
        _, noise = ptr.normal(rng=rng, return_next_rng=True)
        draw = function([], noise, random_seed=42)
        return [float(draw()) for _ in range(2)]

    assert two_draws() == two_draws()


def test_a_reused_dropout_instance_keeps_drawing_new_masks():
    """Using one Dropout object at two points in a network is an ordinary thing to write, and it used to
    freeze the mask for the whole run."""
    X = pt.tensor("X", shape=(None, 3))
    dropout = Dropout(p=0.5, random_state=0)
    prediction = Sequential(
        Linear("fc1", n_in=3, n_out=4), dropout, Linear("fc2", n_in=4, n_out=2), dropout
    )(X)
    for parameter in collect_trainable_params(prediction):
        parameter.set_value(np.ones_like(parameter.get_value()))

    forward = function([X], prediction)
    features = np.ones((4, 3), dtype=config.floatX)
    outputs = [forward(features) for _ in range(4)]

    assert (
        len(dropout.generators) == 2
    )  # one per application, so neither draw is starved of updates
    assert all(
        np.any(outputs[i] != outputs[j]) for i in range(4) for j in range(i + 1, 4)
    )  # a fresh mask on every call, not just eventually
