import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor_ml.optim import (
    adadelta,
    adadelta_updates,
    adagrad,
    adagrad_updates,
    adam,
    adam_updates,
    adamw,
    rmsprop,
    rmsprop_updates,
    sgd,
    sgd_updates,
)
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import function


@pytest.mark.parametrize(
    "rule",
    [
        sgd(learning_rate=1e-2),
        sgd(learning_rate=1e-2, momentum=0.9),
        sgd(learning_rate=1e-2, momentum=0.9, nesterov=True),
        adam(learning_rate=1e-2),
        adamw(learning_rate=1e-2, weight_decay=1e-2),
        adagrad(learning_rate=1e-1),
        adadelta(learning_rate=1.0),
        rmsprop(learning_rate=1e-2),
        rmsprop(learning_rate=1e-2, momentum=0.9),
        rmsprop(learning_rate=1e-2, centered=True),
    ],
    ids=[
        "sgd",
        "sgd_momentum",
        "sgd_nesterov",
        "adam",
        "adamw",
        "adagrad",
        "adadelta",
        "rmsprop",
        "rmsprop_momentum",
        "rmsprop_centered",
    ],
)
def test_rule_reduces_loss(run_training, rule):
    history = run_training(rule, n_steps=100)
    assert history[-1] < history[0]


def test_adam_first_step_is_sign_descent():
    """Bias correction makes Adam's first step exactly ``lr * sign(g)`` per coordinate: the corrected moments
    are ``m_hat = g`` and ``v_hat = g**2``, so the step is ``lr * g / (|g| + eps)``, independent of the
    gradient magnitude."""
    start = np.array([1.0, -2.0, 100.0])  # gradients span two orders of magnitude
    p = trainable(start.copy(), name="w")
    loss = 0.5 * (p**2).sum()  # gradient is exactly p
    lr = 0.1
    function([], loss, updates=adam_updates(loss, [p], learning_rate=lr))()

    step = start - p.get_value()
    np.testing.assert_allclose(step, lr * np.sign(start), rtol=1e-6)


def test_adam_updates_keyed_by_object_with_named_state():
    """State is discovered by object identity; names exist only for serialization."""
    p = trainable(np.zeros(3), name="w")
    loss = (p**2).sum()
    updates = adam_updates(loss, [p])

    assert p in updates  # the exact param object is a key, not a renamed copy
    state_names = {key.name for key in updates if key is not p}
    assert state_names == {"adam/step_count", "w/adam/first_moment", "w/adam/second_moment"}


def test_adagrad_step_decays_as_inverse_sqrt_t():
    """Under a constant gradient the accumulator grows as ``t * g**2``, so AdaGrad's step magnitude decays as
    ``lr / sqrt(t)`` for every coordinate — independent of the gradient magnitude itself."""
    start = np.array([5.0, -3.0])
    p = trainable(start.copy(), name="w")
    g0 = np.array([2.0, -0.5])  # 4x apart, yet both coordinates take the same step size
    loss = (pt.constant(g0) * p).sum()  # constant gradient g0, independent of p
    lr, n_steps = 0.1, 6
    fn = function([], loss, updates=adagrad_updates(loss, [p], learning_rate=lr))

    previous = start.copy()
    for t in range(1, n_steps + 1):
        fn()
        current = p.get_value()
        step_magnitude = np.abs(current - previous)
        np.testing.assert_allclose(step_magnitude, lr / np.sqrt(t), rtol=1e-4)
        previous = current


def test_adadelta_is_invariant_to_gradient_scale():
    """AdaDelta needs no learning-rate tuning because its update is invariant to the scale of the gradient:
    the ``sqrt(accumulated_update) / sqrt(accumulated_gradient)`` ratio cancels any constant factor on the
    loss. Scaling the loss 100x leaves the parameter trajectory unchanged."""
    start = np.array([1.0, -2.0])

    def trajectory(loss_scale):
        p = trainable(start.copy(), name="w")
        loss = 0.5 * loss_scale * (p**2).sum()  # gradient is loss_scale * p
        fn = function([], loss, updates=adadelta_updates(loss, [p], learning_rate=1.0, rho=0.9))
        values = []
        for _ in range(5):
            fn()
            values.append(p.get_value())
        return np.array(values)

    np.testing.assert_allclose(trajectory(1.0), trajectory(100.0), rtol=1e-4)


def test_rmsprop_first_step_normalizes_gradient_magnitude():
    """RMSProp's defining behavior: the first step size depends on ``learning_rate`` and ``rho`` alone, not
    on the gradient magnitude. With ``v_1 = (1 - rho) g**2`` the step is ``lr * g / sqrt((1 - rho) g**2) =
    lr / sqrt(1 - rho)`` along ``-sign(g)`` — identical for every coordinate no matter how large its gradient.
    """
    start = np.array([1.0, -2.0, 100.0])  # gradients span two orders of magnitude
    p = trainable(start.copy(), name="w")
    loss = 0.5 * (p**2).sum()  # gradient is exactly p
    lr, rho = 0.1, 0.9
    function([], loss, updates=rmsprop_updates(loss, [p], learning_rate=lr, rho=rho))()

    step = start - p.get_value()
    expected_magnitude = lr / np.sqrt(1 - rho)
    np.testing.assert_allclose(step, expected_magnitude * np.sign(start), rtol=1e-4)


def test_rmsprop_centered_first_step_uses_centered_variance():
    """Centering subtracts the squared running-mean gradient from the second moment. After one step the
    variance is ``(1 - rho) * rho * g**2``, so the step magnitude is ``lr / sqrt(rho * (1 - rho))`` — larger
    than the uncentered ``lr / sqrt(1 - rho)`` by ``1 / sqrt(rho)``, and still independent of gradient scale.
    """
    start = np.array([1.0, -2.0, 100.0])  # gradients span two orders of magnitude
    p = trainable(start.copy(), name="w")
    loss = 0.5 * (p**2).sum()  # gradient is exactly p
    lr, rho = 0.1, 0.9
    function([], loss, updates=rmsprop_updates(loss, [p], learning_rate=lr, rho=rho, centered=True))()

    step = start - p.get_value()
    expected_magnitude = lr / np.sqrt(rho * (1 - rho))
    np.testing.assert_allclose(step, expected_magnitude * np.sign(start), rtol=1e-4)


def test_precomputed_gradients_accepted():
    p = trainable(np.ones(2), name="w")
    gradients = [pt.constant(np.array([0.5, -0.5]))]
    updates = sgd(learning_rate=1.0)(gradients, [p])
    np.testing.assert_allclose(function([], updates[p])(), [0.5, 1.5])


def test_get_gradients_rejects_count_mismatch():
    weight = trainable(np.ones(2), name="w")
    bias = trainable(np.ones(2), name="b")
    one_gradient = [pt.constant(np.ones(2))]
    with pytest.raises(ValueError, match="1 gradients for 2 parameters"):
        sgd_updates(one_gradient, [weight, bias])
