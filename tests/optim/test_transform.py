import numpy as np
import pytensor.tensor as pt

from pytensor_ml.optim import add_weight_decay, chain, scale, sgd_updates, trace
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import function


def test_scale_applies_factor():
    p = trainable(np.zeros(2), name="w")
    updates = {p: p + pt.constant(np.array([2.0, -4.0]))}
    out = scale(0.25)(updates, [p])
    np.testing.assert_allclose(function([], out[p])(), [0.5, -1.0])


def test_trace_accumulates_velocity_with_decay():
    p = trainable(np.zeros(1), name="w")
    out = trace(0.9)({p: p + pt.constant(np.array([1.0]))}, [p])
    velocity = next(key for key in out if key.name == "w/trace/velocity")
    step = function([], [out[p], out[velocity]], updates=out)

    # Step 1 leaves the decay unexercised: velocity = 0.9 * 0 + 1 = 1, p = 0 + 1 = 1.
    new_p, new_velocity = step()
    np.testing.assert_allclose(new_velocity, [1.0])
    np.testing.assert_allclose(new_p, [1.0])
    # Step 2 exercises the decay: velocity = 0.9 * 1 + 1 = 1.9, p = 1 + 1.9 = 2.9.
    new_p, new_velocity = step()
    np.testing.assert_allclose(new_velocity, [1.9])
    np.testing.assert_allclose(new_p, [2.9])


def test_nesterov_differs_from_classical():
    p = trainable(np.zeros(1), name="w")
    step = pt.constant(np.array([1.0]))
    classical = trace(0.9, nesterov=False)({p: p + step}, [p])
    nesterov = trace(0.9, nesterov=True)({p: p + step}, [p])
    # classical -> velocity = 1; nesterov -> step + decay*velocity = 1 + 0.9 = 1.9
    np.testing.assert_allclose(function([], classical[p])(), [1.0])
    np.testing.assert_allclose(function([], nesterov[p])(), [1.9])


def test_chain_threads_updates_in_order():
    p = trainable(np.ones(1), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p = 1, so unit-rate sgd step is -1
    out = chain(trace(0.9), scale(0.1))(sgd_updates(loss, [p], learning_rate=1.0), [p])

    assert p in out
    assert any(k.name == "w/trace/velocity" for k in out)
    # velocity = -1, scaled step = 0.1 * -1, new p = 1 - 0.1 = 0.9
    np.testing.assert_allclose(function([], out[p])(), [0.9])


def test_chain_reuses_transform_state_across_invocations():
    """A chain's transforms allocate state too, so two functions compiled from one chain must share it.
    Without reuse the second function silently restarts with its own velocity buffer."""
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()
    rule = chain(trace(0.9), scale(0.1))

    first = {key for key in rule(sgd_updates(loss, [p], learning_rate=1.0), [p]) if key is not p}
    second = {key for key in rule(sgd_updates(loss, [p], learning_rate=1.0), [p]) if key is not p}

    assert first and first == second


def test_separately_configured_chains_keep_independent_state():
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()

    def build_updates():
        rule = chain(trace(0.9), scale(0.1))
        return {key for key in rule(sgd_updates(loss, [p], learning_rate=1.0), [p]) if key is not p}

    assert not build_updates() & build_updates()


def test_add_weight_decay_subtracts_decay_term():
    p = trainable(np.array([4.0]), name="w")
    # An empty base step (updates[p] == p) isolates the decay term: new step = -0.1 * 4.
    out = chain(add_weight_decay(0.1), scale(1.0))({p: p}, [p])
    np.testing.assert_allclose(function([], out[p])(), [3.6])


def test_add_weight_decay_skips_masked_params():
    p = trainable(np.array([4.0]), name="bias")
    out = add_weight_decay(0.1, mask=lambda param: "bias" not in param.name)({p: p}, [p])
    np.testing.assert_allclose(function([], out[p])(), [4.0])
