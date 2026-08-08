import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor import config

from pytensor_ml.optim import (
    add_weight_decay,
    chain,
    scalar_state,
    scale,
    scale_by_schedule,
    sgd_updates,
    trace,
)
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


def test_scale_by_schedule_applies_decaying_rate():
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p, so the unit-rate base step is -p

    def schedule(step_count):
        return 0.1 / (1.0 + step_count.astype("float64"))

    out = scale_by_schedule(schedule)(sgd_updates(loss, [p], learning_rate=1.0), [p])
    step = function([], loss, updates=out)

    step()  # step_count=0 -> lr=0.1, step=-2, p = 2 - 0.1 * 2 = 1.8
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=1e-6)
    step()  # step_count=1 -> lr=0.05, step=-1.8, p = 1.8 - 0.05 * 1.8 = 1.71
    np.testing.assert_allclose(p.get_value(), [1.71], rtol=1e-6)


def test_chain_reuses_transform_state_across_invocations():
    """A chain's transforms allocate state too, so two functions compiled from one chain must share it.
    Without reuse the second function silently restarts the schedule with its own step counter."""
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()
    rule = chain(trace(0.9), scale_by_schedule(lambda step_count: pt.constant(0.1)))

    first = {key for key in rule(sgd_updates(loss, [p], learning_rate=1.0), [p]) if key is not p}
    second = {key for key in rule(sgd_updates(loss, [p], learning_rate=1.0), [p]) if key is not p}

    assert first and first == second


def test_separately_configured_chains_keep_independent_state():
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()

    def build_updates():
        rule = chain(trace(0.9), scale_by_schedule(lambda step_count: pt.constant(0.1)))
        return {key for key in rule(sgd_updates(loss, [p], learning_rate=1.0), [p]) if key is not p}

    assert not build_updates() & build_updates()


def test_scale_by_schedule_publishes_the_applied_rate():
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()
    out = scale_by_schedule(lambda step_count: 0.1 / (1.0 + step_count.astype(config.floatX)))(
        sgd_updates(loss, [p], learning_rate=1.0), [p]
    )
    published_rate = next(key for key in out if key.name == "schedule/learning_rate")
    step = function([], loss, updates=out)

    # The rate published is the rate applied this step, not the one used on the step before: p goes
    # 2 -> 1.8 under lr=0.1, and the variable holds 0.1 rather than its initial 0.
    step()
    np.testing.assert_allclose(published_rate.get_value(), 0.1, rtol=1e-6)
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=1e-6)
    step()
    np.testing.assert_allclose(published_rate.get_value(), 0.05, rtol=1e-6)


def test_scale_by_schedule_publishes_to_a_caller_held_variable():
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()
    learning_rate = scalar_state("my/learning_rate")
    out = scale_by_schedule(lambda step_count: pt.constant(0.25), learning_rate=learning_rate)(
        sgd_updates(loss, [p], learning_rate=1.0), [p]
    )
    step = function([], loss, updates=out)

    step()
    np.testing.assert_allclose(learning_rate.get_value(), 0.25, rtol=1e-6)
    assert not any(key.name == "schedule/learning_rate" for key in out)


def test_two_scheduled_scalings_in_one_chain_raise():
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()
    constant_scaling = scale_by_schedule(lambda step_count: pt.constant(0.1))
    with pytest.raises(ValueError, match="Two scheduled scalings in one chain"):
        chain(constant_scaling, constant_scaling)(sgd_updates(loss, [p], learning_rate=1.0), [p])


def test_scale_by_schedule_casts_rate_to_parameter_dtype():
    # A float64 schedule must not upcast a float32 parameter's update, which pytensor rejects outright.
    with config.change_flags(floatX="float32"):
        p = trainable(np.array([2.0], dtype="float32"), name="w")
        loss = 0.5 * (p**2).sum()
        out = scale_by_schedule(lambda step_count: 0.1 / (1.0 + step_count.astype("float64")))(
            sgd_updates(loss, [p], learning_rate=1.0), [p]
        )
        published_rate = next(key for key in out if key.name == "schedule/learning_rate")
        assert out[p].type.dtype == "float32"
        assert out[published_rate].type.dtype == "float32"

        step = function([], loss, updates=out)
        step()
        np.testing.assert_allclose(p.get_value(), [1.8], rtol=1e-6)


def test_two_scheduled_scalings_compose_when_given_distinct_variables():
    """The workaround the rejection above recommends: distinct rate variables let two schedules stack.
    Each publishes its own factor, they share one step counter, and the parameters see the product."""
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()
    warmup_rate = scalar_state("warmup/learning_rate")
    decay_rate = scalar_state("decay/learning_rate")

    out = chain(
        scale_by_schedule(lambda step_count: pt.constant(0.5), learning_rate=warmup_rate),
        scale_by_schedule(lambda step_count: pt.constant(0.2), learning_rate=decay_rate),
    )(sgd_updates(loss, [p], learning_rate=1.0), [p])
    step_count = next(key for key in out if key.name == "schedule/step_count")
    step = function([], loss, updates=out)

    step()  # the applied rate is 0.5 * 0.2, so p = 2 - 0.1 * 2
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=1e-6)
    np.testing.assert_allclose(warmup_rate.get_value(), 0.5, rtol=1e-6)
    np.testing.assert_allclose(decay_rate.get_value(), 0.2, rtol=1e-6)
    assert int(step_count.get_value()) == 1  # one counter, advanced once: both schedules see one t


def test_add_weight_decay_subtracts_decay_term():
    p = trainable(np.array([4.0]), name="w")
    # An empty base step (updates[p] == p) isolates the decay term: new step = -0.1 * 4.
    out = chain(add_weight_decay(0.1), scale(1.0))({p: p}, [p])
    np.testing.assert_allclose(function([], out[p])(), [3.6])


def test_add_weight_decay_skips_masked_params():
    p = trainable(np.array([4.0]), name="bias")
    out = add_weight_decay(0.1, mask=lambda param: "bias" not in param.name)({p: p}, [p])
    np.testing.assert_allclose(function([], out[p])(), [4.0])
