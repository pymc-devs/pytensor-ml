import numpy as np
import pytest

from pytensor import config
from pytensor.tensor import lscalar

from pytensor_ml.optim import (
    chain,
    compile_train,
    cosine_annealing,
    scale_by_schedule,
    sgd,
)
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import function

# Every schedule takes (learning_rate, total_steps, min_learning_rate) and owes the same contract at the
# horizon, so those properties are asserted once for all of them.
SCHEDULES = [cosine_annealing]


def evaluate_schedule(schedule, steps):
    step_count = lscalar("step_count")
    rate_at = function([step_count], schedule(step_count))
    return np.array([rate_at(step) for step in steps])


@pytest.mark.parametrize("schedule_factory", SCHEDULES, ids=lambda factory: factory.__name__)
def test_schedule_decreases_monotonically(schedule_factory):
    rates = evaluate_schedule(schedule_factory(0.5, 10, min_learning_rate=0.05), range(11))
    assert np.all(np.diff(rates) < 0.0)


@pytest.mark.parametrize("schedule_factory", SCHEDULES, ids=lambda factory: factory.__name__)
def test_schedule_accepts_single_step_horizon(schedule_factory):
    rates = evaluate_schedule(schedule_factory(0.5, 1, min_learning_rate=0.05), [0, 1])
    np.testing.assert_allclose(rates, [0.5, 0.05], rtol=1e-6)


@pytest.mark.parametrize("schedule_factory", SCHEDULES, ids=lambda factory: factory.__name__)
def test_schedule_holds_floor_past_total_steps(schedule_factory):
    rates = evaluate_schedule(schedule_factory(1.0, 4, min_learning_rate=0.25), [4, 5, 100])
    np.testing.assert_allclose(rates, 0.25, rtol=1e-6)


@pytest.mark.parametrize("schedule_factory", SCHEDULES, ids=lambda factory: factory.__name__)
def test_schedule_reads_floatX_at_graph_build_time(schedule_factory):
    schedule = schedule_factory(1e-3, 10)
    with config.change_flags(floatX="float32"):
        assert schedule(lscalar("step_count")).type.dtype == "float32"
    with config.change_flags(floatX="float64"):
        assert schedule(lscalar("step_count")).type.dtype == "float64"


@pytest.mark.parametrize("schedule_factory", SCHEDULES, ids=lambda factory: factory.__name__)
@pytest.mark.parametrize("total_steps", [0, -1])
def test_schedule_rejects_empty_horizon(schedule_factory, total_steps):
    with pytest.raises(ValueError, match="total_steps must be at least 1"):
        schedule_factory(1e-3, total_steps)


def test_cosine_annealing_hits_curve_anchor_points():
    # A non-zero floor exercises the affine map onto [min_learning_rate, learning_rate]: the half cosine
    # runs 1 -> 0.5 -> 0, so the rate runs 1.0 -> 0.625 -> 0.25.
    rates = evaluate_schedule(cosine_annealing(1.0, 4, min_learning_rate=0.25), [0, 2, 4])
    np.testing.assert_allclose(rates, [1.0, 0.625, 0.25], rtol=1e-6)


@pytest.mark.parametrize("schedule_factory", SCHEDULES, ids=lambda factory: factory.__name__)
def test_schedule_drives_training_through_the_learning_rate_union(schedule_factory):
    """Passing a schedule as `learning_rate` substitutes it into the rule's own rate, which is a different
    path from `scale_by_schedule` and the one a new schedule is most likely to miss."""
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p, so the unit-rate base step is -p
    rule = sgd(learning_rate=schedule_factory(0.1, 2))
    published_rate = next(key for key in rule(loss, [p]) if key.name == "sgd/learning_rate")

    compile_train(loss, rule)()  # every schedule starts at its initial rate, so p = 2 - 0.1 * 2
    np.testing.assert_allclose(published_rate.get_value(), 0.1, rtol=1e-6)
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=1e-6)


def test_cosine_annealing_drives_training_step():
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p, so the unit-rate base step is -p
    rule = chain(sgd(learning_rate=1.0), scale_by_schedule(cosine_annealing(0.1, 2)))
    step = compile_train(loss, rule)

    step()  # step 0 -> lr = 0.1, p = 2 - 0.1 * 2 = 1.8
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=1e-6)
    step()  # step 1 -> lr = 0.05, p = 1.8 - 0.05 * 1.8 = 1.71
    np.testing.assert_allclose(p.get_value(), [1.71], rtol=1e-6)
    step()  # step 2 -> lr = 0, p unchanged
    np.testing.assert_allclose(p.get_value(), [1.71], rtol=1e-6)
