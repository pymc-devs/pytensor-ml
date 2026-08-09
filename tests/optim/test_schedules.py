import numpy as np
import pytest

from pytensor import config
from pytensor.tensor import lscalar

from pytensor_ml.optim import (
    chain,
    compile_train,
    cosine_annealing,
    exponential_decay,
    linear_decay,
    polynomial_decay,
    scale_by_schedule,
    sgd,
    step_decay,
)
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import function

# Every schedule takes (learning_rate, total_steps, min_learning_rate) and owes the same contract at the
# horizon, so those properties are asserted once for all of them.
SCHEDULES = [cosine_annealing, linear_decay, exponential_decay, polynomial_decay]


def evaluate_schedule(schedule, steps):
    step_count = lscalar("step_count")
    rate_at = function([step_count], schedule(step_count))
    return np.array([rate_at(step) for step in steps])


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_decreases_monotonically(schedule_factory):
    rates = evaluate_schedule(schedule_factory(0.5, 10, min_learning_rate=0.05), range(11))
    assert np.all(np.diff(rates) < 0.0)


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_accepts_single_step_horizon(schedule_factory):
    rates = evaluate_schedule(schedule_factory(0.5, 1, min_learning_rate=0.05), [0, 1])
    np.testing.assert_allclose(rates, [0.5, 0.05], rtol=1e-6)


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_holds_floor_past_total_steps(schedule_factory):
    rates = evaluate_schedule(schedule_factory(1.0, 4, min_learning_rate=0.25), [4, 5, 100])
    np.testing.assert_allclose(rates, 0.25, rtol=1e-6)


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_reads_floatX_at_graph_build_time(schedule_factory):
    schedule = schedule_factory(1e-3, 10, 1e-5)
    with config.change_flags(floatX="float32"):
        assert schedule(lscalar("step_count")).type.dtype == "float32"
    with config.change_flags(floatX="float64"):
        assert schedule(lscalar("step_count")).type.dtype == "float64"


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_rejects_a_floor_above_the_initial_rate(schedule_factory):
    """The rates are adjacent positional arguments, so swapping them is easy and would otherwise produce a
    schedule that climbs while the docstring calls it a floor."""
    with pytest.raises(ValueError, match="min_learning_rate must not exceed learning_rate"):
        schedule_factory(0.001, 4, 0.1)


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_accepts_an_equal_floor_as_a_constant_rate(schedule_factory):
    # The floor check is `>`, not `>=`, so a floor equal to the initial rate is a constant schedule.
    rates = evaluate_schedule(schedule_factory(0.1, 4, min_learning_rate=0.1), range(6))
    np.testing.assert_allclose(rates, 0.1, rtol=1e-6)


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
@pytest.mark.parametrize("total_steps", [0, -1])
def test_schedule_rejects_empty_horizon(schedule_factory, total_steps):
    with pytest.raises(ValueError, match="total_steps must be at least 1"):
        schedule_factory(1e-3, total_steps, 1e-5)


def test_cosine_annealing_hits_curve_anchor_points():
    # A non-zero floor exercises the affine map onto [min_learning_rate, learning_rate]: the half cosine
    # runs 1 -> 0.5 -> 0, so the rate runs 1.0 -> 0.625 -> 0.25.
    rates = evaluate_schedule(cosine_annealing(1.0, 4, min_learning_rate=0.25), [0, 2, 4])
    np.testing.assert_allclose(rates, [1.0, 0.625, 0.25], rtol=1e-6)


def test_linear_decay_falls_by_a_constant_amount():
    # The constant decrement is what separates linear from every other decay: 0.75 spread over 4 steps.
    rates = evaluate_schedule(linear_decay(1.0, 4, min_learning_rate=0.25), range(5))
    np.testing.assert_allclose(np.diff(rates), -0.1875, rtol=1e-6)
    np.testing.assert_allclose(rates[[0, -1]], [1.0, 0.25], rtol=1e-6)


def test_linear_decay_holds_the_initial_rate_until_transition_begin():
    # total_steps is the length of the decay itself, so the floor arrives at transition_begin + total_steps.
    schedule = linear_decay(1.0, 4, min_learning_rate=0.25, transition_begin=3)
    rates = evaluate_schedule(schedule, [0, 3, 4, 5, 7, 100])
    np.testing.assert_allclose(rates, [1.0, 1.0, 0.8125, 0.625, 0.25, 0.25], rtol=1e-6)


@pytest.mark.parametrize("schedule_factory", [linear_decay, exponential_decay, polynomial_decay])
def test_schedule_holds_the_initial_rate_until_transition_begin(schedule_factory):
    """Each schedule has to pass its own `transition_begin` through, so the shared clamp being right is not
    enough — asserted as a property because the curve between B and B + T differs per schedule."""
    rates = evaluate_schedule(
        schedule_factory(1.0, 4, 0.25, transition_begin=3), [0, 1, 3, 4, 7, 100]
    )

    np.testing.assert_allclose(rates[:3], 1.0, rtol=1e-6)  # held through step B
    assert rates[3] < 1.0  # decaying afterwards
    np.testing.assert_allclose(rates[4:], 0.25, rtol=1e-6)  # floor from step B + T


def test_polynomial_decay_bends_with_the_power():
    # (1 - p) ** 2 over four steps: quick drop, then a long flat tail.
    rates = evaluate_schedule(polynomial_decay(1.0, 4, power=2.0), range(5))
    np.testing.assert_allclose(rates, [1.0, 0.5625, 0.25, 0.0625, 0.0], rtol=1e-6)


def test_polynomial_decay_at_power_one_matches_linear_decay():
    """`linear_decay` is the `power=1` case, and the two are separate implementations, so pin the identity
    rather than trusting that they stay in step."""
    steps = range(6)
    polynomial = evaluate_schedule(
        polynomial_decay(1.0, 4, min_learning_rate=0.25, power=1.0), steps
    )
    linear = evaluate_schedule(linear_decay(1.0, 4, min_learning_rate=0.25), steps)
    np.testing.assert_allclose(polynomial, linear, rtol=1e-6)


def test_polynomial_decay_reaches_the_floor_at_a_fractional_power():
    # A power below one holds the rate high and then drops, and raises exactly zero to a fractional
    # exponent at the horizon, which is where a NaN would appear.
    rates = evaluate_schedule(polynomial_decay(1.0, 4, 0.25, power=0.5), [0, 2, 4, 100])
    np.testing.assert_allclose(rates, [1.0, 0.75 * 2**-0.5 + 0.25, 0.25, 0.25], rtol=1e-6)
    assert rates[1] > 0.625  # above the straight line, which passes through 0.625 at the midpoint


@pytest.mark.parametrize("schedule_factory", [linear_decay, exponential_decay, polynomial_decay])
def test_delayed_schedules_agree_on_their_positional_arguments(schedule_factory):
    """Copying a call from one schedule to another must not change what it means, so the fourth positional
    argument is `transition_begin` in every schedule that takes a delay — not, say, a curve parameter."""
    steps = [0, 3, 4, 7]
    positional = evaluate_schedule(schedule_factory(1.0, 4, 0.25, 3), steps)
    keyword = evaluate_schedule(schedule_factory(1.0, 4, 0.25, transition_begin=3), steps)
    np.testing.assert_allclose(positional, keyword, rtol=1e-6)


@pytest.mark.parametrize("power", [0.0, -1.0])
def test_polynomial_decay_rejects_a_non_positive_power(power):
    with pytest.raises(ValueError, match="power must be positive"):
        polynomial_decay(0.1, 4, power=power)


def test_exponential_decay_falls_by_a_constant_factor():
    # The constant ratio is what separates geometric decay from linear: 1.0 -> 0.0625 over 4 steps halves.
    rates = evaluate_schedule(exponential_decay(1.0, 4, 0.0625), range(5))
    np.testing.assert_allclose(rates[1:] / rates[:-1], 0.5, rtol=1e-6)
    np.testing.assert_allclose(rates[[0, -1]], [1.0, 0.0625], rtol=1e-6)


@pytest.mark.parametrize(
    ("learning_rate", "min_learning_rate"),
    [(0.1, 0.0), (0.0, 0.0)],
    ids=["zero_floor", "zero_initial"],
)
def test_exponential_decay_rejects_a_non_positive_rate(learning_rate, min_learning_rate):
    with pytest.raises(ValueError, match="needs positive rates"):
        exponential_decay(learning_rate, 4, min_learning_rate)


@pytest.mark.parametrize("schedule_factory", [linear_decay, exponential_decay, polynomial_decay])
def test_schedule_rejects_negative_transition_begin(schedule_factory):
    with pytest.raises(ValueError, match="transition_begin must not be negative"):
        schedule_factory(1e-3, 10, 1e-5, transition_begin=-1)


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_drives_training_through_the_learning_rate_union(schedule_factory):
    """Passing a schedule as `learning_rate` substitutes it into the rule's own rate, which is a different
    path from `scale_by_schedule` and the one a new schedule is most likely to miss."""
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p, so the unit-rate base step is -p
    rule = sgd(learning_rate=schedule_factory(0.1, 2, 0.01))
    published_rate = next(key for key in rule(loss, [p]) if key.name == "sgd/learning_rate")

    compile_train(loss, rule)()  # every schedule starts at its initial rate, so p = 2 - 0.1 * 2
    np.testing.assert_allclose(published_rate.get_value(), 0.1, rtol=1e-6)
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=1e-6)


# Geometric decay never reaches zero, so the schedule that cannot hit a zero floor sits this one out.
@pytest.mark.parametrize("schedule_factory", [cosine_annealing, linear_decay])
def test_schedule_drives_training_through_scale_by_schedule(schedule_factory):
    # Decaying to zero over two steps, every curve passes through the same midpoint, so one set of expected
    # values covers them: 0.1 -> 0.05 -> 0.
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p, so the unit-rate base step is -p
    rule = chain(sgd(learning_rate=1.0), scale_by_schedule(schedule_factory(0.1, 2)))
    step = compile_train(loss, rule)

    step()  # step 0 -> lr = 0.1, p = 2 - 0.1 * 2 = 1.8
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=1e-6)
    step()  # step 1 -> lr = 0.05, p = 1.8 - 0.05 * 1.8 = 1.71
    np.testing.assert_allclose(p.get_value(), [1.71], rtol=1e-6)
    step()  # step 2 -> lr = 0, p unchanged
    np.testing.assert_allclose(p.get_value(), [1.71], rtol=1e-6)


# step_decay decays indefinitely instead of over a horizon, so it takes (decay_every, decay_factor) rather
# than (total_steps, min_learning_rate) positionally and sits outside the shared contract tests above.
def test_step_decay_drops_by_a_factor_at_each_interval():
    rates = evaluate_schedule(step_decay(1.0, decay_every=3, decay_factor=0.5), range(10))
    expected = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.125]
    np.testing.assert_allclose(rates, expected, rtol=1e-6)


def test_step_decay_stops_at_the_floor():
    schedule = step_decay(1.0, decay_every=3, decay_factor=0.5, min_learning_rate=0.3)
    rates = evaluate_schedule(schedule, [3, 6, 9, 100])
    np.testing.assert_allclose(rates, [0.5, 0.3, 0.3, 0.3], rtol=1e-6)


def test_step_decay_delays_the_first_drop_until_transition_begin():
    schedule = step_decay(1.0, decay_every=3, decay_factor=0.5, transition_begin=2)
    rates = evaluate_schedule(schedule, [0, 4, 5, 8])
    np.testing.assert_allclose(rates, [1.0, 1.0, 0.5, 0.25], rtol=1e-6)


def test_step_decay_underflows_to_zero_without_a_floor():
    # A long run multiplies the factor thousands of times; the result has to reach zero rather than a NaN.
    rate = evaluate_schedule(step_decay(1.0, decay_every=1, decay_factor=0.5), [10_000])
    np.testing.assert_array_equal(rate, [0.0])


def test_step_decay_requires_keyword_arguments():
    """The positional slots mean something different here than in the horizon-based schedules, so passing
    them positionally is an error rather than a silent reinterpretation."""
    with pytest.raises(TypeError, match="positional"):
        step_decay(1.0, 3)  # type: ignore[misc]


def test_step_decay_reads_floatX_at_graph_build_time():
    # Not covered by the shared contract test, and this is the one schedule with an integer intermediate
    # (the drop count), so it is the likeliest to leak a float64 rate into a float32 graph.
    schedule = step_decay(1e-3, decay_every=10, min_learning_rate=1e-5)
    with config.change_flags(floatX="float32"):
        assert schedule(lscalar("step_count")).type.dtype == "float32"
    with config.change_flags(floatX="float64"):
        assert schedule(lscalar("step_count")).type.dtype == "float64"


def test_step_decay_accepts_a_unit_factor_as_a_constant_rate():
    # The factor check admits 1.0, which is a flat schedule rather than a rejected input.
    rates = evaluate_schedule(step_decay(0.1, decay_every=3, decay_factor=1.0), [0, 3, 30])
    np.testing.assert_allclose(rates, 0.1, rtol=1e-6)


@pytest.mark.parametrize("decay_every", [0, -1])
def test_step_decay_rejects_a_non_positive_interval(decay_every):
    with pytest.raises(ValueError, match="decay_every must be at least 1"):
        step_decay(1.0, decay_every=decay_every)


@pytest.mark.parametrize("decay_factor", [0.0, -0.5, 1.5])
def test_step_decay_rejects_a_factor_outside_the_unit_interval(decay_factor):
    with pytest.raises(ValueError, match=r"decay_factor must be in \(0, 1]"):
        step_decay(1.0, decay_every=3, decay_factor=decay_factor)


def test_step_decay_rejects_a_floor_above_the_initial_rate():
    with pytest.raises(ValueError, match="min_learning_rate must not exceed learning_rate"):
        step_decay(0.001, decay_every=3, min_learning_rate=0.1)
