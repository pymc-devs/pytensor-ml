from functools import cache

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from scipy.optimize import line_search as scipy_line_search

from pytensor_ml.optim.line_search import (
    STEP,
    LineSearchOp,
    find_piece,
    search_along,
    zoom_line_search,
)

X = pt.vector("x", dtype="float64")
D = pt.vector("d", dtype="float64")


def rosenbrock(x, xp=pt):
    return xp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


OBJECTIVES = {
    "quadratic": lambda x, xp=pt: xp.sum((x - 3.0) ** 2) / 2,
    "rosenbrock": rosenbrock,
    "log_barrier": lambda x, xp=pt: xp.sum((x - 0.9) ** 2 - xp.log(1 - x)),
}


@cache
def compiled(objective, max_steps=20):
    """One search function per objective and budget, with the point and the direction as inputs, so
    every case on an objective runs on one compiled graph."""
    loss = OBJECTIVES[objective](X)
    [gradient] = pt.grad(loss, [X])
    result = search_along(loss, [X], [gradient], [D], zoom_line_search(max_steps))
    return pytensor.function([X, D], list(result[:3])), pytensor.function([X], gradient)


def search(objective, x0, direction_of, max_steps=20):
    run, gradient = compiled(objective, max_steps)
    direction = direction_of(gradient(x0))
    step_size, failed, evaluations = run(x0, direction)
    return float(step_size), bool(failed), int(evaluations), direction


ROSENBROCK_START = np.array([-1.2, 1.0])

# Each row is (objective, start, direction from the gradient, budget) and optax 0.2.8's answer from
# `scale_by_zoom_linesearch(max_linesearch_steps=budget, initial_guess_strategy="one")`: the step it
# takes, whether it failed, and how many trials it evaluated.
PINNED = {
    "accepts_the_guess": (("quadratic", np.zeros(2), lambda g: np.ones(2), 20), (1.0, False, 1)),
    "zooms_into_a_bracket": (
        ("rosenbrock", ROSENBROCK_START, lambda g: -g, 20),
        (0.0009383102759, False, 10),
    ),
    "grows_until_it_brackets": (
        ("rosenbrock", ROSENBROCK_START, lambda g: -g / np.linalg.norm(g) * 1e-3, 20),
        (16.0, False, 5),
    ),
    "backs_out_of_a_nan_region": (
        ("log_barrier", np.zeros(1), lambda g: np.array([2.0]), 20),
        (0.1128293312, False, 4),
    ),
    "fails_uphill_and_keeps_the_last_trial": (
        ("rosenbrock", ROSENBROCK_START, lambda g: g, 20),
        (4.031038437e-07, True, 20),
    ),
    "fails_when_out_of_steps": (("rosenbrock", ROSENBROCK_START, lambda g: -g, 2), (0.5, True, 2)),
}


@pytest.mark.parametrize("case, expected", PINNED.values(), ids=PINNED.keys())
def test_the_search_takes_the_step_optax_takes(case, expected):
    """The port follows optax's search trial for trial, so the step, the failure flag and the number of
    trials all agree, in the cases that exercise each branch: the guess accepted at once, the bracket
    grown, the bracket zoomed, a NaN trial backed out of, and both ways of failing."""
    step_size, failed, evaluations, _ = search(*case)

    np.testing.assert_allclose(step_size, expected[0], rtol=1e-9)
    assert (failed, evaluations) == expected[1:]


@pytest.mark.parametrize(
    "name", ["accepts_the_guess", "zooms_into_a_bracket", "grows_until_it_brackets"]
)
def test_the_search_agrees_with_scipys_strong_wolfe_search(name):
    """An implementation that shares no code with the port, and lands on the same step wherever the
    two searches visit the same trials."""
    case, _ = PINNED[name]
    objective, x0, _, max_steps = case
    step_size, _, _, direction = search(*case)

    f = OBJECTIVES[objective]
    _, gradient = compiled(objective, max_steps)
    expected = scipy_line_search(
        lambda x: float(f(x, np)), gradient, x0, direction, c1=1e-4, c2=0.9, maxiter=max_steps
    )[0]

    np.testing.assert_allclose(step_size, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "name", ["zooms_into_a_bracket", "grows_until_it_brackets", "backs_out_of_a_nan_region"]
)
def test_an_accepted_step_satisfies_the_strong_wolfe_conditions(name):
    """Checked directly on the objective rather than against another search: sufficient decrease, by
    Armijo or by the approximate Wolfe test, and a slope flattened to within ``curv_rtol``."""
    case, _ = PINNED[name]
    objective, x0, _, max_steps = case
    step_size, failed, _, direction = search(*case)
    f = OBJECTIVES[objective]
    _, gradient = compiled(objective, max_steps)

    value0, slope0 = f(x0, np), gradient(x0) @ direction
    value, slope = (
        f(x0 + step_size * direction, np),
        gradient(x0 + step_size * direction) @ direction,
    )
    armijo = value <= value0 + 1e-4 * step_size * slope0
    approximate_wolfe = slope <= (2e-4 - 1) * slope0 and value <= value0 + 1e-6 * abs(value0)

    assert not failed
    assert armijo or approximate_wolfe
    assert abs(slope) <= 0.9 * abs(slope0)


# Each row is (search arguments, first trial) and the step, failure flag and trial count on the quadratic
# from zero along ones, where the slope at a step t is 2 * (t - 3) against -6 at the start.
FIRST_TRIALS = {
    "stops_at_the_cap": (({"max_learning_rate": 0.1}, 1.0), (0.1, False, 1)),
    "falls_back_onto_the_cap": (({"max_steps": 1, "max_learning_rate": 5.9}, 5.9), (5.9, True, 1)),
    "accepts_a_guess_other_than_one": (({}, 3.0), (3.0, False, 1)),
}


@pytest.mark.parametrize("case, expected", FIRST_TRIALS.values(), ids=FIRST_TRIALS.keys())
def test_the_first_trial_follows_the_guess_and_the_cap(case, expected):
    """At 0.1 the cap stops a search that has not bracketed, which counts as done. At 5.9 the one trial
    allowed misses the curvature condition, so the search fails and falls back on that trial, which met
    sufficient decrease. A guess of 3, the minimizer, is accepted where a guess of one would be too."""
    arguments, guess = case
    loss = OBJECTIVES["quadratic"](X)
    result = search_along(
        loss, [X], [pt.grad(loss, X)], [D], zoom_line_search(**arguments), guess=guess
    )

    step_size, failed, evaluations = pytensor.function([X, D], list(result[:3]))(
        np.zeros(2), np.ones(2)
    )

    assert (float(step_size), bool(failed), int(evaluations)) == expected


def test_a_search_from_a_nan_point_takes_no_step():
    """No trial can decrease a loss that is already NaN, and moving anyway would carry the NaN into
    the parameters, so the search falls back on a step of zero."""
    step_size, failed, evaluations, _ = search("log_barrier", np.array([1.5]), lambda g: np.ones(1))

    assert (step_size, failed, evaluations) == (0.0, True, 20)


def test_the_search_is_one_node():
    loss = OBJECTIVES["quadratic"](X)
    result = search_along(loss, [X], [pt.grad(loss, X)], [D], zoom_line_search())

    assert isinstance(result.step_size.owner.op, LineSearchOp)
    assert result.step_size.owner is result.evaluations.owner


def test_a_loss_that_draws_random_numbers_is_refused():
    """Every trial would draw again, so the search would compare values of different functions."""
    generator = pytensor.shared(np.random.default_rng(0), name="noise_rng")
    _, noise = pt.random.normal(rng=generator, return_next_rng=True)
    loss = pt.sum(X**2) + noise

    with pytest.raises(ValueError, match="draws from noise_rng"):
        search_along(loss, [X], [2 * X], [D], zoom_line_search())


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"max_steps": 0}, "max_steps must be at least 1"),
        ({"slope_rtol": 0.9, "curv_rtol": 0.5}, "0 < slope_rtol < curv_rtol < 1"),
        ({"curv_rtol": 1.0}, "0 < slope_rtol < curv_rtol < 1"),
        ({"increase_factor": 1.0}, "increase_factor must exceed 1"),
    ],
    ids=["no_steps", "conditions_swapped", "curvature_unbounded", "no_growth"],
)
def test_hyperparameters_that_cannot_work_are_refused(arguments, message):
    with pytest.raises(ValueError, match=message):
        zoom_line_search(**arguments)


def test_a_single_precision_search_stays_in_single_precision():
    """Hyperparameters and safeguards float32 cannot hold exactly would otherwise promote the search's
    arithmetic to float64, deciding its conditions at a precision the float32 backends do not share,
    and the float64 trial step that results cannot be fed to the float32 trial at all."""
    x = pt.vector("x", dtype="float32", shape=(2,))
    d = pt.vector("d", dtype="float32", shape=(2,))
    loss = OBJECTIVES["rosenbrock"](x)
    search = zoom_line_search(max_learning_rate=100.1, increase_factor=1.7, tol=1e-7)
    result = search_along(loss, [x], [pt.grad(loss, x)], [d], search)
    x0 = ROSENBROCK_START.astype("float32")
    direction = np.full(2, 1e-3, dtype="float32")

    step = find_piece(result.step_size.owner.op, STEP)
    assert not [variable for variable in step.fgraph.variables if variable.type.dtype == "float64"]

    step_size, failed, evaluations = pytensor.function([x, d], list(result[:3]))(x0, direction)

    # The same search in float64 grows the step until it brackets, then zooms, over six trials.
    np.testing.assert_allclose(step_size, 14.19857, rtol=1e-5)
    assert (bool(failed), int(evaluations)) == (False, 6)


def test_the_search_moves_shared_parameters_along_a_direction_of_them():
    """The shape an optimizer builds: shared parameters, and directions that are expressions of them."""
    point = pytensor.shared(ROSENBROCK_START.copy(), name="point")
    loss = OBJECTIVES["rosenbrock"](point)
    [gradient] = pt.grad(loss, [point])
    result = search_along(loss, [point], [gradient], [-gradient], zoom_line_search())

    step_size, failed, evaluations = pytensor.function([], list(result[:3]))()

    expected = PINNED["zooms_into_a_bracket"][1]
    np.testing.assert_allclose(step_size, expected[0], rtol=1e-9)
    assert (bool(failed), int(evaluations)) == expected[1:]


def test_a_parameter_the_loss_does_not_read_is_refused():
    unused = pt.vector("unused", dtype="float64")
    loss = OBJECTIVES["quadratic"](X)

    with pytest.raises(ValueError, match="does not depend on unused"):
        search_along(loss, [X, unused], [X - 3.0, unused], [D, unused], zoom_line_search())
