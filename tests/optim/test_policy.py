from itertools import pairwise

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor import config

from pytensor_ml.optim import adam, compile_train, reduce_on_plateau, scalar_state, sgd
from pytensor_ml.params import trainable

RTOL = 1e-6


def plateaued_problem():
    """A parameter parked at the optimum, so the loss is zero and can never improve again. Every step after
    the first is a bad one, which is what makes the policy's counting observable."""
    p = trainable(np.zeros(1, dtype=config.floatX), name="w")
    return p, 0.5 * (p**2).sum()


def run(rule, loss, scale, n_steps):
    """Return the scale after each of ``n_steps`` steps."""
    step = compile_train(loss, rule, inputs=[])
    scales = []
    for _ in range(n_steps):
        step()
        scales.append(float(scale.get_value()))
    return scales


def test_cuts_after_exactly_patience_steps_without_improvement():
    p, loss = plateaued_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(adam(learning_rate=scale * 0.05), scale, factor=0.5, patience=3)

    # The first step improves on the initial infinite best, so counting starts after it, and each cut needs
    # a fresh `patience` bad steps rather than following on from the last.
    assert run(rule, loss, scale, 8) == [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25]


def test_cools_down_before_counting_again():
    """Without a cooldown the count resets on the cut and immediately starts toward the next, which is how
    a noisy loss cuts dozens of times in a short run."""
    p, loss = plateaued_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(
        adam(learning_rate=scale * 0.05), scale, factor=0.5, patience=3, cooldown=2
    )

    scales = run(rule, loss, scale, 11)

    # Cuts land patience + cooldown apart rather than every patience steps.
    assert scales == [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25]


def test_never_goes_below_the_floor():
    """A loss that never improves cuts forever; unfloored the scale underflows and training stops without
    any error to say so."""
    p, loss = plateaued_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(
        adam(learning_rate=scale * 0.05), scale, factor=0.5, patience=1, min_scale=0.2
    )

    assert min(run(rule, loss, scale, 20)) == pytest.approx(0.2, rel=RTOL)


def test_an_improvement_resets_the_count():
    """The parameter starts away from the optimum, so adam improves the loss every step and nothing ever
    counts as bad. A policy that counted regardless would cut a run that is going fine."""
    p = trainable(np.array([2.0], dtype=config.floatX), name="w")
    loss = 0.5 * (p**2).sum()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(adam(learning_rate=scale * 0.05), scale, factor=0.5, patience=3)

    assert set(run(rule, loss, scale, 20)) == {1.0}


def test_an_improvement_too_small_to_matter_does_not_reset_the_count():
    """`rtol` is what makes this usable on a per-batch loss: without it any improvement at all, however
    tiny, resets the count, so a loss drifting down by a rounding error never plateaus."""
    p, loss = plateaued_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    # A relative tolerance of 1 demands the loss halve to count, which zero never does after the first step.
    rule = reduce_on_plateau(
        adam(learning_rate=scale * 0.05), scale, factor=0.5, patience=2, rtol=1.0
    )

    assert run(rule, loss, scale, 3)[-1] == pytest.approx(0.5, rel=RTOL)


def test_a_noisy_loss_cuts_itself_into_the_ground_without_a_floor_and_a_cooldown():
    """The trajectory that made both parameters non-optional. A per-batch loss stops improving constantly,
    so with neither guard the policy cuts on almost every patience window and the scale underflows -- the run
    stops training with nothing raised to say so."""

    def noisy_run(cooldown, min_scale, n_steps=120):
        p = trainable(np.zeros(1, dtype=config.floatX), name="w")
        X = pt.tensor("X", shape=(None,))
        loss = ((p - X) ** 2).mean()
        scale = scalar_state("plateau/scale", fill_value=1.0)
        rule = reduce_on_plateau(
            adam(learning_rate=scale * 0.05),
            scale,
            factor=0.5,
            patience=3,
            cooldown=cooldown,
            min_scale=min_scale,
        )
        step = compile_train(loss, rule, inputs=[X])
        rng = np.random.default_rng(0)
        scales = []
        for _ in range(n_steps):
            step(rng.normal(size=8).astype(config.floatX))
            scales.append(float(scale.get_value()))
        cuts = sum(1 for before, after in pairwise(scales) if after < before)
        return cuts, min(scales)

    unguarded_cuts, unguarded_floor = noisy_run(cooldown=0, min_scale=0.0)
    assert unguarded_cuts > 30
    assert unguarded_floor < 1e-9  # a rate this small is not training anything

    guarded_cuts, guarded_floor = noisy_run(cooldown=10, min_scale=0.01)
    assert guarded_cuts < unguarded_cuts / 3
    assert guarded_floor == pytest.approx(0.01, rel=RTOL)


def test_rejects_precomputed_gradients():
    """The policy decides from the loss, and `loss_or_gradients` is a union, so a caller passing gradients
    would otherwise hand it a list and get a confusing failure deep in the comparison."""
    p, loss = plateaued_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(sgd(learning_rate=scale * 0.05), scale)

    with pytest.raises(ValueError, match="needs the loss graph"):
        rule([pytensor.gradient.grad(loss, p)], [p])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"factor": 1.0}, "factor must lie in"),
        ({"factor": 0.0}, "factor must lie in"),
        ({"rtol": -0.1}, "must be non-negative"),
        ({"rtol": 2.0}, "at most 1.0"),
        ({"patience": 0}, "at least 1"),
        ({"cooldown": -1}, "non-negative"),
    ],
    ids=[
        "factor_one",
        "factor_zero",
        "negative_rtol",
        "rtol_above_one",
        "no_patience",
        "negative_cooldown",
    ],
)
def test_rejects_settings_that_cannot_work(kwargs, message):
    scale = scalar_state("plateau/scale", fill_value=1.0)

    with pytest.raises(ValueError, match=message):
        reduce_on_plateau(sgd(learning_rate=scale), scale, **kwargs)
