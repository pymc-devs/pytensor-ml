from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import Schedule


def _validate_horizon(total_steps: int, transition_begin: int = 0) -> None:
    """Raise ``ValueError`` unless the horizon is usable."""
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}.")
    if transition_begin < 0:
        raise ValueError(f"transition_begin must not be negative, got {transition_begin}.")


def _clamped_progress(
    step_count: TensorVariable, total_steps: int, transition_begin: int = 0
) -> TensorVariable:
    """Return the fraction of the horizon completed at ``step_count``, held at 0 before it starts and 1
    after it ends, so every schedule flattens outside its horizon rather than running past it."""
    floatX = config.floatX
    step_limit = np.asarray(total_steps, dtype=floatX)
    begin = np.asarray(transition_begin, dtype=floatX)
    return pt.clip(step_count.astype(floatX) - begin, 0.0, step_limit) / step_limit


def cosine_schedule(
    learning_rate: float,
    total_steps: int,
    final_learning_rate: float = 0.0,
) -> Schedule:
    r"""
    Move the learning rate from ``learning_rate`` to ``final_learning_rate`` along a half cosine.

    At step :math:`t` of :math:`T` total steps,

    .. math::

        \eta_t = \eta_f + \frac{1}{2} (\eta_0 - \eta_f)
                 \left(1 + \cos\left(\pi \frac{\min(t, T)}{T}\right)\right)

    so the rate leaves :math:`\eta_0` slowly, moves fastest at the midpoint, and flattens into
    :math:`\eta_f`, where it stays for any step past :math:`T`. A ``final_learning_rate`` above
    ``learning_rate`` ramps up instead of decaying.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at step zero.
    total_steps : int
        Number of steps :math:`T` over which the rate reaches its endpoint. Must be at least one.
    final_learning_rate : float, optional
        Endpoint :math:`\eta_f` reached at step ``total_steps``, above or below ``learning_rate``.
        Default 0.0.

    Returns
    -------
    Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.

    Examples
    --------
    Hand the schedule to an optimizer in place of a rate:

    .. code-block:: python

        from pytensor_ml.optim import adam, cosine_schedule

        rule = adam(learning_rate=cosine_schedule(3e-4, 10_000))
    """
    _validate_horizon(total_steps)

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(final_learning_rate, dtype=floatX)

        progress = _clamped_progress(step_count, total_steps)
        # Weighting both endpoints keeps each one exact: at weight 0 or 1 the other term vanishes, where
        # `final + (initial - final) * factor` recovers `initial` by cancellation instead.
        weight = 0.5 * (1.0 - pt.cos(np.pi * progress))
        return initial_rate * (1.0 - weight) + final_rate * weight

    return schedule


def linear_schedule(
    learning_rate: float,
    total_steps: int,
    final_learning_rate: float = 0.0,
    transition_begin: int = 0,
) -> Schedule:
    r"""
    Move the learning rate from ``learning_rate`` to ``final_learning_rate`` at a constant rate.

    At step :math:`t`, decaying over :math:`T` steps starting from step :math:`B`,

    .. math::

        \eta_t = \eta_0 + (\eta_f - \eta_0)
                 \frac{\min(\max(t - B, 0),\; T)}{T}

    so the rate holds at :math:`\eta_0` through step :math:`B`, falls by the same amount every step for
    :math:`T` steps, then holds at :math:`\eta_f` from step :math:`B + T` on.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at every step up to ``transition_begin``.
    total_steps : int
        Number of steps :math:`T` the decay itself spans. Must be at least one.
    final_learning_rate : float, optional
        Endpoint :math:`\eta_f` reached at step ``transition_begin + total_steps``, above or below
        ``learning_rate``. Default 0.0.
    transition_begin : int, optional
        Number of steps :math:`B` to hold the initial rate before decaying. Must not be negative.
        Default 0.

    Returns
    -------
    Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.
    """
    _validate_horizon(total_steps, transition_begin)

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(final_learning_rate, dtype=floatX)

        progress = _clamped_progress(step_count, total_steps, transition_begin)
        return initial_rate * (1.0 - progress) + final_rate * progress

    return schedule


def exponential_schedule(
    learning_rate: float,
    total_steps: int,
    final_learning_rate: float,
    transition_begin: int = 0,
) -> Schedule:
    r"""
    Move the learning rate from ``learning_rate`` to ``final_learning_rate`` by a constant factor per step.

    At step :math:`t`, decaying over :math:`T` steps starting from step :math:`B`, and writing
    :math:`p = \min(\max(t - B, 0),\; T) / T`,

    .. math::

        \eta_t = \eta_0 \left(\frac{\eta_f}{\eta_0}\right)^{p}

    so the rate is multiplied by the same factor every step — falling fastest in absolute terms early on —
    and holds at :math:`\eta_f` from step :math:`B + T` on.

    ``final_learning_rate`` is required and must be positive: a geometric path approaches zero without
    ever reaching it, so there is no endpoint of zero to move to.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at every step up to ``transition_begin``. Must be positive.
    total_steps : int
        Number of steps :math:`T` the decay itself spans. Must be at least one.
    final_learning_rate : float
        Endpoint :math:`\eta_f` reached at step ``transition_begin + total_steps``. Must be positive,
        and may be above ``learning_rate`` to ramp up geometrically.
    transition_begin : int, optional
        Number of steps :math:`B` to hold the initial rate before decaying. Must not be negative.
        Default 0.

    Returns
    -------
    Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.
    """
    _validate_horizon(total_steps, transition_begin)
    if final_learning_rate <= 0.0 or learning_rate <= 0.0:
        raise ValueError(
            f"exponential_schedule needs positive rates, got learning_rate={learning_rate} and "
            f"final_learning_rate={final_learning_rate}. A geometric path never reaches zero, so pick an "
            f"endpoint you are willing to train at."
        )

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        # Taken in float64 on the host, so a small ratio keeps its precision under floatX=float32.
        log_decay = np.asarray(np.log(final_learning_rate / learning_rate), dtype=floatX)

        progress = _clamped_progress(step_count, total_steps, transition_begin)
        return initial_rate * pt.exp(log_decay * progress)

    return schedule


def polynomial_schedule(
    learning_rate: float,
    total_steps: int,
    final_learning_rate: float = 0.0,
    transition_begin: int = 0,
    power: float = 1.0,
) -> Schedule:
    r"""
    Move the learning rate from ``learning_rate`` to ``final_learning_rate`` along a power curve.

    At step :math:`t`, decaying over :math:`T` steps starting from step :math:`B`, and writing
    :math:`p = \min(\max(t - B, 0),\; T) / T`,

    .. math::

        \eta_t = \eta_f + (\eta_0 - \eta_f) (1 - p)^{\gamma}

    so :math:`\gamma > 1` moves the rate quickly and then flattens, :math:`\gamma < 1` holds it near
    :math:`\eta_0` and then moves, and :math:`\gamma = 1` is a straight line. The rate holds at
    :math:`\eta_f` from step :math:`B + T` on.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at every step up to ``transition_begin``.
    total_steps : int
        Number of steps :math:`T` the decay itself spans. Must be at least one.
    final_learning_rate : float, optional
        Endpoint :math:`\eta_f` reached at step ``transition_begin + total_steps``, above or below
        ``learning_rate``. Default 0.0.
    transition_begin : int, optional
        Number of steps :math:`B` to hold the initial rate before decaying. Must not be negative.
        Default 0.
    power : float, optional
        Exponent :math:`\gamma` applied to the remaining fraction of the horizon. Must be positive.
        Default 1.0, which decays linearly.

    Returns
    -------
    Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.
    """
    _validate_horizon(total_steps, transition_begin)
    if power <= 0.0:
        raise ValueError(f"power must be positive, got {power}.")

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(final_learning_rate, dtype=floatX)
        exponent = np.asarray(power, dtype=floatX)

        remaining = 1.0 - _clamped_progress(step_count, total_steps, transition_begin)
        weight = 1.0 - remaining**exponent
        return initial_rate * (1.0 - weight) + final_rate * weight

    return schedule


def step_decay(
    learning_rate: float,
    *,
    decay_every: int,
    decay_factor: float = 0.1,
    min_learning_rate: float = 0.0,
    transition_begin: int = 0,
) -> Schedule:
    r"""
    Multiply the learning rate by ``decay_factor`` every ``decay_every`` steps.

    At step :math:`t`, dropping by :math:`\gamma` every :math:`E` steps and starting from step :math:`B`,

    .. math::

        \eta_t = \max\left(\eta_0\, \gamma^{\lfloor \max(t - B,\, 0) / E \rfloor},\; \eta_{\min}\right)

    so the rate is a staircase rather than a curve, and it decays indefinitely rather than over a fixed
    horizon. Every argument after the rate is keyword-only, because with no horizon its positional slots
    would not line up with the other schedules'.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned until the first drop.
    decay_every : int
        Number of steps :math:`E` between drops. Must be at least one.
    decay_factor : float, optional
        Multiplier :math:`\gamma` applied at each drop. Must be in ``(0, 1]``. Default 0.1.
    min_learning_rate : float, optional
        Floor :math:`\eta_{\min}` the staircase never goes below. Default 0.0.
    transition_begin : int, optional
        Number of steps :math:`B` to hold the initial rate before the first drop can occur. Must not be
        negative. Default 0.

    Returns
    -------
    Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.
    """
    if decay_every < 1:
        raise ValueError(f"decay_every must be at least 1, got {decay_every}.")
    if not 0.0 < decay_factor <= 1.0:
        raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor}.")
    if transition_begin < 0:
        raise ValueError(f"transition_begin must not be negative, got {transition_begin}.")

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        floor_rate = np.asarray(min_learning_rate, dtype=floatX)
        factor = np.asarray(decay_factor, dtype=floatX)

        drops = pt.maximum(step_count - transition_begin, 0) // decay_every
        return pt.maximum(initial_rate * factor ** drops.astype(floatX), floor_rate)

    return schedule


def constant_schedule(learning_rate: float) -> Schedule:
    r"""
    Hold the learning rate at ``learning_rate`` forever.

    Useful as a segment of :func:`join_schedules`, where a constant stretch before or between decays is
    otherwise awkward to express.

    Parameters
    ----------
    learning_rate : float
        The rate :math:`\eta_0` returned at every step.

    Returns
    -------
    Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.
    """

    def schedule(step_count: TensorVariable) -> TensorVariable:
        # The rate has to stay a function of `step_count`: callers compile the schedule against the
        # counter, and a graph that ignores its input raises UnusedInputError.
        return pt.zeros_like(step_count, dtype=config.floatX) + np.asarray(
            learning_rate, dtype=config.floatX
        )

    return schedule


def join_schedules(schedules: Sequence[Schedule], boundaries: Sequence[int]) -> Schedule:
    r"""
    Run ``schedules`` one after another, switching at ``boundaries``.

    Each schedule after the first receives the step count *since its boundary*, so a decay placed second
    starts from its own step zero rather than partway down its curve. That is what makes warmup followed by
    decay a composition rather than a special case.

    Every segment is evaluated at every step and the result selected, so a later schedule is called with a
    negative count before its boundary. The schedules in this module clamp that to zero; a custom schedule
    must tolerate it.

    Parameters
    ----------
    schedules : sequence of Schedule
        The schedules to run in order. Must not be empty.
    boundaries : sequence of int
        Steps at which to hand over to the next schedule, one fewer than ``schedules``, strictly
        increasing and positive.

    Returns
    -------
    Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.
    """
    if not schedules:
        raise ValueError("join_schedules needs at least one schedule.")
    if len(boundaries) != len(schedules) - 1:
        raise ValueError(
            f"join_schedules needs one fewer boundary than schedules, got {len(boundaries)} "
            f"boundaries for {len(schedules)} schedules."
        )
    if any(boundary < 1 for boundary in boundaries):
        raise ValueError(f"boundaries must be positive, got {list(boundaries)}.")
    if any(later <= earlier for earlier, later in pairwise(boundaries)):
        raise ValueError(f"boundaries must be strictly increasing, got {list(boundaries)}.")

    def schedule(step_count: TensorVariable) -> TensorVariable:
        rate = schedules[0](step_count)
        for boundary, next_schedule in zip(boundaries, schedules[1:]):
            rate = pt.where(step_count < boundary, rate, next_schedule(step_count - boundary))
        return rate

    return schedule
