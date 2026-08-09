import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import Schedule


def _validate_rates(learning_rate: float, min_learning_rate: float) -> None:
    """Raise ``ValueError`` unless the two rates describe a decay."""
    if min_learning_rate > learning_rate:
        raise ValueError(
            f"min_learning_rate must not exceed learning_rate, got {min_learning_rate} > "
            f"{learning_rate}. Schedules in this module decay, so the floor is the smaller of the two."
        )


def _validate_horizon(
    learning_rate: float,
    total_steps: int,
    min_learning_rate: float,
    transition_begin: int = 0,
) -> None:
    """Raise ``ValueError`` unless the rates and the horizon describe a decay."""
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}.")
    if transition_begin < 0:
        raise ValueError(f"transition_begin must not be negative, got {transition_begin}.")
    _validate_rates(learning_rate, min_learning_rate)


def _clamped_progress(
    step_count: TensorVariable, total_steps: int, transition_begin: int = 0
) -> TensorVariable:
    """Return the fraction of the decay completed at ``step_count``, held at 0 before it starts and 1
    after it ends, so every schedule flattens outside its horizon rather than running past it."""
    floatX = config.floatX
    step_limit = np.asarray(total_steps, dtype=floatX)
    begin = np.asarray(transition_begin, dtype=floatX)
    return pt.clip(step_count.astype(floatX) - begin, 0.0, step_limit) / step_limit


def cosine_annealing(
    learning_rate: float,
    total_steps: int,
    min_learning_rate: float = 0.0,
) -> Schedule:
    r"""
    Anneal the learning rate from ``learning_rate`` to ``min_learning_rate`` along a half cosine.

    At step :math:`t` of :math:`T` total steps,

    .. math::

        \eta_t = \eta_{\min} + \frac{1}{2} (\eta_0 - \eta_{\min})
                 \left(1 + \cos\left(\pi \frac{\min(t, T)}{T}\right)\right)

    so the rate leaves :math:`\eta_0` slowly, falls fastest at the midpoint, and flattens into
    :math:`\eta_{\min}`, where it stays for any step past :math:`T`.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at step zero.
    total_steps : int
        Number of steps :math:`T` over which the rate reaches its floor. Must be at least one.
    min_learning_rate : float, optional
        Floor :math:`\eta_{\min}` reached at step ``total_steps``. Default 0.0.

    Returns
    -------
    Schedule
        A callable mapping the symbolic step count to a scalar learning rate, for
        :func:`~pytensor_ml.optim.transform.scale_by_schedule`.

    Examples
    --------
    Hand the schedule to an optimizer in place of a rate:

    .. code-block:: python

        from pytensor_ml.optim import adam, cosine_annealing

        rule = adam(learning_rate=cosine_annealing(3e-4, 10_000))
    """
    _validate_horizon(learning_rate, total_steps, min_learning_rate)

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(min_learning_rate, dtype=floatX)

        progress = _clamped_progress(step_count, total_steps)
        cosine_factor = 0.5 * (1.0 + pt.cos(np.pi * progress))
        return final_rate + (initial_rate - final_rate) * cosine_factor

    return schedule


def linear_decay(
    learning_rate: float,
    total_steps: int,
    min_learning_rate: float = 0.0,
    transition_begin: int = 0,
) -> Schedule:
    r"""
    Decay the learning rate from ``learning_rate`` to ``min_learning_rate`` at a constant rate.

    At step :math:`t`, decaying over :math:`T` steps starting from step :math:`B`,

    .. math::

        \eta_t = \eta_0 + (\eta_{\min} - \eta_0)
                 \frac{\min(\max(t - B, 0),\; T)}{T}

    so the rate holds at :math:`\eta_0` through step :math:`B`, falls by the same amount every step for
    :math:`T` steps, then holds at :math:`\eta_{\min}` from step :math:`B + T` on.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at every step up to ``transition_begin``.
    total_steps : int
        Number of steps :math:`T` the decay itself spans. Must be at least one.
    min_learning_rate : float, optional
        Floor :math:`\eta_{\min}` reached at step ``transition_begin + total_steps``. Default 0.0.
    transition_begin : int, optional
        Number of steps :math:`B` to hold the initial rate before decaying. Must not be negative.
        Default 0.

    Returns
    -------
    Schedule
        A callable mapping the symbolic step count to a scalar learning rate, for
        :func:`~pytensor_ml.optim.transform.scale_by_schedule`.
    """
    _validate_horizon(learning_rate, total_steps, min_learning_rate, transition_begin)

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(min_learning_rate, dtype=floatX)

        progress = _clamped_progress(step_count, total_steps, transition_begin)
        return initial_rate + (final_rate - initial_rate) * progress

    return schedule


def exponential_decay(
    learning_rate: float,
    total_steps: int,
    min_learning_rate: float,
    transition_begin: int = 0,
) -> Schedule:
    r"""
    Decay the learning rate from ``learning_rate`` to ``min_learning_rate`` by a constant factor per step.

    At step :math:`t`, decaying over :math:`T` steps starting from step :math:`B`, and writing
    :math:`p = \min(\max(t - B, 0),\; T) / T`,

    .. math::

        \eta_t = \eta_0 \left(\frac{\eta_{\min}}{\eta_0}\right)^{p}

    so the rate is multiplied by the same factor every step — falling fastest in absolute terms early on —
    and holds at :math:`\eta_{\min}` from step :math:`B + T` on.

    ``min_learning_rate`` is required and must be positive: geometric decay approaches zero without
    ever reaching it, so there is no floor of zero to decay to.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at every step up to ``transition_begin``. Must be positive.
    total_steps : int
        Number of steps :math:`T` the decay itself spans. Must be at least one.
    min_learning_rate : float
        Floor :math:`\eta_{\min}` reached at step ``transition_begin + total_steps``. Must be positive.
    transition_begin : int, optional
        Number of steps :math:`B` to hold the initial rate before decaying. Must not be negative.
        Default 0.

    Returns
    -------
    Schedule
        A callable mapping the symbolic step count to a scalar learning rate, for
        :func:`~pytensor_ml.optim.transform.scale_by_schedule`.
    """
    _validate_horizon(learning_rate, total_steps, min_learning_rate, transition_begin)
    if min_learning_rate <= 0.0 or learning_rate <= 0.0:
        raise ValueError(
            f"exponential_decay needs positive rates, got learning_rate={learning_rate} and "
            f"min_learning_rate={min_learning_rate}. Geometric decay never reaches zero, so pick a "
            f"floor you are willing to train at."
        )

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        # Taken in float64 on the host, so a small ratio keeps its precision under floatX=float32.
        log_decay = np.asarray(np.log(min_learning_rate / learning_rate), dtype=floatX)

        progress = _clamped_progress(step_count, total_steps, transition_begin)
        return initial_rate * pt.exp(log_decay * progress)

    return schedule


def polynomial_decay(
    learning_rate: float,
    total_steps: int,
    min_learning_rate: float = 0.0,
    transition_begin: int = 0,
    power: float = 1.0,
) -> Schedule:
    r"""
    Decay the learning rate from ``learning_rate`` to ``min_learning_rate`` along a power curve.

    At step :math:`t`, decaying over :math:`T` steps starting from step :math:`B`, and writing
    :math:`p = \min(\max(t - B, 0),\; T) / T`,

    .. math::

        \eta_t = \eta_{\min} + (\eta_0 - \eta_{\min}) (1 - p)^{\gamma}

    so :math:`\gamma > 1` drops the rate quickly and then flattens, :math:`\gamma < 1` holds it high and
    then drops, and :math:`\gamma = 1` is a straight line. The rate holds at :math:`\eta_{\min}` from step
    :math:`B + T` on.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at every step up to ``transition_begin``.
    total_steps : int
        Number of steps :math:`T` the decay itself spans. Must be at least one.
    min_learning_rate : float, optional
        Floor :math:`\eta_{\min}` reached at step ``transition_begin + total_steps``. Default 0.0.
    transition_begin : int, optional
        Number of steps :math:`B` to hold the initial rate before decaying. Must not be negative.
        Default 0.
    power : float, optional
        Exponent :math:`\gamma` applied to the remaining fraction of the horizon. Must be positive.
        Default 1.0, which decays linearly.

    Returns
    -------
    Schedule
        A callable mapping the symbolic step count to a scalar learning rate, for
        :func:`~pytensor_ml.optim.transform.scale_by_schedule`.
    """
    _validate_horizon(learning_rate, total_steps, min_learning_rate, transition_begin)
    if power <= 0.0:
        raise ValueError(f"power must be positive, got {power}.")

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(min_learning_rate, dtype=floatX)
        exponent = np.asarray(power, dtype=floatX)

        remaining = 1.0 - _clamped_progress(step_count, total_steps, transition_begin)
        return final_rate + (initial_rate - final_rate) * remaining**exponent

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
        A callable mapping the symbolic step count to a scalar learning rate, for
        :func:`~pytensor_ml.optim.transform.scale_by_schedule`.
    """
    if decay_every < 1:
        raise ValueError(f"decay_every must be at least 1, got {decay_every}.")
    if not 0.0 < decay_factor <= 1.0:
        raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor}.")
    if transition_begin < 0:
        raise ValueError(f"transition_begin must not be negative, got {transition_begin}.")
    _validate_rates(learning_rate, min_learning_rate)

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        floor_rate = np.asarray(min_learning_rate, dtype=floatX)
        factor = np.asarray(decay_factor, dtype=floatX)

        drops = pt.maximum(step_count - transition_begin, 0) // decay_every
        return pt.maximum(initial_rate * factor ** drops.astype(floatX), floor_rate)

    return schedule
