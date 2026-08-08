import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import Schedule


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
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}.")
    if min_learning_rate > learning_rate:
        raise ValueError(
            f"min_learning_rate must not exceed learning_rate, got {min_learning_rate} > "
            f"{learning_rate}. Schedules in this module decay, so the floor is the smaller of the two."
        )

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(min_learning_rate, dtype=floatX)
        step_limit = np.asarray(total_steps, dtype=floatX)

        progress = pt.minimum(step_count.astype(floatX), step_limit) / step_limit
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
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}.")
    if transition_begin < 0:
        raise ValueError(f"transition_begin must not be negative, got {transition_begin}.")
    if min_learning_rate > learning_rate:
        raise ValueError(
            f"min_learning_rate must not exceed learning_rate, got {min_learning_rate} > "
            f"{learning_rate}. Schedules in this module decay, so the floor is the smaller of the two."
        )

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(min_learning_rate, dtype=floatX)
        step_limit = np.asarray(total_steps, dtype=floatX)
        begin = np.asarray(transition_begin, dtype=floatX)

        progress = pt.clip(step_count.astype(floatX) - begin, 0.0, step_limit) / step_limit
        return initial_rate + (final_rate - initial_rate) * progress

    return schedule
