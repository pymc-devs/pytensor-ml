from collections.abc import Callable, Sequence

import pytensor.tensor as pt

from pytensor_ml.optim.base import (
    Parameter,
    Schedule,
    Transform,
    Updates,
    counter,
    scalar_state,
    state_for,
)


def trace(decay: float = 0.9, nesterov: bool = False) -> Transform:
    r"""
    Accumulate steps into a velocity buffer (classical or Nesterov momentum).

    Operating in step space (:math:`s = \text{updates}[p] - p`), the velocity is
    :math:`v \leftarrow \rho v + s`, and the new step is :math:`v` (classical) or :math:`s + \rho v`
    (Nesterov lookahead).

    Parameters
    ----------
    decay : float
        Momentum coefficient :math:`\rho`. Default 0.9.
    nesterov : bool
        Apply the Nesterov lookahead correction. Default False.

    Returns
    -------
    Transform
        A transform that folds momentum into the updates dict.
    """

    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        next_updates = dict(updates)
        for parameter in parameters:
            step = updates[parameter] - parameter
            velocity = state_for(parameter, "trace/velocity")
            new_velocity = decay * velocity + step
            next_updates[velocity] = new_velocity
            next_updates[parameter] = parameter + (
                step + decay * new_velocity if nesterov else new_velocity
            )
        return next_updates

    return transform


def scale(factor: float) -> Transform:
    """
    Scale each step by a constant factor.

    Typically the terminal transform in a chain, used to apply the learning rate after a unit-rate base rule.

    Parameters
    ----------
    factor : float
        Multiplier applied to every step.

    Returns
    -------
    Transform
        A transform that rescales the updates dict.
    """

    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        next_updates = dict(updates)
        for parameter in parameters:
            next_updates[parameter] = parameter + factor * (updates[parameter] - parameter)
        return next_updates

    return transform


def scale_by_schedule(schedule_fn: Schedule, learning_rate: Parameter | None = None) -> Transform:
    """
    Scale each step by a learning rate produced from an owned step counter.

    The counter starts at zero and increments by one each call. ``schedule_fn`` receives it symbolically and
    must return a scalar learning rate, so the schedule lives on the graph and stays synchronized with the
    optimizer's own time step.

    The rate *this transform* applies is also written to a scalar shared variable, so it can be checkpointed
    alongside the counter and read from Python after a step. Pass ``learning_rate`` to supply that variable
    and hold a reference to it; otherwise one is allocated as ``"schedule/learning_rate"``. Two scheduled
    scalings in one chain each publish their own factor and share one step counter, so the rate reaching the
    parameters is their product. The graph writes the variable on every step and never reads it, so
    ``set_value`` on it does not steer a schedule that is a function of the step count.

    Parameters
    ----------
    schedule_fn : callable
        Maps the symbolic step counter (a TensorVariable) to a scalar learning rate.
    learning_rate : shared tensor variable, optional
        Scalar variable to publish the applied rate to, cast to its dtype. Allocated internally when
        omitted.

    Returns
    -------
    Transform
        A transform that rescales the updates dict by the scheduled rate, publishes it, and advances the
        counter.
    """

    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        step_count = counter("schedule/step_count")
        published_rate = (
            scalar_state("schedule/learning_rate") if learning_rate is None else learning_rate
        )
        if published_rate in updates:
            raise ValueError(
                f"Two scheduled scalings in one chain would share the state {published_rate.name!r}, "
                "leaving it holding only the second rate while both are applied. Multiply the schedules "
                "into a single schedule function, or pass distinct `learning_rate` variables."
            )

        rate = pt.cast(schedule_fn(step_count), published_rate.type.dtype)

        next_updates = dict(updates)
        for parameter in parameters:
            next_updates[parameter] = parameter + rate * (updates[parameter] - parameter)
        next_updates[published_rate] = rate
        next_updates[step_count] = step_count + 1
        return next_updates

    return transform


def add_weight_decay(
    weight_decay: float = 0.01,
    mask: Callable[[Parameter], bool] | None = None,
) -> Transform:
    r"""
    Subtract a decoupled weight-decay term :math:`\lambda p` from each step.

    Place this before a terminal :func:`scale` so the final update is
    :math:`p \leftarrow p + \eta (s - \lambda p)`, giving decay that scales with the learning rate but is
    decoupled from any adaptive step rescaling earlier in the chain.

    Parameters
    ----------
    weight_decay : float
        Decay coefficient :math:`\lambda`. Default 0.01.
    mask : callable, optional
        Predicate ``(parameter) -> bool`` selecting which parameters receive decay. Decay is applied to every
        parameter when omitted.

    Returns
    -------
    Transform
        A transform that folds weight decay into the updates dict.
    """

    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        next_updates = dict(updates)
        for parameter in parameters:
            step = updates[parameter] - parameter
            decayed_step = (
                step - weight_decay * parameter if (mask is None or mask(parameter)) else step
            )
            next_updates[parameter] = parameter + decayed_step
        return next_updates

    return transform
