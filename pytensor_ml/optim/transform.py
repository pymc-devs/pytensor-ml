from collections.abc import Callable, Sequence

from pytensor_ml.optim.base import (
    Parameter,
    Rate,
    Schedule,
    Transform,
    Updates,
    counter,
    reuses_state,
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


def scale(factor: Rate) -> Transform:
    """
    Scale each step by a constant factor.

    Typically the terminal transform in a chain, used to apply the learning rate after a unit-rate base rule.

    Parameters
    ----------
    factor : float or shared tensor variable
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


def scale_by_schedule(schedule: Schedule, *, namespace: str = "scale_by_schedule") -> Transform:
    """
    Scale each step by a schedule read off a training clock of its own.

    The terminal transform for scheduling a rate *after* a rule rather than inside it. The two are different
    graphs whenever anything sits between them: a clip placed before this one bounds a step at unit rate, so
    its bound stays in gradient units instead of moving with the rate, while a rule given the schedule as its
    ``learning_rate`` has already applied the rate by the time the clip sees the step.

    .. code-block:: python

        schedule = cosine_schedule(3e-4, 10_000)
        rule = chain(adam(1.0), clip_by_global_norm(1.0), scale_by_schedule(schedule))

    The clock advances on its own. It is a :class:`~pytensor_ml.params.StepCounter`, so
    :func:`~pytensor_ml.pytensorf.collect_clock_updates` finds it in the graph and writes its advance into
    the compiled step, and every other clock the step reads counts the same steps as this one.

    Parameters
    ----------
    schedule : Schedule
        A ``(step_count) -> rate`` callable such as :func:`~pytensor_ml.optim.schedules.cosine_schedule`,
        applied to the clock and multiplied into every step.
    namespace : str
        Prefix for the clock this transform allocates, as ``"{namespace}/step_count"``. Give two scheduled
        scales in one graph different namespaces so their clocks stay distinct at the serialization
        boundary. Default ``"scale_by_schedule"``.

    Returns
    -------
    Transform
        A transform that rescales the updates dict by the rate its clock currently reads.
    """

    @reuses_state
    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        return scale(schedule(counter(f"{namespace}/step_count")))(updates, parameters)

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
