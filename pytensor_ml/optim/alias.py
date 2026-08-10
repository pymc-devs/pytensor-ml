from collections.abc import Callable, Sequence

from pytensor_ml.optim.base import (
    LearningRate,
    LossOrGradients,
    Parameter,
    Rate,
    UpdateRule,
    Updates,
    counter,
    reuses_state,
)
from pytensor_ml.optim.rules import (
    _require_numeric_learning_rate,
    adadelta_updates,
    adagrad_updates,
    adam_updates,
    adamax_updates,
    adamw_updates,
    nadam_updates,
    rmsprop_updates,
    rprop_updates,
    sgd_updates,
)
from pytensor_ml.optim.transform import scale, trace


def _at_learning_rate(
    learning_rate: LearningRate,
    name: str,
    build_updates: Callable[[Rate], Updates],
) -> Updates:
    """Build updates at ``learning_rate``, reading a schedule off the clock the rule counts its own steps on.
    Both reach that clock through :func:`counter` under ``"{name}/step_count"``, so a rule that keeps a step
    count hands the schedule the same variable instead of a second one measuring the same time."""
    if not callable(learning_rate):
        return build_updates(learning_rate)

    return build_updates(learning_rate(counter(f"{name}/step_count")))


def sgd(
    learning_rate: LearningRate = 0.01, momentum: float = 0.0, nesterov: bool = False
) -> UpdateRule:
    """
    Stochastic gradient descent, optionally with momentum.

    Parameters
    ----------
    learning_rate : float, shared tensor variable, symbolic scalar, or Schedule
        Step size. A float is baked into the graph; a scalar shared variable can be steered from Python with
        ``set_value``; any scalar graph is used as the rate directly, as in
        ``cosine_schedule(3e-4, 10_000)(step_counter())``; an unapplied schedule is read off the clock the
        rule counts its own steps on. Default 0.01.
    momentum : float
        Momentum coefficient. A value of 0 (the default) gives plain SGD.
    nesterov : bool
        Use Nesterov momentum. Ignored when ``momentum`` is 0. Default False.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        def build_updates(rate: Rate) -> Updates:
            if not momentum:
                return sgd_updates(loss_or_gradients, parameters, learning_rate=rate)
            updates = sgd_updates(loss_or_gradients, parameters, learning_rate=1.0)
            updates = trace(momentum, nesterov)(updates, parameters)
            return scale(rate)(updates, parameters)

        return _at_learning_rate(learning_rate, "sgd", build_updates)

    return rule


def adam(
    learning_rate: LearningRate = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    amsgrad: bool = False,
) -> UpdateRule:
    """
    Adam optimizer. See :func:`~pytensor_ml.optim.rules.adam_updates` for the update rule.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule; see
    :func:`sgd`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return _at_learning_rate(
            learning_rate,
            "adam",
            lambda rate: adam_updates(
                loss_or_gradients,
                parameters,
                learning_rate=rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                amsgrad=amsgrad,
            ),
        )

    return rule


def adamw(
    learning_rate: LearningRate = 1e-3,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    amsgrad: bool = False,
    mask: Callable[[Parameter], bool] | None = None,
) -> UpdateRule:
    """
    AdamW optimizer (Adam with decoupled weight decay). See
    :func:`~pytensor_ml.optim.rules.adamw_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule; see
    :func:`sgd`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return _at_learning_rate(
            learning_rate,
            "adamw",
            lambda rate: adamw_updates(
                loss_or_gradients,
                parameters,
                learning_rate=rate,
                weight_decay=weight_decay,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                amsgrad=amsgrad,
                mask=mask,
            ),
        )

    return rule


def nadam(
    learning_rate: LearningRate = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> UpdateRule:
    """
    Nadam optimizer (Adam with Nesterov momentum). See
    :func:`~pytensor_ml.optim.rules.nadam_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule; see
    :func:`sgd`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return _at_learning_rate(
            learning_rate,
            "nadam",
            lambda rate: nadam_updates(
                loss_or_gradients,
                parameters,
                learning_rate=rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
            ),
        )

    return rule


def adamax(
    learning_rate: LearningRate = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> UpdateRule:
    """
    AdaMax optimizer (Adam with an infinity-norm denominator). See
    :func:`~pytensor_ml.optim.rules.adamax_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule; see
    :func:`sgd`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return _at_learning_rate(
            learning_rate,
            "adamax",
            lambda rate: adamax_updates(
                loss_or_gradients,
                parameters,
                learning_rate=rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
            ),
        )

    return rule


def rprop(
    learning_rate: float = 1e-2,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    step_min: float = 1e-6,
    step_max: float = 50.0,
) -> UpdateRule:
    """
    Rprop optimizer (resilient backpropagation). See :func:`~pytensor_ml.optim.rules.rprop_updates`.

    Unlike the other rules, ``learning_rate`` must be a plain number: it initializes the per-parameter
    step sizes Rprop then adapts, so it never enters the graph and cannot be scheduled or steered.
    """
    _require_numeric_learning_rate(learning_rate)

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return rprop_updates(
            loss_or_gradients,
            parameters,
            learning_rate=learning_rate,
            eta_minus=eta_minus,
            eta_plus=eta_plus,
            step_min=step_min,
            step_max=step_max,
        )

    return rule


def rmsprop(
    learning_rate: LearningRate = 1e-2,
    rho: float = 0.9,
    momentum: float = 0.0,
    epsilon: float = 1e-8,
    centered: bool = False,
) -> UpdateRule:
    """
    RMSProp optimizer. See :func:`~pytensor_ml.optim.rules.rmsprop_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule; see
    :func:`sgd`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return _at_learning_rate(
            learning_rate,
            "rmsprop",
            lambda rate: rmsprop_updates(
                loss_or_gradients,
                parameters,
                learning_rate=rate,
                rho=rho,
                momentum=momentum,
                epsilon=epsilon,
                centered=centered,
            ),
        )

    return rule


def adagrad(learning_rate: LearningRate = 0.01, epsilon: float = 1e-8) -> UpdateRule:
    """
    AdaGrad optimizer. See :func:`~pytensor_ml.optim.rules.adagrad_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule; see
    :func:`sgd`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return _at_learning_rate(
            learning_rate,
            "adagrad",
            lambda rate: adagrad_updates(
                loss_or_gradients, parameters, learning_rate=rate, epsilon=epsilon
            ),
        )

    return rule


def adadelta(
    learning_rate: LearningRate = 1.0, rho: float = 0.9, epsilon: float = 1e-8
) -> UpdateRule:
    """
    AdaDelta optimizer. See :func:`~pytensor_ml.optim.rules.adadelta_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule; see
    :func:`sgd`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return _at_learning_rate(
            learning_rate,
            "adadelta",
            lambda rate: adadelta_updates(
                loss_or_gradients, parameters, learning_rate=rate, rho=rho, epsilon=epsilon
            ),
        )

    return rule
