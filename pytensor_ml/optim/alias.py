from collections.abc import Callable, Sequence

from pytensor_ml.optim.base import (
    LossOrGradients,
    Parameter,
    UpdateRule,
    Updates,
    reuses_state,
)
from pytensor_ml.optim.rules import (
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


def sgd(learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = False) -> UpdateRule:
    """
    Stochastic gradient descent, optionally with momentum.

    Parameters
    ----------
    learning_rate : float
        Step size. Default 0.01.
    momentum : float
        Momentum coefficient. A value of 0 (the default) gives plain SGD.
    nesterov : bool
        Use Nesterov momentum. Ignored when ``momentum`` is 0. Default False.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        if not momentum:
            return sgd_updates(loss_or_gradients, parameters, learning_rate=learning_rate)
        updates = sgd_updates(loss_or_gradients, parameters, learning_rate=1.0)
        updates = trace(momentum, nesterov)(updates, parameters)
        return scale(learning_rate)(updates, parameters)

    return rule


def adam(
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    amsgrad: bool = False,
) -> UpdateRule:
    """
    Adam optimizer. See :func:`~pytensor_ml.optim.rules.adam_updates` for the update rule.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return adam_updates(
            loss_or_gradients,
            parameters,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            amsgrad=amsgrad,
        )

    return rule


def adamw(
    learning_rate: float = 1e-3,
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
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return adamw_updates(
            loss_or_gradients,
            parameters,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            mask=mask,
        )

    return rule


def nadam(
    learning_rate: float = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> UpdateRule:
    """
    Nadam optimizer (Adam with Nesterov momentum). See
    :func:`~pytensor_ml.optim.rules.nadam_updates`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return nadam_updates(
            loss_or_gradients,
            parameters,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )

    return rule


def adamax(
    learning_rate: float = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> UpdateRule:
    """
    AdaMax optimizer (Adam with an infinity-norm denominator). See
    :func:`~pytensor_ml.optim.rules.adamax_updates`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return adamax_updates(
            loss_or_gradients,
            parameters,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
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
    """

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
    learning_rate: float = 1e-2,
    rho: float = 0.9,
    momentum: float = 0.0,
    epsilon: float = 1e-8,
    centered: bool = False,
) -> UpdateRule:
    """
    RMSProp optimizer. See :func:`~pytensor_ml.optim.rules.rmsprop_updates`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return rmsprop_updates(
            loss_or_gradients,
            parameters,
            learning_rate=learning_rate,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered,
        )

    return rule


def adagrad(learning_rate: float = 0.01, epsilon: float = 1e-8) -> UpdateRule:
    """
    AdaGrad optimizer. See :func:`~pytensor_ml.optim.rules.adagrad_updates`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return adagrad_updates(
            loss_or_gradients, parameters, learning_rate=learning_rate, epsilon=epsilon
        )

    return rule


def adadelta(learning_rate: float = 1.0, rho: float = 0.9, epsilon: float = 1e-8) -> UpdateRule:
    """
    AdaDelta optimizer. See :func:`~pytensor_ml.optim.rules.adadelta_updates`.
    """

    @reuses_state
    def rule(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        return adadelta_updates(
            loss_or_gradients, parameters, learning_rate=learning_rate, rho=rho, epsilon=epsilon
        )

    return rule
