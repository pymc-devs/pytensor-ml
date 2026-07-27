from collections.abc import Callable

from pytensor_ml.optim.base import Parameter, UpdateRule
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

    Returns
    -------
    UpdateRule
        A configured optimizer ``(loss_or_gradients, parameters) -> Updates``.
    """

    def rule(loss_or_gradients, parameters):
        if not momentum:
            return sgd_updates(loss_or_gradients, parameters, learning_rate)
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

    Returns
    -------
    UpdateRule
        A configured optimizer ``(loss_or_gradients, parameters) -> Updates``.
    """

    def rule(loss_or_gradients, parameters):
        return adam_updates(
            loss_or_gradients, parameters, learning_rate, beta1, beta2, epsilon, amsgrad
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

    Returns
    -------
    UpdateRule
        A configured optimizer ``(loss_or_gradients, parameters) -> Updates``.
    """

    def rule(loss_or_gradients, parameters):
        return adamw_updates(
            loss_or_gradients,
            parameters,
            learning_rate,
            weight_decay,
            beta1,
            beta2,
            epsilon,
            amsgrad,
            mask,
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

    Returns
    -------
    UpdateRule
        A configured optimizer ``(loss_or_gradients, parameters) -> Updates``.
    """

    def rule(loss_or_gradients, parameters):
        return nadam_updates(loss_or_gradients, parameters, learning_rate, beta1, beta2, epsilon)

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

    Returns
    -------
    UpdateRule
        A configured optimizer ``(loss_or_gradients, parameters) -> Updates``.
    """

    def rule(loss_or_gradients, parameters):
        return adamax_updates(loss_or_gradients, parameters, learning_rate, beta1, beta2, epsilon)

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

    Returns
    -------
    UpdateRule
        A configured optimizer ``(loss_or_gradients, parameters) -> Updates``.
    """

    def rule(loss_or_gradients, parameters):
        return rprop_updates(
            loss_or_gradients, parameters, learning_rate, eta_minus, eta_plus, step_min, step_max
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

    Returns
    -------
    UpdateRule
        A configured optimizer ``(loss_or_gradients, parameters) -> Updates``.
    """

    def rule(loss_or_gradients, parameters):
        return rmsprop_updates(
            loss_or_gradients, parameters, learning_rate, rho, momentum, epsilon, centered
        )

    return rule


def adagrad(learning_rate: float = 0.01, epsilon: float = 1e-8) -> UpdateRule:
    """
    AdaGrad optimizer. See :func:`~pytensor_ml.optim.rules.adagrad_updates`.

    Returns
    -------
    UpdateRule
        A configured optimizer ``(loss_or_gradients, parameters) -> Updates``.
    """

    def rule(loss_or_gradients, parameters):
        return adagrad_updates(loss_or_gradients, parameters, learning_rate, epsilon)

    return rule


def adadelta(learning_rate: float = 1.0, rho: float = 0.9, epsilon: float = 1e-8) -> UpdateRule:
    """
    AdaDelta optimizer. See :func:`~pytensor_ml.optim.rules.adadelta_updates`.

    Returns
    -------
    UpdateRule
        A configured optimizer ``(loss_or_gradients, parameters) -> Updates``.
    """

    def rule(loss_or_gradients, parameters):
        return adadelta_updates(loss_or_gradients, parameters, learning_rate, rho, epsilon)

    return rule
