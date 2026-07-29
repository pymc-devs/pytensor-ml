from collections.abc import Callable, Sequence

import pytensor.tensor as pt

from pytensor import config
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import (
    LossOrGradients,
    Parameter,
    Updates,
    counter,
    get_gradients,
    state_for,
)


def sgd_updates(
    loss_or_gradients: LossOrGradients,
    parameters: Sequence[Parameter],
    learning_rate: float = 1.0,
) -> Updates:
    r"""
    Vanilla stochastic gradient descent: :math:`p \leftarrow p - \eta g`.

    A default ``learning_rate`` of 1.0 makes the result a unit-rate descent direction, ready to seed a chain
    whose terminal :func:`~pytensor_ml.optim.transform.scale` applies the actual rate.

    Parameters
    ----------
    loss_or_gradients : TensorVariable or sequence of TensorVariable
        Scalar loss to differentiate, or precomputed gradients.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float
        Step size :math:`\eta`. Default 1.0.

    Returns
    -------
    Updates
        Mapping from each parameter to its next value.
    """
    gradients = get_gradients(loss_or_gradients, parameters)
    return {
        parameter: parameter - learning_rate * gradient
        for parameter, gradient in zip(parameters, gradients)
    }


def adam_updates(
    loss_or_gradients: LossOrGradients,
    parameters: Sequence[Parameter],
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    amsgrad: bool = False,
) -> Updates:
    r"""
    Adam optimizer.

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        p &\leftarrow p - \eta \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}

    Parameters
    ----------
    loss_or_gradients : TensorVariable or sequence of TensorVariable
        Scalar loss to differentiate, or precomputed gradients.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float
        Step size :math:`\eta`. Default 1e-3.
    beta1 : float
        Exponential decay rate for the first moment :math:`m`. Default 0.9.
    beta2 : float
        Exponential decay rate for the second moment :math:`v`. Default 0.999.
    epsilon : float
        Constant added to the denominator for numerical stability. Default 1e-8.
    amsgrad : bool
        Use the AMSGrad variant, dividing by the running maximum of the second moment so the denominator is
        non-decreasing. Default False.

    Returns
    -------
    Updates
        Mapping from each parameter and its moment buffers to their next values.
    """
    gradients = get_gradients(loss_or_gradients, parameters)

    step_count = counter("adam/step_count")
    new_step_count = step_count + 1
    new_step_count_float = new_step_count.astype(config.floatX)
    first_moment_bias_correction = 1 - beta1**new_step_count_float
    second_moment_bias_correction = 1 - beta2**new_step_count_float

    updates: Updates = {step_count: new_step_count}
    for parameter, gradient in zip(parameters, gradients):
        first_moment = state_for(parameter, "adam/first_moment")
        second_moment = state_for(parameter, "adam/second_moment")

        new_first_moment = beta1 * first_moment + (1 - beta1) * gradient
        new_second_moment = beta2 * second_moment + (1 - beta2) * gradient**2
        updates[first_moment] = new_first_moment
        updates[second_moment] = new_second_moment

        second_moment_for_denominator = amsgrad_second_moment(
            parameter, new_second_moment, updates, amsgrad
        )
        corrected_first_moment = new_first_moment / first_moment_bias_correction
        corrected_second_moment = second_moment_for_denominator / second_moment_bias_correction

        updates[parameter] = parameter - learning_rate * corrected_first_moment / (
            pt.sqrt(corrected_second_moment) + epsilon
        )

    return updates


def amsgrad_second_moment(
    parameter: Parameter,
    new_second_moment: TensorVariable,
    updates: Updates,
    amsgrad: bool,
) -> TensorVariable:
    """
    Return the second moment to divide by, tracking its running maximum when ``amsgrad`` is set.

    The maximum buffer is registered in ``updates`` in place so the caller's dict carries it forward.
    """
    if not amsgrad:
        return new_second_moment
    max_second_moment = state_for(parameter, "adam/max_second_moment")
    new_max_second_moment = pt.maximum(max_second_moment, new_second_moment)
    updates[max_second_moment] = new_max_second_moment
    return new_max_second_moment


def adamw_updates(
    loss_or_gradients: LossOrGradients,
    parameters: Sequence[Parameter],
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    amsgrad: bool = False,
    mask: Callable[[Parameter], bool] | None = None,
) -> Updates:
    r"""
    AdamW: Adam with decoupled weight decay applied directly to the parameter, not the gradient.

    .. math::

        p \leftarrow p - \eta \left( \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} + \lambda p \right)

    Keeping the decay term :math:`\lambda p` outside the moment estimates keeps it correct under a scheduled
    learning rate.

    Parameters
    ----------
    loss_or_gradients : TensorVariable or sequence of TensorVariable
        Scalar loss to differentiate, or precomputed gradients.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float
        Step size :math:`\eta`. Default 1e-3.
    weight_decay : float
        Decoupled decay coefficient :math:`\lambda`. Default 0.01.
    beta1 : float
        Exponential decay rate for the first moment. Default 0.9.
    beta2 : float
        Exponential decay rate for the second moment. Default 0.999.
    epsilon : float
        Constant added to the denominator for numerical stability. Default 1e-8.
    amsgrad : bool
        Use the AMSGrad variant, dividing by the running maximum of the second moment so the denominator is
        non-decreasing. Default False.
    mask : callable, optional
        Predicate ``(parameter) -> bool`` selecting which parameters receive decay. Decay is applied to every
        parameter when omitted.

    Returns
    -------
    Updates
        Mapping from each parameter and its moment buffers to their next values.
    """
    gradients = get_gradients(loss_or_gradients, parameters)

    step_count = counter("adam/step_count")
    new_step_count = step_count + 1
    new_step_count_float = new_step_count.astype(config.floatX)
    first_moment_bias_correction = 1 - beta1**new_step_count_float
    second_moment_bias_correction = 1 - beta2**new_step_count_float

    updates: Updates = {step_count: new_step_count}
    for parameter, gradient in zip(parameters, gradients):
        first_moment = state_for(parameter, "adam/first_moment")
        second_moment = state_for(parameter, "adam/second_moment")

        new_first_moment = beta1 * first_moment + (1 - beta1) * gradient
        new_second_moment = beta2 * second_moment + (1 - beta2) * gradient**2
        updates[first_moment] = new_first_moment
        updates[second_moment] = new_second_moment

        second_moment_for_denominator = amsgrad_second_moment(
            parameter, new_second_moment, updates, amsgrad
        )
        adam_update = (new_first_moment / first_moment_bias_correction) / (
            pt.sqrt(second_moment_for_denominator / second_moment_bias_correction) + epsilon
        )
        decay_term = weight_decay * parameter if (mask is None or mask(parameter)) else 0.0
        updates[parameter] = parameter - learning_rate * (adam_update + decay_term)

    return updates


def nadam_updates(
    loss_or_gradients: LossOrGradients,
    parameters: Sequence[Parameter],
    learning_rate: float = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> Updates:
    r"""
    Nadam: Adam with Nesterov momentum applied to the first-moment estimate.

    .. math::

        p \leftarrow p - \eta \frac{\beta_1 \hat{m}_t + (1 - \beta_1) g_t / (1 - \beta_1^t)}
                                   {\sqrt{\hat{v}_t} + \epsilon}

    where :math:`\hat{m}_t` and :math:`\hat{v}_t` are the bias-corrected Adam moments. Replacing
    :math:`\hat{m}_t` with the look-ahead numerator is the Nesterov step applied to Adam.

    Parameters
    ----------
    loss_or_gradients : TensorVariable or sequence of TensorVariable
        Scalar loss to differentiate, or precomputed gradients.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float
        Step size :math:`\eta`. Default 2e-3.
    beta1 : float
        Exponential decay rate for the first moment. Default 0.9.
    beta2 : float
        Exponential decay rate for the second moment. Default 0.999.
    epsilon : float
        Constant added to the denominator for numerical stability. Default 1e-8.

    Returns
    -------
    Updates
        Mapping from each parameter and its moment buffers to their next values.
    """
    gradients = get_gradients(loss_or_gradients, parameters)

    step_count = counter("nadam/step_count")
    new_step_count = step_count + 1
    new_step_count_float = new_step_count.astype(config.floatX)
    first_moment_bias_correction = 1 - beta1**new_step_count_float
    second_moment_bias_correction = 1 - beta2**new_step_count_float

    updates: Updates = {step_count: new_step_count}
    for parameter, gradient in zip(parameters, gradients):
        first_moment = state_for(parameter, "nadam/first_moment")
        second_moment = state_for(parameter, "nadam/second_moment")

        new_first_moment = beta1 * first_moment + (1 - beta1) * gradient
        new_second_moment = beta2 * second_moment + (1 - beta2) * gradient**2

        corrected_second_moment = new_second_moment / second_moment_bias_correction
        nesterov_first_moment = (
            beta1 * new_first_moment + (1 - beta1) * gradient
        ) / first_moment_bias_correction

        updates[first_moment] = new_first_moment
        updates[second_moment] = new_second_moment
        updates[parameter] = parameter - learning_rate * nesterov_first_moment / (
            pt.sqrt(corrected_second_moment) + epsilon
        )

    return updates


def adamax_updates(
    loss_or_gradients: LossOrGradients,
    parameters: Sequence[Parameter],
    learning_rate: float = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> Updates:
    r"""
    AdaMax: Adam variant using an exponentially weighted infinity norm instead of the second moment.

    .. math::

        u_t &= \max(\beta_2 u_{t-1}, |g_t|) \\
        p &\leftarrow p - \frac{\eta}{1 - \beta_1^t} \frac{m_t}{u_t}

    The infinity norm :math:`u` needs no bias correction, so only the first moment is corrected.

    Parameters
    ----------
    loss_or_gradients : TensorVariable or sequence of TensorVariable
        Scalar loss to differentiate, or precomputed gradients.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float
        Step size :math:`\eta`. Default 2e-3.
    beta1 : float
        Exponential decay rate for the first moment. Default 0.9.
    beta2 : float
        Decay rate for the infinity norm :math:`u`. Default 0.999.
    epsilon : float
        Floor added to :math:`|g_t|` so the denominator stays positive. Default 1e-8.

    Returns
    -------
    Updates
        Mapping from each parameter and its state buffers to their next values.
    """
    gradients = get_gradients(loss_or_gradients, parameters)

    step_count = counter("adamax/step_count")
    new_step_count = step_count + 1
    new_step_count_float = new_step_count.astype(config.floatX)
    first_moment_bias_correction = 1 - beta1**new_step_count_float

    updates: Updates = {step_count: new_step_count}
    for parameter, gradient in zip(parameters, gradients):
        first_moment = state_for(parameter, "adamax/first_moment")
        infinity_norm = state_for(parameter, "adamax/infinity_norm")

        new_first_moment = beta1 * first_moment + (1 - beta1) * gradient
        new_infinity_norm = pt.maximum(beta2 * infinity_norm, pt.abs(gradient) + epsilon)

        updates[first_moment] = new_first_moment
        updates[infinity_norm] = new_infinity_norm
        updates[parameter] = parameter - (learning_rate / first_moment_bias_correction) * (
            new_first_moment / new_infinity_norm
        )

    return updates


def adagrad_updates(
    loss_or_gradients: LossOrGradients,
    parameters: Sequence[Parameter],
    learning_rate: float = 0.01,
    epsilon: float = 1e-8,
) -> Updates:
    r"""
    AdaGrad: per-parameter learning rate scaled by the inverse root of accumulated squared gradients.

    .. math::

        G &\leftarrow G + g^2 \\
        p &\leftarrow p - \eta \frac{g}{\sqrt{G + \epsilon}}

    Parameters
    ----------
    loss_or_gradients : TensorVariable or sequence of TensorVariable
        Scalar loss to differentiate, or precomputed gradients.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float
        Step size :math:`\eta`. Default 0.01.
    epsilon : float
        Constant added under the root for numerical stability. Default 1e-8.

    Returns
    -------
    Updates
        Mapping from each parameter and its accumulator to their next values.
    """
    gradients = get_gradients(loss_or_gradients, parameters)

    updates: Updates = {}
    for parameter, gradient in zip(parameters, gradients):
        sum_squared_gradients = state_for(parameter, "adagrad/sum_squared_gradients")
        new_sum_squared_gradients = sum_squared_gradients + gradient**2
        updates[sum_squared_gradients] = new_sum_squared_gradients
        updates[parameter] = parameter - learning_rate * gradient / pt.sqrt(
            new_sum_squared_gradients + epsilon
        )

    return updates


def rmsprop_updates(
    loss_or_gradients: LossOrGradients,
    parameters: Sequence[Parameter],
    learning_rate: float = 1e-2,
    rho: float = 0.9,
    momentum: float = 0.0,
    epsilon: float = 1e-8,
    centered: bool = False,
) -> Updates:
    r"""
    RMSProp: per-parameter learning rate scaled by a decaying average of squared gradients.

    .. math::

        v &\leftarrow \rho v + (1 - \rho) g^2 \\
        p &\leftarrow p - \eta \frac{g}{\sqrt{v + \epsilon}}

    When ``centered`` is set the variance estimate is centered by a decaying average of the gradient,
    :math:`\sqrt{v - \bar{g}^2 + \epsilon}`. When ``momentum`` is nonzero the scaled gradient is accumulated
    into a velocity buffer before the step.

    Parameters
    ----------
    loss_or_gradients : TensorVariable or sequence of TensorVariable
        Scalar loss to differentiate, or precomputed gradients.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float
        Step size :math:`\eta`. Default 1e-2.
    rho : float
        Decay rate for the running average of squared gradients. Default 0.9.
    momentum : float
        Momentum coefficient. A value of 0 (the default) gives plain RMSProp.
    epsilon : float
        Constant added under the root for numerical stability. Default 1e-8.
    centered : bool
        Center the variance estimate by the squared running mean of the gradient. Default False.

    Returns
    -------
    Updates
        Mapping from each parameter and its state buffers to their next values.
    """
    gradients = get_gradients(loss_or_gradients, parameters)

    updates: Updates = {}
    for parameter, gradient in zip(parameters, gradients):
        mean_squared_gradient = state_for(parameter, "rmsprop/mean_squared_gradient")
        new_mean_squared_gradient = rho * mean_squared_gradient + (1 - rho) * gradient**2
        updates[mean_squared_gradient] = new_mean_squared_gradient

        variance = new_mean_squared_gradient
        if centered:
            mean_gradient = state_for(parameter, "rmsprop/mean_gradient")
            new_mean_gradient = rho * mean_gradient + (1 - rho) * gradient
            updates[mean_gradient] = new_mean_gradient
            variance = variance - new_mean_gradient**2

        scaled_gradient = gradient / pt.sqrt(variance + epsilon)

        if momentum:
            velocity = state_for(parameter, "rmsprop/velocity")
            new_velocity = momentum * velocity + scaled_gradient
            updates[velocity] = new_velocity
            updates[parameter] = parameter - learning_rate * new_velocity
        else:
            updates[parameter] = parameter - learning_rate * scaled_gradient

    return updates


def adadelta_updates(
    loss_or_gradients: LossOrGradients,
    parameters: Sequence[Parameter],
    learning_rate: float = 1.0,
    rho: float = 0.9,
    epsilon: float = 1e-8,
) -> Updates:
    r"""
    AdaDelta: AdaGrad variant with a decaying window of squared gradients and squared updates.

    .. math::

        v &\leftarrow \rho v + (1 - \rho) g^2 \\
        \Delta &= \frac{\sqrt{u + \epsilon}}{\sqrt{v + \epsilon}} g \\
        u &\leftarrow \rho u + (1 - \rho) \Delta^2 \\
        p &\leftarrow p - \eta \Delta

    Parameters
    ----------
    loss_or_gradients : TensorVariable or sequence of TensorVariable
        Scalar loss to differentiate, or precomputed gradients.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float
        Step size :math:`\eta`. Default 1.0.
    rho : float
        Decay rate for the running averages. Default 0.9.
    epsilon : float
        Constant added under the roots for numerical stability. Default 1e-8.

    Returns
    -------
    Updates
        Mapping from each parameter and its two accumulators to their next values.
    """
    gradients = get_gradients(loss_or_gradients, parameters)

    updates: Updates = {}
    for parameter, gradient in zip(parameters, gradients):
        accumulated_squared_gradient = state_for(parameter, "adadelta/accumulated_squared_gradient")
        accumulated_squared_update = state_for(parameter, "adadelta/accumulated_squared_update")

        new_accumulated_squared_gradient = (
            rho * accumulated_squared_gradient + (1 - rho) * gradient**2
        )
        update = (
            pt.sqrt(accumulated_squared_update + epsilon)
            / pt.sqrt(new_accumulated_squared_gradient + epsilon)
            * gradient
        )
        new_accumulated_squared_update = rho * accumulated_squared_update + (1 - rho) * update**2

        updates[accumulated_squared_gradient] = new_accumulated_squared_gradient
        updates[accumulated_squared_update] = new_accumulated_squared_update
        updates[parameter] = parameter - learning_rate * update

    return updates


def rprop_updates(
    loss_or_gradients: LossOrGradients,
    parameters: Sequence[Parameter],
    learning_rate: float = 1e-2,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    step_min: float = 1e-6,
    step_max: float = 50.0,
) -> Updates:
    r"""
    Rprop: resilient backpropagation, stepping by a per-parameter magnitude that adapts to gradient-sign
    agreement and ignores gradient magnitude.

    Each coordinate's step size grows by ``eta_plus`` while the gradient keeps its sign and shrinks by
    ``eta_minus`` when the sign flips; on a flip the step is skipped and the remembered gradient is zeroed so
    the next iteration is treated as neutral. This is the non-backtracking variant (Rprop\ :sup:`-`).

    Being a full-batch method, Rprop assumes the gradient is not stochastic across steps.

    Parameters
    ----------
    loss_or_gradients : TensorVariable or sequence of TensorVariable
        Scalar loss to differentiate, or precomputed gradients.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float
        Initial per-parameter step size. Default 1e-2.
    eta_minus : float
        Multiplicative decrease applied on a gradient-sign flip. Default 0.5.
    eta_plus : float
        Multiplicative increase applied when the gradient sign is unchanged. Default 1.2.
    step_min : float
        Lower clamp on the step size. Default 1e-6.
    step_max : float
        Upper clamp on the step size. Default 50.0.

    Returns
    -------
    Updates
        Mapping from each parameter and its state buffers to their next values.
    """
    gradients = get_gradients(loss_or_gradients, parameters)

    updates: Updates = {}
    for parameter, gradient in zip(parameters, gradients):
        previous_gradient = state_for(parameter, "rprop/previous_gradient")
        step_size = state_for(parameter, "rprop/step_size", fill_value=learning_rate)

        sign_agreement = gradient * previous_gradient
        step_multiplier = pt.switch(
            sign_agreement > 0,
            eta_plus,
            pt.switch(sign_agreement < 0, eta_minus, 1.0),
        )
        new_step_size = pt.clip(step_size * step_multiplier, step_min, step_max)
        effective_gradient = pt.switch(sign_agreement < 0, 0.0, gradient)

        updates[step_size] = new_step_size
        updates[previous_gradient] = effective_gradient
        updates[parameter] = parameter - pt.sign(effective_gradient) * new_step_size

    return updates
