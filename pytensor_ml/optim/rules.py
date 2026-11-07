from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor import config
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.conditional import ifelse
from pytensor.graph.basic import Variable
from pytensor.scalar import upcast
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import (
    LearningRate,
    LossGradientsOrUpdates,
    Parameter,
    Steps,
    Updates,
    gradients_to_descend,
    rate_on,
    read_rate,
    scalar_state,
    state_for,
    to_floatx,
)
from pytensor_ml.optim.lbfgs import LBFGSDirection, flat_dot
from pytensor_ml.optim.line_search import LineSearch, SearchResult, search_along
from pytensor_ml.params import step_counter


def sgd_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    learning_rate: LearningRate = 1.0,
    namespace: str = "sgd",
) -> Updates:
    r"""
    Vanilla stochastic gradient descent: :math:`p \leftarrow p - \eta g`.

    A default ``learning_rate`` of 1.0 makes the result a unit-rate descent direction, ready to seed a chain
    whose terminal :func:`~pytensor_ml.optim.transform.scale` applies the actual rate.

    Parameters
    ----------
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Scalar loss to differentiate, precomputed gradients, or the updates dict an earlier transform in
        a chain produced.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float or shared tensor variable
        Step size :math:`\eta`. Default 1.0.

    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate state
        rather than reusing each other's. Default is the rule's own name.

    Returns
    -------
    updates : Updates
        Mapping from each parameter to its next value.

    Examples
    --------
    The update function behind :func:`sgd`, for compiling the step yourself rather
    than going through :func:`~pytensor_ml.optim.train.compile_train`. It returns the updates dict directly, and its rate defaults to 1.0 rather than the alias's 0.01:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import sgd_updates
        from pytensor_ml.pytensorf import collect_trainable_params, function

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        updates = sgd_updates(loss, collect_trainable_params(loss), learning_rate=0.1)
        step = function([X, target], loss, updates=updates)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    incoming, gradients = gradients_to_descend(loss_gradients_or_updates, parameters, namespace)
    learning_rate, clock_update = read_rate(learning_rate, namespace)
    steps: dict[SharedVariable, TensorVariable] = {
        parameter: parameter - learning_rate * gradient
        for parameter, gradient in zip(parameters, gradients)
    }
    return Steps(incoming).replacing(clock_update).replacing(steps)


def adam_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    learning_rate: LearningRate = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    amsgrad: bool = False,
    namespace: str = "adam",
) -> Updates:
    r"""
    Adam optimizer.

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        p &\leftarrow p - \eta \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}

    Parameters
    ----------
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Scalar loss to differentiate, precomputed gradients, or the updates dict an earlier transform in
        a chain produced.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float or shared tensor variable
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

    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate state
        rather than reusing each other's. Default is the rule's own name.

    Returns
    -------
    updates : Updates
        Mapping from each parameter and its moment buffers to their next values.

    Examples
    --------
    The update function behind :func:`adam`, for compiling the step yourself rather
    than going through :func:`~pytensor_ml.optim.train.compile_train`. It returns the updates dict directly:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam_updates
        from pytensor_ml.pytensorf import collect_trainable_params, function

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        updates = adam_updates(loss, collect_trainable_params(loss), learning_rate=1e-3)
        step = function([X, target], loss, updates=updates)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    return _adam_family_updates(
        loss_gradients_or_updates,
        parameters,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        amsgrad=amsgrad,
        namespace=namespace,
    )


def _adam_family_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    *,
    learning_rate: LearningRate,
    beta1: float,
    beta2: float,
    epsilon: float,
    amsgrad: bool,
    namespace: str,
    weight_decay: float = 0.0,
    mask: Callable[[Parameter], bool] | None = None,
) -> Updates:
    """
    Adam's update, shared by :func:`adam_updates` and :func:`adamw_updates`.

    Parameters
    ----------
    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate moments
        and separate step counts rather than reusing each other's.
    weight_decay : float
        Coefficient of the decoupled decay term added to the update. Zero adds no term at all. Default 0.0.
    mask : callable, optional
        Predicate selecting which parameters the decay reaches. Every parameter when omitted.
    """
    incoming, gradients = gradients_to_descend(loss_gradients_or_updates, parameters, namespace)
    step_count = step_counter(f"{namespace}/step_count")
    learning_rate = to_floatx(rate_on(learning_rate, step_count))
    new_step_count = step_count + 1
    new_step_count_float = new_step_count.astype(config.floatX)
    first_moment_bias_correction = 1 - beta1**new_step_count_float
    second_moment_bias_correction = 1 - beta2**new_step_count_float

    updates: Updates = Steps(incoming)
    updates[step_count] = new_step_count
    for parameter, gradient in zip(parameters, gradients):
        first_moment = state_for(parameter, f"{namespace}/first_moment")
        second_moment = state_for(parameter, f"{namespace}/second_moment")

        new_first_moment = beta1 * first_moment + (1 - beta1) * gradient
        new_second_moment = beta2 * second_moment + (1 - beta2) * gradient**2
        updates[first_moment] = new_first_moment
        updates[second_moment] = new_second_moment

        second_moment_for_denominator = (
            _running_max_second_moment(
                parameter, new_second_moment, updates, f"{namespace}/max_second_moment"
            )
            if amsgrad
            else new_second_moment
        )
        corrected_first_moment = new_first_moment / first_moment_bias_correction
        corrected_second_moment = second_moment_for_denominator / second_moment_bias_correction
        denominator = pt.sqrt(corrected_second_moment) + epsilon

        # Decoupled decay is added inside the rate multiplication -- that is what decouples it from the
        # moments -- and without decay there is nothing to add. The two groupings agree in exact arithmetic
        # and differ in the last bits, so folding them into one moves a third of float32 adam results.
        if weight_decay:
            decay_term = weight_decay * parameter if (mask is None or mask(parameter)) else 0.0
            step = learning_rate * (corrected_first_moment / denominator + decay_term)
        else:
            step = learning_rate * corrected_first_moment / denominator

        updates[parameter] = parameter - step

    return updates


def _running_max_second_moment(
    parameter: Parameter,
    new_second_moment: TensorVariable,
    updates: Updates,
    slot: str,
) -> TensorVariable:
    """
    Return the running maximum of the second moment, which AMSGrad divides by instead of the moment itself.

    The buffer is allocated under ``slot`` and registered in ``updates`` in place, so the caller's dict
    carries it forward.
    """
    max_second_moment = state_for(parameter, slot)
    new_max_second_moment = pt.maximum(max_second_moment, new_second_moment)
    updates[max_second_moment] = new_max_second_moment
    return new_max_second_moment


def adamw_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    learning_rate: LearningRate = 1e-3,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    amsgrad: bool = False,
    mask: Callable[[Parameter], bool] | None = None,
    namespace: str = "adamw",
) -> Updates:
    r"""
    AdamW: Adam with decoupled weight decay applied directly to the parameter, not the gradient.

    .. math::

        p \leftarrow p - \eta \left( \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} + \lambda p \right)

    Keeping the decay term :math:`\lambda p` outside the moment estimates keeps it correct under a scheduled
    learning rate.

    Parameters
    ----------
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Scalar loss to differentiate, precomputed gradients, or the updates dict an earlier transform in
        a chain produced.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float or shared tensor variable
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

    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate state
        rather than reusing each other's. Default is the rule's own name.

    Returns
    -------
    updates : Updates
        Mapping from each parameter and its moment buffers to their next values.

    Examples
    --------
    The update function behind :func:`adamw`, for compiling the step yourself rather
    than going through :func:`~pytensor_ml.optim.train.compile_train`. It returns the updates dict directly, decoupling the weight decay from the adaptive rate:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adamw_updates
        from pytensor_ml.pytensorf import collect_trainable_params, function

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        updates = adamw_updates(loss, collect_trainable_params(loss), learning_rate=1e-3, weight_decay=0.01)
        step = function([X, target], loss, updates=updates)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    return _adam_family_updates(
        loss_gradients_or_updates,
        parameters,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        amsgrad=amsgrad,
        namespace=namespace,
        weight_decay=weight_decay,
        mask=mask,
    )


def nadam_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    learning_rate: LearningRate = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    namespace: str = "nadam",
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
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Scalar loss to differentiate, precomputed gradients, or the updates dict an earlier transform in
        a chain produced.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float or shared tensor variable
        Step size :math:`\eta`. Default 2e-3.
    beta1 : float
        Exponential decay rate for the first moment. Default 0.9.
    beta2 : float
        Exponential decay rate for the second moment. Default 0.999.
    epsilon : float
        Constant added to the denominator for numerical stability. Default 1e-8.

    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate state
        rather than reusing each other's. Default is the rule's own name.

    Returns
    -------
    updates : Updates
        Mapping from each parameter and its moment buffers to their next values.

    Examples
    --------
    The update function behind :func:`nadam`, for compiling the step yourself rather
    than going through :func:`~pytensor_ml.optim.train.compile_train`. It returns the updates dict directly:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import nadam_updates
        from pytensor_ml.pytensorf import collect_trainable_params, function

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        updates = nadam_updates(loss, collect_trainable_params(loss), learning_rate=2e-3)
        step = function([X, target], loss, updates=updates)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    incoming, gradients = gradients_to_descend(loss_gradients_or_updates, parameters, namespace)
    step_count = step_counter(f"{namespace}/step_count")
    learning_rate = to_floatx(rate_on(learning_rate, step_count))
    new_step_count = step_count + 1
    new_step_count_float = new_step_count.astype(config.floatX)
    first_moment_bias_correction = 1 - beta1**new_step_count_float
    second_moment_bias_correction = 1 - beta2**new_step_count_float

    updates: Updates = Steps(incoming)
    updates[step_count] = new_step_count
    for parameter, gradient in zip(parameters, gradients):
        first_moment = state_for(parameter, f"{namespace}/first_moment")
        second_moment = state_for(parameter, f"{namespace}/second_moment")

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
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    learning_rate: LearningRate = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    namespace: str = "adamax",
) -> Updates:
    r"""
    AdaMax: Adam variant using an exponentially weighted infinity norm instead of the second moment.

    .. math::

        u_t &= \max(\beta_2 u_{t-1}, |g_t|) \\
        p &\leftarrow p - \frac{\eta}{1 - \beta_1^t} \frac{m_t}{u_t}

    The infinity norm :math:`u` needs no bias correction, so only the first moment is corrected.

    Parameters
    ----------
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Scalar loss to differentiate, precomputed gradients, or the updates dict an earlier transform in
        a chain produced.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float or shared tensor variable
        Step size :math:`\eta`. Default 2e-3.
    beta1 : float
        Exponential decay rate for the first moment. Default 0.9.
    beta2 : float
        Decay rate for the infinity norm :math:`u`. Default 0.999.
    epsilon : float
        Floor added to :math:`|g_t|` so the denominator stays positive. Default 1e-8.

    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate state
        rather than reusing each other's. Default is the rule's own name.

    Returns
    -------
    updates : Updates
        Mapping from each parameter and its state buffers to their next values.

    Examples
    --------
    The update function behind :func:`adamax`, for compiling the step yourself rather
    than going through :func:`~pytensor_ml.optim.train.compile_train`. It returns the updates dict directly:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adamax_updates
        from pytensor_ml.pytensorf import collect_trainable_params, function

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        updates = adamax_updates(loss, collect_trainable_params(loss), learning_rate=2e-3)
        step = function([X, target], loss, updates=updates)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    incoming, gradients = gradients_to_descend(loss_gradients_or_updates, parameters, namespace)
    step_count = step_counter(f"{namespace}/step_count")
    learning_rate = to_floatx(rate_on(learning_rate, step_count))
    new_step_count = step_count + 1
    new_step_count_float = new_step_count.astype(config.floatX)
    first_moment_bias_correction = 1 - beta1**new_step_count_float

    updates: Updates = Steps(incoming)
    updates[step_count] = new_step_count
    for parameter, gradient in zip(parameters, gradients):
        first_moment = state_for(parameter, f"{namespace}/first_moment")
        infinity_norm = state_for(parameter, f"{namespace}/infinity_norm")

        new_first_moment = beta1 * first_moment + (1 - beta1) * gradient
        new_infinity_norm = pt.maximum(beta2 * infinity_norm, pt.abs(gradient) + epsilon)

        updates[first_moment] = new_first_moment
        updates[infinity_norm] = new_infinity_norm
        updates[parameter] = parameter - (learning_rate / first_moment_bias_correction) * (
            new_first_moment / new_infinity_norm
        )

    return updates


def adagrad_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    learning_rate: LearningRate = 0.01,
    epsilon: float = 1e-8,
    namespace: str = "adagrad",
) -> Updates:
    r"""
    AdaGrad: per-parameter learning rate scaled by the inverse root of accumulated squared gradients.

    .. math::

        G &\leftarrow G + g^2 \\
        p &\leftarrow p - \eta \frac{g}{\sqrt{G + \epsilon}}

    Parameters
    ----------
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Scalar loss to differentiate, precomputed gradients, or the updates dict an earlier transform in
        a chain produced.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float or shared tensor variable
        Step size :math:`\eta`. Default 0.01.
    epsilon : float
        Constant added under the root for numerical stability. Default 1e-8.

    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate state
        rather than reusing each other's. Default is the rule's own name.

    Returns
    -------
    updates : Updates
        Mapping from each parameter and its accumulator to their next values.

    Examples
    --------
    The update function behind :func:`adagrad`, for compiling the step yourself rather
    than going through :func:`~pytensor_ml.optim.train.compile_train`. It returns the updates dict directly:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adagrad_updates
        from pytensor_ml.pytensorf import collect_trainable_params, function

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        updates = adagrad_updates(loss, collect_trainable_params(loss), learning_rate=1e-2)
        step = function([X, target], loss, updates=updates)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    incoming, gradients = gradients_to_descend(loss_gradients_or_updates, parameters, namespace)
    learning_rate, clock_update = read_rate(learning_rate, namespace)

    updates: Updates = Steps(incoming).replacing(clock_update)
    for parameter, gradient in zip(parameters, gradients):
        sum_squared_gradients = state_for(parameter, f"{namespace}/sum_squared_gradients")
        new_sum_squared_gradients = sum_squared_gradients + gradient**2
        updates[sum_squared_gradients] = new_sum_squared_gradients
        updates[parameter] = parameter - learning_rate * gradient / pt.sqrt(
            new_sum_squared_gradients + epsilon
        )

    return updates


def rmsprop_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    learning_rate: LearningRate = 1e-2,
    rho: float = 0.9,
    momentum: float = 0.0,
    epsilon: float = 1e-8,
    centered: bool = False,
    namespace: str = "rmsprop",
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
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Scalar loss to differentiate, precomputed gradients, or the updates dict an earlier transform in
        a chain produced.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float or shared tensor variable
        Step size :math:`\eta`. Default 1e-2.
    rho : float
        Decay rate for the running average of squared gradients. Default 0.9.
    momentum : float
        Momentum coefficient. A value of 0 (the default) gives plain RMSProp.
    epsilon : float
        Constant added under the root for numerical stability. Default 1e-8.
    centered : bool
        Center the variance estimate by the squared running mean of the gradient. Default False.

    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate state
        rather than reusing each other's. Default is the rule's own name.

    Returns
    -------
    updates : Updates
        Mapping from each parameter and its state buffers to their next values.

    Examples
    --------
    The update function behind :func:`rmsprop`, for compiling the step yourself rather
    than going through :func:`~pytensor_ml.optim.train.compile_train`. It returns the updates dict directly:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import rmsprop_updates
        from pytensor_ml.pytensorf import collect_trainable_params, function

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        updates = rmsprop_updates(loss, collect_trainable_params(loss), learning_rate=1e-2)
        step = function([X, target], loss, updates=updates)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    incoming, gradients = gradients_to_descend(loss_gradients_or_updates, parameters, namespace)
    learning_rate, clock_update = read_rate(learning_rate, namespace)

    updates: Updates = Steps(incoming).replacing(clock_update)
    for parameter, gradient in zip(parameters, gradients):
        mean_squared_gradient = state_for(parameter, f"{namespace}/mean_squared_gradient")
        new_mean_squared_gradient = rho * mean_squared_gradient + (1 - rho) * gradient**2
        updates[mean_squared_gradient] = new_mean_squared_gradient

        variance = new_mean_squared_gradient
        if centered:
            mean_gradient = state_for(parameter, f"{namespace}/mean_gradient")
            new_mean_gradient = rho * mean_gradient + (1 - rho) * gradient
            updates[mean_gradient] = new_mean_gradient
            variance = variance - new_mean_gradient**2

        scaled_gradient = gradient / pt.sqrt(variance + epsilon)

        if momentum:
            velocity = state_for(parameter, f"{namespace}/velocity")
            new_velocity = momentum * velocity + scaled_gradient
            updates[velocity] = new_velocity
            updates[parameter] = parameter - learning_rate * new_velocity
        else:
            updates[parameter] = parameter - learning_rate * scaled_gradient

    return updates


def adadelta_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    learning_rate: LearningRate = 1.0,
    rho: float = 0.9,
    epsilon: float = 1e-8,
    namespace: str = "adadelta",
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
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Scalar loss to differentiate, precomputed gradients, or the updates dict an earlier transform in
        a chain produced.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float or shared tensor variable
        Step size :math:`\eta`. Default 1.0.
    rho : float
        Decay rate for the running averages. Default 0.9.
    epsilon : float
        Constant added under the roots for numerical stability. Default 1e-8.

    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate state
        rather than reusing each other's. Default is the rule's own name.

    Returns
    -------
    updates : Updates
        Mapping from each parameter and its two accumulators to their next values.

    Examples
    --------
    The update function behind :func:`adadelta`, for compiling the step yourself rather
    than going through :func:`~pytensor_ml.optim.train.compile_train`. It returns the updates dict directly, needing no learning rate of its own:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adadelta_updates
        from pytensor_ml.pytensorf import collect_trainable_params, function

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        updates = adadelta_updates(loss, collect_trainable_params(loss))
        step = function([X, target], loss, updates=updates)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    incoming, gradients = gradients_to_descend(loss_gradients_or_updates, parameters, namespace)
    learning_rate, clock_update = read_rate(learning_rate, namespace)

    updates: Updates = Steps(incoming).replacing(clock_update)
    for parameter, gradient in zip(parameters, gradients):
        accumulated_squared_gradient = state_for(
            parameter, f"{namespace}/accumulated_squared_gradient"
        )
        accumulated_squared_update = state_for(parameter, f"{namespace}/accumulated_squared_update")

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


def _require_numeric_learning_rate(learning_rate: LearningRate) -> None:
    """
    Raise ``TypeError`` unless ``learning_rate`` is a plain number.

    Rprop's rate initializes the per-parameter step sizes it then adapts, so it is consumed at allocation
    time and never reaches the graph. A schedule or shared variable would fail deep inside numpy instead.

    Parameters
    ----------
    learning_rate : float, shared tensor variable, or Schedule
        The rate to check.
    """
    if callable(learning_rate) or isinstance(learning_rate, Variable):
        raise TypeError(
            "rprop's learning rate initializes its per-parameter step sizes rather than scaling the step, "
            "so it must be a plain number and cannot be scheduled or steered. Schedule a rule whose rate "
            "multiplies the step, or scale rprop's finished step with "
            "`chain(rprop(...), scale(curve(clock)))`, which is a different algorithm."
        )


def rprop_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    learning_rate: float = 1e-2,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    step_min: float = 1e-6,
    step_max: float = 50.0,
    namespace: str = "rprop",
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
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Scalar loss to differentiate, precomputed gradients, or the updates dict an earlier transform in
        a chain produced.
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

    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate state
        rather than reusing each other's. Default is the rule's own name.

    Returns
    -------
    updates : Updates
        Mapping from each parameter and its state buffers to their next values.

    Examples
    --------
    The update function behind :func:`rprop`, for compiling the step yourself rather
    than going through :func:`~pytensor_ml.optim.train.compile_train`. It returns the updates dict directly, stepping by gradient sign alone:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import rprop_updates
        from pytensor_ml.pytensorf import collect_trainable_params, function

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        updates = rprop_updates(loss, collect_trainable_params(loss), learning_rate=1e-2)
        step = function([X, target], loss, updates=updates)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    _require_numeric_learning_rate(learning_rate)

    incoming, gradients = gradients_to_descend(loss_gradients_or_updates, parameters, namespace)

    updates: Updates = Steps(incoming)
    for parameter, gradient in zip(parameters, gradients):
        previous_gradient = state_for(parameter, f"{namespace}/previous_gradient")
        step_size = state_for(parameter, f"{namespace}/step_size", fill_value=learning_rate)

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


def lbfgs_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    learning_rate: LearningRate = 1.0,
    memory_size: int = 10,
    scale_init_precond: bool = True,
    line_search: LineSearch | None = None,
    namespace: str = "lbfgs",
) -> Updates:
    r"""
    L-BFGS: descend along the gradient multiplied by a limited-memory inverse-Hessian approximation.

    The approximation is built from the last ``memory_size`` accepted pairs of parameter differences
    :math:`s = p_{k+1} - p_k` and gradient differences :math:`y = g_{k+1} - g_k`, applied to the gradient
    by the two-loop recursion of :class:`~pytensor_ml.optim.lbfgs.LBFGSDirection` starting from
    :math:`\gamma I`, with :math:`\gamma = s^\top y / y^\top y` for the newest pair. A pair enters the
    memory only when :math:`y^\top s > \epsilon\, y^\top y`, which keeps the approximation positive
    definite, so a step through a non-convex region leaves the memory as it was. Before any pair is
    accepted :math:`\gamma = \min(1, 1 / \|g\|)`, which keeps the first step inside the unit ball. The
    step is :math:`p \leftarrow p - \eta H g`.

    The direction is well scaled once the memory holds a pair, so :math:`\eta = 1` is the natural rate.
    Given a ``line_search``, the rule searches along :math:`-\eta H g` for a step size :math:`t` and
    moves to :math:`p - t \eta H g`, so ``learning_rate`` is the first trial step. Without one it takes
    every step at the rate, which is then the only safeguard against a bad direction. Consecutive
    gradients have to be measured on the same objective for their difference to be curvature, and a
    search compares the loss at several points, so the rule assumes a deterministic, full-batch loss.

    Three uses break that assumption. A gradient transform ahead of the rule in a chain, such as
    :func:`~pytensor_ml.optim.clipping.clip_by_global_norm`, hands it gradients whose differences are not
    curvature, and the rule can diverge. With a line search the rule refuses that input outright. Clip
    after the rule instead: the parameter differences are read off the parameters, so a clipped step
    still forms a valid pair. Wrapping the rule in :func:`~pytensor_ml.optim.guards.skip_if` does not
    rescue a bad step either, because the loss is deterministic and a skipped step is recomputed
    unchanged on the next call until the guard raises. A step that a guard would skip calls for a line
    search, which shortens it, or a smaller rate. Finally, a parameter written between steps,
    with ``set_value`` for instance, forms a pair from a move the rule did not make, and that pair stays
    in the memory for up to ``memory_size`` steps.

    With a search, the rule keeps the loss and gradients at the point it moved to, and the next step reuses
    those gradients when its parameters equal what was kept and its loss agrees with it to within 1024
    machine epsilons, relative to the kept loss. A step therefore runs one forward pass and one gradient
    per trial. A clip after the rule, a ``set_value``, new data or a changed shared variable fails that
    comparison and costs one more gradient, but a change that moves the loss by less than that tolerance
    passes it. On mlx both sides of that choice run, so the gradients at the current point are computed
    on every step.

    Parameters
    ----------
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Scalar loss to differentiate, precomputed gradients, or the updates dict an earlier transform in
        a chain produced.
    parameters : sequence of shared tensor variable
        Parameters to update.
    learning_rate : float or shared tensor variable
        Step size :math:`\eta`. Default 1.0.
    memory_size : int
        Number of pairs the memory holds. Default 10.
    scale_init_precond : bool
        Start the recursion from :math:`\gamma I` as above. When False it starts from the identity, and
        the first step is the raw gradient. Default True.
    line_search : LineSearch, optional
        Search for a step size along each step, with ``learning_rate`` as the first trial, for example
        :func:`~pytensor_ml.optim.zoom_line_search`. The search evaluates the loss, so the first argument
        has to be the loss itself. Its step size multiplies the step already scaled by ``learning_rate``,
        so a bound such as ``max_learning_rate`` caps that multiplier. The rule records the last
        multiplier in ``{namespace}/line_search/step_size``, the last search's trial count in
        ``{namespace}/line_search/evaluations``, and the number of searches that have failed so far in
        ``{namespace}/line_search/failures``. The trial it keeps is in the ``evaluated``, ``value``,
        ``point`` and ``gradient`` slots of ``{namespace}/line_search``. Default None, which takes every
        step at ``learning_rate``. The :func:`~pytensor_ml.optim.lbfgs` alias searches by default.
    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate state
        rather than reusing each other's. Default is the rule's own name.

    Returns
    -------
    updates : Updates
        Mapping from each parameter, its memory buffers and, with a search, the search's records to
        their next values.

    Examples
    --------
    Compile the step yourself rather than going through :func:`~pytensor_ml.optim.train.compile_train`.
    The rule returns the updates dict directly, and takes the fixed step unless given a search:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import lbfgs_updates
        from pytensor_ml.pytensorf import collect_trainable_params, function

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        updates = lbfgs_updates(loss, collect_trainable_params(loss), learning_rate=0.5)
        step = function([X, target], loss, updates=updates)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    if memory_size < 1:
        raise ValueError(f"memory_size must be at least 1, got {memory_size}.")
    loss = (
        loss_gradients_or_updates if isinstance(loss_gradients_or_updates, TensorVariable) else None
    )
    if line_search is not None and loss is None:
        raise ValueError(
            f"{namespace} searches along its step by evaluating the loss, so with a line search it needs "
            f"the loss graph rather than gradients or an updates dict. It therefore has to come first in "
            f"a chain, where the loss still reaches it."
        )

    incoming, gradients = gradients_to_descend(loss_gradients_or_updates, parameters, namespace)
    step_count = step_counter(f"{namespace}/step_count")
    learning_rate = to_floatx(rate_on(learning_rate, step_count))

    loss_gradients = gradients
    if line_search is not None and loss is not None:
        kept = _allocate_kept_trial(loss, parameters, namespace)
        gradients = _reuse_kept_gradients(kept, loss, parameters, loss_gradients)

    pairs_written = scalar_state(f"{namespace}/pairs_written", dtype="int64")
    previous_values = [state_for(p, f"{namespace}/previous_value") for p in parameters]
    previous_gradients = [state_for(p, f"{namespace}/previous_gradient") for p in parameters]
    value_memory = [
        state_for(p, f"{namespace}/value_differences", history_size=memory_size) for p in parameters
    ]
    gradient_memory = [
        state_for(p, f"{namespace}/gradient_differences", history_size=memory_size)
        for p in parameters
    ]
    # One curvature per slot, at the dtype of the cross-parameter dot that measures it. A slot that holds
    # no pair keeps a zero, which the recursion reads as a pair that contributes nothing.
    curvatures = pytensor.shared(
        np.zeros(memory_size, dtype=upcast(*(gradient.dtype for gradient in gradients))),
        name=f"{namespace}/curvatures",
        shape=(memory_size,),
    )

    # The buffers hold zeros before the first step, so the differences read off them are meaningless
    # until a previous point exists; the guard below never lets those into the memory.
    value_differences = [p - previous for p, previous in zip(parameters, previous_values)]
    gradient_differences = [g - previous for g, previous in zip(gradients, previous_gradients)]
    curvature = flat_dot(gradient_differences, value_differences)
    gradient_change = flat_dot(gradient_differences, gradient_differences)
    epsilon = max(np.finfo(gradient.dtype).eps for gradient in gradients)
    # Near a minimum y . y underflows to zero, so the relative test alone admits a pair whose inverse
    # curvature or identity scale overflows; both have to be representable to enter the memory.
    accept = (
        (step_count > 0)
        & (curvature > epsilon * gradient_change)
        & pt.isfinite(pt.reciprocal(curvature))
        & pt.isfinite(curvature / gradient_change)
    )

    # Rejection rewrites the slot with itself, so the write stays in place and unconditional; only the
    # count decides whether the slot is now part of the memory. The slot is a one-element index vector
    # rather than a scalar because mlx cannot trace a scalar index (pymc-devs/pytensor#2422).
    slot = (pairs_written % memory_size)[None]
    new_value_memory = [
        pt.set_subtensor(memory[slot], pt.switch(accept, s[None], memory[slot]))
        for memory, s in zip(value_memory, value_differences)
    ]
    new_gradient_memory = [
        pt.set_subtensor(memory[slot], pt.switch(accept, y[None], memory[slot]))
        for memory, y in zip(gradient_memory, gradient_differences)
    ]
    # Stored inverted, from the very dot the guard tested, so an admitted pair's curvature is positive by
    # construction. The divisor is swapped out on rejection so a zero curvature never reaches it.
    rho = pt.reciprocal(pt.switch(accept, curvature, 1.0)).astype(curvatures.dtype)
    new_curvatures = pt.set_subtensor(
        curvatures[slot], pt.switch(accept, rho[None], curvatures[slot])
    )
    new_pairs_written = pairs_written + accept.astype(pairs_written.dtype)

    updates: Updates = Steps(incoming)
    if scale_init_precond:
        # The newest admitted pair's s . y / y . y, carried from the step that admitted it, since a
        # rejected step leaves the newest pair in memory unchanged.
        newest_pair_scale = scalar_state(f"{namespace}/identity_scale", dtype=curvatures.dtype)
        new_newest_pair_scale = pt.switch(
            accept,
            curvature / pt.switch(accept, gradient_change, 1.0),
            newest_pair_scale,
        ).astype(newest_pair_scale.dtype)
        gradient_norm = pt.sqrt(flat_dot(gradients, gradients))
        identity_scale = pt.switch(
            new_pairs_written > 0,
            new_newest_pair_scale,
            pt.minimum(1.0, 1.0 / pt.switch(gradient_norm > 0, gradient_norm, 1.0)),
        )
        updates[newest_pair_scale] = new_newest_pair_scale
    else:
        identity_scale = 1.0

    directions = LBFGSDirection(n_parameters=len(parameters), memory_size=memory_size)(
        new_pairs_written,
        identity_scale,
        new_curvatures,
        *gradients,
        *new_value_memory,
        *new_gradient_memory,
        return_list=True,
    )

    updates[step_count] = step_count + 1
    updates[pairs_written] = new_pairs_written
    updates[curvatures] = new_curvatures
    for index, parameter in enumerate(parameters):
        updates[previous_values[index]] = parameter
        updates[previous_gradients[index]] = gradients[index]
        updates[value_memory[index]] = new_value_memory[index]
        updates[gradient_memory[index]] = new_gradient_memory[index]

    steps = [
        (-learning_rate * direction).astype(parameter.dtype)
        for parameter, direction in zip(parameters, directions)
    ]
    if line_search is None or loss is None:
        for parameter, step in zip(parameters, steps):
            updates[parameter] = parameter + step
        return updates

    search = search_along(
        loss,
        parameters,
        loss_gradients,
        steps,
        line_search,
        start_gradients=gradients,
    )
    for parameter, step in zip(parameters, steps):
        updates[parameter] = parameter + search.step_size.astype(parameter.dtype) * step

    _keep_last_trial(updates, kept, search, parameters)

    step_size = scalar_state(f"{namespace}/line_search/step_size", fill_value=1.0)
    evaluations = scalar_state(f"{namespace}/line_search/evaluations", dtype="int64")
    failures = scalar_state(f"{namespace}/line_search/failures", dtype="int64")
    updates[step_size] = search.step_size.astype(step_size.dtype)
    updates[evaluations] = search.evaluations.astype(evaluations.dtype)
    updates[failures] = failures + search.failed.astype(failures.dtype)

    return updates


_KEPT_LOSS_TOLERANCE_IN_EPSILONS = 1024


class _KeptTrial(NamedTuple):
    """The point a search last moved to, with the loss and gradients measured there."""

    evaluated: Parameter
    value: Parameter
    points: list[Parameter]
    gradients: list[Parameter]


def _allocate_kept_trial(
    loss: TensorVariable, parameters: Sequence[Parameter], namespace: str
) -> _KeptTrial:
    return _KeptTrial(
        evaluated=scalar_state(f"{namespace}/line_search/evaluated", dtype="bool"),
        value=scalar_state(f"{namespace}/line_search/value", dtype=loss.dtype),
        points=[state_for(p, f"{namespace}/line_search/point") for p in parameters],
        gradients=[state_for(p, f"{namespace}/line_search/gradient") for p in parameters],
    )


def _reuse_kept_gradients(
    kept: _KeptTrial,
    loss: TensorVariable,
    parameters: Sequence[Parameter],
    gradients: Sequence[TensorVariable],
) -> list[TensorVariable]:
    # Equal parameters are not enough: the loss also reads data and shared variables, and comparing the
    # loss is what lets a changed objective, or a NaN batch, reach the search and a guard. The kept value
    # comes out of the search's own compiled graph, so it agrees with `loss` only up to rounding.
    tolerance = _KEPT_LOSS_TOLERANCE_IN_EPSILONS * np.finfo(loss.dtype).eps
    reusable = kept.evaluated & pt.isclose(loss, kept.value, rtol=tolerance, atol=0.0)
    for parameter, point in zip(parameters, kept.points):
        reusable = reusable & pt.all(pt.eq(parameter, point))
    reused: list[TensorVariable] = [
        old.astype(fresh.dtype) for old, fresh in zip(kept.gradients, gradients)
    ]
    return ifelse(reusable, reused, list(gradients))  # type: ignore[return-value]


def _keep_last_trial(
    updates: Updates,
    kept: _KeptTrial,
    search: SearchResult,
    parameters: Sequence[Parameter],
) -> None:
    # A search that fell back moved short of its last trial, so what is kept stays as it was. It still
    # describes the point it was measured at, which the next step compares against.
    took_last_trial = pt.eq(search.step_size, search.trial_step)
    updates[kept.evaluated] = kept.evaluated | took_last_trial
    updates[kept.value] = pt.switch(took_last_trial, search.trial_value, kept.value).astype(
        kept.value.dtype
    )
    for parameter, point, gradient, trial_gradient in zip(
        parameters, kept.points, kept.gradients, search.trial_gradients
    ):
        updates[point] = pt.switch(took_last_trial, updates[parameter], point)
        updates[gradient] = pt.switch(took_last_trial, trial_gradient, gradient).astype(
            gradient.dtype
        )
