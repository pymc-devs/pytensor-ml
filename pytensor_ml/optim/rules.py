from collections.abc import Callable, Sequence

import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Variable
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

    The direction is well scaled once the memory holds a pair, so :math:`\eta = 1` is the natural rate
    and a line search the natural way to back off from it. Consecutive gradients have to be measured on
    the same objective for their difference to be curvature, so the rule assumes a deterministic,
    full-batch loss.

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

    namespace : str
        Prefix for every state slot this rule allocates, so two rules in one graph keep separate state
        rather than reusing each other's. Default is the rule's own name.

    Returns
    -------
    updates : Updates
        Mapping from each parameter and its memory buffers to their next values.

    Examples
    --------
    Compile the step yourself rather than going through :func:`~pytensor_ml.optim.train.compile_train`.
    The rule returns the updates dict directly, with no line search:

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

    incoming, gradients = gradients_to_descend(loss_gradients_or_updates, parameters, namespace)
    step_count = step_counter(f"{namespace}/step_count")
    learning_rate = to_floatx(rate_on(learning_rate, step_count))

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

    # The buffers hold zeros before the first step, so the differences read off them are meaningless
    # until a previous point exists; the guard below never lets those into the memory.
    value_differences = [p - previous for p, previous in zip(parameters, previous_values)]
    gradient_differences = [g - previous for g, previous in zip(gradients, previous_gradients)]
    curvature = flat_dot(gradient_differences, value_differences)
    gradient_change = flat_dot(gradient_differences, gradient_differences)
    epsilon = max(np.finfo(gradient.dtype).eps for gradient in gradients)
    accept = (step_count > 0) & (curvature > epsilon * gradient_change)

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
    new_pairs_written = pairs_written + accept.astype(pairs_written.dtype)

    if scale_init_precond:
        has_pair = new_pairs_written > 0
        newest = ((new_pairs_written - 1) % memory_size)[None]
        newest_s = [memory[newest] for memory in new_value_memory]
        newest_y = [memory[newest] for memory in new_gradient_memory]
        newest_curvature = flat_dot(newest_s, newest_y)
        newest_change = pt.switch(has_pair, flat_dot(newest_y, newest_y), 1.0)
        gradient_norm = pt.sqrt(flat_dot(gradients, gradients))
        identity_scale = pt.switch(
            has_pair,
            newest_curvature / newest_change,
            pt.minimum(1.0, 1.0 / pt.switch(gradient_norm > 0, gradient_norm, 1.0)),
        )
    else:
        identity_scale = 1.0

    directions = LBFGSDirection(n_parameters=len(parameters), memory_size=memory_size)(
        new_pairs_written,
        identity_scale,
        *gradients,
        *new_value_memory,
        *new_gradient_memory,
        return_list=True,
    )

    updates: Updates = Steps(incoming)
    updates[step_count] = step_count + 1
    updates[pairs_written] = new_pairs_written
    for index, parameter in enumerate(parameters):
        updates[previous_values[index]] = parameter
        updates[previous_gradients[index]] = gradients[index]
        updates[value_memory[index]] = new_value_memory[index]
        updates[gradient_memory[index]] = new_gradient_memory[index]
        updates[parameter] = parameter - learning_rate * directions[index]

    return updates
