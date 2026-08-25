from collections.abc import Sequence

import pytensor.tensor as pt

from pytensor_ml.optim.base import Parameter, Transform, Updates


def clip_by_global_norm(max_norm: float = 1.0) -> Transform:
    r"""
    Rescale all steps by a single factor so their global L2 norm does not exceed ``max_norm``.

    With :math:`\|s\|` the norm of the concatenated steps, every step is multiplied by
    :math:`\min(1, \text{max\_norm} / (\|s\| + \epsilon))`, preserving the update direction while bounding its
    magnitude.

    Parameters
    ----------
    max_norm : float
        Maximum allowed global norm. Default 1.0.

    Returns
    -------
    transform : Transform
        A transform that clips the updates dict by global norm.

    Examples
    --------
    Bound the whole update rather than each coordinate, so the direction of the step survives and only
    its magnitude is capped:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, chain, clip_by_global_norm, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError(), ndim_out=2)

        step = compile_train(loss, chain(adam(1e-3), clip_by_global_norm(1.0)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        steps = [updates[parameter] - parameter for parameter in parameters]
        global_norm = pt.sqrt(sum(pt.sum(step**2) for step in steps))
        clip_scale = pt.minimum(1.0, max_norm / (global_norm + 1e-8))
        next_updates = dict(updates)
        for parameter, step in zip(parameters, steps):
            next_updates[parameter] = parameter + clip_scale * step
        return next_updates

    return transform


def clip_by_value(min_value: float = -1.0, max_value: float = 1.0) -> Transform:
    """
    Clamp each step element-wise into ``[min_value, max_value]``.

    Parameters
    ----------
    min_value : float
        Lower bound. Default -1.0.
    max_value : float
        Upper bound. Default 1.0.

    Returns
    -------
    transform : Transform
        A transform that clips the updates dict element-wise.

    Examples
    --------
    Clip each coordinate on its own, which bounds the step but tilts its direction whenever only some
    coordinates are clipped:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, chain, clip_by_value, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError(), ndim_out=2)

        step = compile_train(loss, chain(adam(1e-3), clip_by_value(-0.1, 0.1)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        next_updates = dict(updates)
        for parameter in parameters:
            next_updates[parameter] = parameter + pt.clip(
                updates[parameter] - parameter, min_value, max_value
            )
        return next_updates

    return transform
