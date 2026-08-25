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
    """

    def transform(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        next_updates = dict(updates)
        for parameter in parameters:
            next_updates[parameter] = parameter + pt.clip(
                updates[parameter] - parameter, min_value, max_value
            )
        return next_updates

    return transform
