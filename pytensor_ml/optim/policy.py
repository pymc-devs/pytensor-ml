from collections.abc import Sequence

import numpy as np
import pytensor.tensor as pt

from pytensor_ml.optim.base import (
    LossOrGradients,
    Parameter,
    UpdateRule,
    Updates,
    reuses_state,
    scalar_state,
)


def reduce_on_plateau(
    rule: UpdateRule,
    scale: Parameter,
    *,
    factor: float = 0.1,
    patience: int = 10,
    cooldown: int = 0,
    rtol: float = 1e-4,
    atol: float = 0.0,
    min_scale: float = 0.0,
    accumulation_size: int = 1,
) -> UpdateRule:
    r"""
    Cut ``scale`` by ``factor`` once the loss has stopped improving.

    Wraps a rule whose rate is built from ``scale``, so the policy owns a multiplier rather than the rate
    itself and composes with any schedule:

    .. code-block:: python

        scale = scalar_state("plateau/scale", fill_value=1.0)
        rule = reduce_on_plateau(adam(learning_rate=scale * 1e-3), scale, patience=5)

    Unlike torch's, this runs once per training step rather than once per epoch on a validation metric, so
    the loss it sees is whatever expression the rule is given. A per-batch loss is a noisy signal for it;
    ``accumulation_size`` is the answer, deciding on the mean of a window rather than on one batch. Set it
    to the number of steps in an epoch to get torch's cadence, on a better estimate than a single batch.

    Parameters
    ----------
    rule : UpdateRule
        The rule to wrap. Its rate must be built from ``scale`` for the cuts to reach the step.
    scale : shared tensor variable
        The multiplier this policy owns. Nothing else may write it.
    factor : float
        Multiplier applied on a cut, in the open interval (0, 1). Default 0.1.
    patience : int
        Steps without improvement before a cut. Default 10.
    cooldown : int
        Steps to wait after a cut before counting again, during which no step counts as bad. Without one,
        the counter resets on the cut and immediately starts toward the next. Default 0.
    rtol : float
        Relative improvement needed to count, as ``loss < (1 - rtol) * best - atol``. Default 1e-4.
    atol : float
        Absolute improvement needed to count. Default 0.0.
    min_scale : float
        Floor the scale cannot go below. Without one a noisy loss cuts repeatedly and underflows to zero.
        Default 0.0.
    accumulation_size : int
        Losses to average before deciding anything. Nothing advances mid-window -- not the count, not the
        cooldown, not the best seen. Default 1, which decides on every step from that step's loss.

    Returns
    -------
    UpdateRule
        The wrapped rule, which also writes the scale and the policy's own history.
    """
    if not 0.0 < factor < 1.0:
        raise ValueError(f"factor must lie in (0, 1), got {factor}.")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError(f"rtol and atol must be non-negative, got rtol={rtol}, atol={atol}.")
    if rtol > 1.0:
        raise ValueError(f"rtol is a relative tolerance and must be at most 1.0, got {rtol}.")
    if patience < 1:
        raise ValueError(f"patience is a number of steps and must be at least 1, got {patience}.")
    if cooldown < 0:
        raise ValueError(f"cooldown is a number of steps and must be non-negative, got {cooldown}.")
    if accumulation_size < 1:
        raise ValueError(
            f"accumulation_size is a number of losses to average and must be at least 1, got "
            f"{accumulation_size}."
        )

    @reuses_state
    def policy(loss_or_gradients: LossOrGradients, parameters: Sequence[Parameter]) -> Updates:
        if isinstance(loss_or_gradients, list | tuple):
            raise ValueError(
                "reduce_on_plateau decides from the loss, so it needs the loss graph rather than "
                "precomputed gradients. Pass the scalar loss, or drop the policy."
            )

        loss = loss_or_gradients
        updates = dict(rule(loss, parameters))

        best_loss = scalar_state("plateau/best_loss", fill_value=np.inf)
        waited = scalar_state("plateau/wait")
        cooling = scalar_state("plateau/cooldown")
        observed = scalar_state("plateau/observed")
        mean_loss = scalar_state("plateau/mean_loss")

        # Everything below is gated on `deciding`, so a window that is still filling advances nothing. At the
        # default size of one the window is a single step and the mean is that step's loss.
        seen = observed + 1
        running_mean = (observed * mean_loss + loss) / seen
        deciding = seen >= accumulation_size

        improved = deciding & (running_mean < (1 - rtol) * best_loss - atol)
        counted = pt.where(deciding, pt.where(improved, 0.0, waited + 1), waited)

        # Cooling down zeroes the count rather than pausing it, so the steps immediately after a cut cannot
        # add up to the next one before the network has had a chance to respond to the rate it just got.
        in_cooldown = cooling > 0
        cutting = deciding & ~in_cooldown & (counted >= patience)
        next_cooling = pt.where(in_cooldown, cooling - 1, pt.where(cutting, cooldown, 0.0))

        updates[scale] = pt.where(cutting, pt.maximum(scale * factor, min_scale), scale).astype(
            scale.dtype
        )
        updates[best_loss] = pt.where(improved, running_mean, best_loss).astype(best_loss.dtype)
        updates[waited] = pt.where(deciding & (in_cooldown | cutting), 0.0, counted).astype(
            waited.dtype
        )
        updates[cooling] = pt.where(deciding, next_cooling, cooling).astype(cooling.dtype)
        updates[observed] = pt.where(deciding, 0.0, seen).astype(observed.dtype)
        updates[mean_loss] = pt.where(deciding, 0.0, running_mean).astype(mean_loss.dtype)

        return updates

    return policy
