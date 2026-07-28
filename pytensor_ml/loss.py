from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal

import pytensor.tensor as pt

from pytensor.tensor.basic import as_tensor_variable

Reductions = Literal["mean", "sum"]
ReductionFunction = Callable[[pt.TensorVariable], pt.TensorVariable]

# A callable covers the unreduced case (``reduction=lambda x: x``), for weighting individual losses.
ReductionLike = Reductions | ReductionFunction

_REDUCTIONS: dict[Reductions, ReductionFunction] = {"mean": pt.mean, "sum": pt.sum}


def _as_reduction(reduction: ReductionLike) -> ReductionFunction:
    return reduction if callable(reduction) else _REDUCTIONS[reduction]


class Loss(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred) -> pt.TensorVariable: ...

    def __call__(self, y_true, y_pred) -> pt.TensorVariable:
        return self.loss(y_true, y_pred)


class SquaredError(Loss):
    def __init__(self, reduction: ReductionLike = "mean"):
        self.reduction = _as_reduction(reduction)

    def loss(self, y_true, y_pred) -> pt.TensorVariable:
        y_true = as_tensor_variable(y_true)
        y_pred = as_tensor_variable(y_pred)
        return self.reduction((y_true - y_pred) ** 2)


class CrossEntropy(Loss):
    def __init__(
        self,
        reduction: ReductionLike = "mean",
        expect_logits: bool = False,
        expect_onehot_labels: bool = False,
    ):
        self.reduction = _as_reduction(reduction)
        self.expect_logits = expect_logits
        self.expect_onehot_labels = expect_onehot_labels

    def loss(self, y_true: pt.TensorVariable, y_pred: pt.TensorVariable) -> pt.TensorVariable:
        """
        Parameters
        ----------
        y_true : TensorVariable
            Ground-truth class membership: a matrix of one-hot rows when ``expect_onehot_labels`` is set,
            otherwise a vector of integer class labels.
        y_pred : TensorVariable
            Predicted class membership, as unnormalized logits when ``expect_logits`` is set, otherwise as
            probabilities.

        Returns
        -------
        loss : TensorVariable
            The reduced loss -- a scalar under the named reductions, or whatever shape a callable
            ``reduction`` leaves.
        """
        y_true = as_tensor_variable(y_true)
        y_pred = as_tensor_variable(y_pred)
        if self.expect_logits:
            log_softmax = pt.special.log_softmax(y_pred, axis=-1)
        else:
            log_softmax = pt.log(y_pred)

        if not self.expect_onehot_labels:
            log_softmax = pt.take_along_axis(log_softmax, y_true[..., None], axis=-1)[..., 0]
            return -self.reduction(log_softmax)

        return -self.reduction((y_true * log_softmax).sum(axis=-1))


def supervised_loss(
    prediction: pt.TensorVariable,
    loss_fn: Loss,
    ndim_out: int = 1,
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """
    Build a training loss and its target placeholder from a model prediction.

    The target is a fresh input variable shaped like the labelled slice of ``prediction``: its first
    ``ndim_out`` dimensions match ``prediction`` and any trailing dimensions are dropped. For example, a
    ``(batch, classes)`` logit prediction with ``ndim_out=2`` yields a ``(batch, classes)`` target.

    Parameters
    ----------
    prediction : TensorVariable
        Model output to compare against the target.
    loss_fn : Loss
        Callable ``(target, prediction) -> scalar loss``.
    ndim_out : int
        Number of leading prediction dimensions the target shares. Default 1.

    Returns
    -------
    loss : TensorVariable
        Scalar training loss.
    target : TensorVariable
        Input placeholder for the ground-truth labels, to be supplied at call time.
    """
    label_slice = (slice(None),) * ndim_out + (0,) * (prediction.ndim - ndim_out)
    target = prediction[label_slice].type()
    target.name = "target"
    return loss_fn(target, prediction), target
