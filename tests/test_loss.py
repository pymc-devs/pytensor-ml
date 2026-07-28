import numpy as np
import pytensor.tensor as pt
import pytest

from scipy.special import softmax
from sklearn.metrics import log_loss

from pytensor_ml.loss import CrossEntropy, Reductions, SquaredError


def generate_categorical_data(expect_logits: bool, seed: int = 0):
    # Seeded: sklearn is an exact oracle here, so extra random draws buy little, and an unreproducible
    # numerical failure costs a lot. Logits are drawn from a normal so the log-softmax path sees negative
    # values, which a uniform draw never produced.
    rng = np.random.default_rng(seed)
    n_classes = rng.integers(2, 10)
    y_true = rng.integers(0, n_classes, size=(100,))
    y_true_onehot = np.eye(n_classes)[y_true]
    y_pred = (
        rng.normal(size=(100, n_classes))
        if expect_logits
        else rng.dirichlet(np.ones(n_classes), size=(100,))
    )

    return y_true, y_true_onehot, y_pred


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("expect_logits", [True, False])
@pytest.mark.parametrize("expect_onehot_labels", [True, False])
def test_cross_entropy(reduction: Reductions, expect_logits, expect_onehot_labels):
    loss = CrossEntropy(
        reduction=reduction, expect_logits=expect_logits, expect_onehot_labels=expect_onehot_labels
    )

    y_true, y_true_onehot, y_pred = generate_categorical_data(expect_logits)

    if expect_onehot_labels:
        loss_value = loss(y_true_onehot, y_pred).eval()
    else:
        loss_value = loss(y_true, y_pred).eval()

    if expect_logits:
        y_pred = softmax(y_pred, axis=-1)

    sklearn_loss = log_loss(y_true, y_pred, normalize=reduction == "mean")
    np.testing.assert_allclose(loss_value, sklearn_loss)


@pytest.mark.parametrize(
    "reduction, expected", [("mean", 4.25 / 3), ("sum", 4.25), (lambda x: x, [0.25, 0.0, 4.0])]
)
def test_squared_error(reduction, expected):
    y_true = pt.as_tensor([1.0, 2.0, 3.0])
    y_pred = pt.as_tensor([1.5, 2.0, 1.0])

    loss_value = SquaredError(reduction=reduction)(y_true, y_pred).eval()

    np.testing.assert_allclose(loss_value, expected)


def test_cross_entropy_accepts_a_callable_reduction():
    y_true, _, y_pred = generate_categorical_data(expect_logits=True)

    per_sample = CrossEntropy(expect_logits=True, reduction=lambda x: x)(y_true, y_pred).eval()
    pooled = CrossEntropy(expect_logits=True, reduction="mean")(y_true, y_pred).eval()

    assert per_sample.shape == y_true.shape
    np.testing.assert_allclose(per_sample.mean(), pooled)
