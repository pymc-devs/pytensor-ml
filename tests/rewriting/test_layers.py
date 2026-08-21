import pytensor.tensor as pt
import pytest

from pytensor.graph.basic import equal_computations

from pytensor_ml.layers import BatchNorm, Dropout, Linear, PredictionBatchNormLayer, Sequential
from pytensor_ml.pytensorf import rewrite_for_prediction


def test_remove_dropout():
    first_layer = Linear("Layer_1", n_in=6, n_out=3)
    second_layer = Linear("Layer_2", n_in=3, n_out=1)
    X = pt.tensor("X", shape=(None, 6))

    with_dropout = Sequential(
        first_layer, Dropout("Dropout_1", p=0.5), second_layer, Dropout("Dropout_2", p=0.5)
    )(X)
    without_dropout = Sequential(first_layer, second_layer)(X)

    assert equal_computations([rewrite_for_prediction(with_dropout)], [without_dropout])


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no_affine"])
@pytest.mark.parametrize(
    "consumed_downstream", [False, True], ids=["as_graph_output", "consumed_by_a_layer"]
)
def test_rewrite_batch_stats_to_running_average_stats(consumed_downstream, affine):
    linear = Linear("Layer_1", n_in=6, n_out=3)
    batch_norm = BatchNorm("bn", n_in=3, affine=affine)
    # A downstream SymbolicOp type-checks its inputs more strictly than a graph output does.
    head = Linear("Layer_2", n_in=3, n_out=1) if consumed_downstream else None
    X = pt.tensor("X", shape=(None, 6))

    normalized = batch_norm(linear(X))
    trained = head(normalized) if head else normalized

    affine_params = [batch_norm.loc, batch_norm.scale] if affine else []
    prediction_normalized = PredictionBatchNormLayer(
        name="bn", n_in=3, epsilon=batch_norm.epsilon, affine=affine
    )(linear(X), *affine_params, batch_norm.running_mean, batch_norm.running_var)
    expected = head(prediction_normalized) if head else prediction_normalized

    assert equal_computations([rewrite_for_prediction(trained)], [expected])


def test_rewrite_leaves_batch_norm_without_running_stats_alone():
    X = pt.tensor("X", shape=(None, 6))
    normalized = Sequential(
        Linear("Layer_1", n_in=6, n_out=3), BatchNorm("bn", track_running_stats=False)
    )(X)

    assert equal_computations([rewrite_for_prediction(normalized)], [normalized])
