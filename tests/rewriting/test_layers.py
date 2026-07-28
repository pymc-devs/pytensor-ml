import pytensor.tensor as pt
import pytest

from pytensor.graph import graph_inputs
from pytensor.graph.fg import FunctionGraph

from pytensor_ml.layers import (
    BatchNorm2D,
    BatchNormLayer,
    Dropout,
    DropoutLayer,
    Linear,
    PredictionBatchNormLayer,
    Sequential,
)
from pytensor_ml.pytensorf import rewrite_for_prediction


@pytest.fixture()
def feature_extractor_and_rng():
    d1 = Dropout("Dropout_1", p=0.5)
    d2 = Dropout("Dropout_2", p=0.5)
    feature_extractor = Sequential(
        Linear("Layer_1", n_in=6, n_out=3), d1, Linear("Layer_2", n_in=3, n_out=1), d2
    )

    rngs = [d1.rng, d2.rng]

    return feature_extractor, rngs


def test_remove_dropout(feature_extractor_and_rng):
    feature_extractor, rngs = feature_extractor_and_rng

    X = pt.tensor("X", shape=(None, 6))
    latent = feature_extractor(X)

    fg = FunctionGraph(inputs=list(graph_inputs([latent])) + rngs, outputs=[latent])

    assert len([node.op for node in fg.apply_nodes if isinstance(node.op, DropoutLayer)]) == 2
    fg = rewrite_for_prediction(fg)

    assert len([node.op for node in fg.apply_nodes if isinstance(node.op, DropoutLayer)]) == 0


@pytest.mark.parametrize(
    "consumed_downstream", [False, True], ids=["as_graph_output", "consumed_by_a_layer"]
)
def test_rewrite_batch_stats_to_running_average_stats(consumed_downstream):
    layers = [Linear("Layer_1", n_in=6, n_out=3), BatchNorm2D()]
    if consumed_downstream:
        # A downstream SymbolicOp type-checks its inputs more strictly than a graph output does.
        layers.append(Linear("Layer_2", n_in=3, n_out=1))

    feature_extractor = Sequential(*layers)
    X = pt.tensor("X", shape=(None, 6))
    latent = feature_extractor(X)

    fg = FunctionGraph(inputs=list(graph_inputs([latent])), outputs=[latent])
    assert len([node.op for node in fg.apply_nodes if isinstance(node.op, BatchNormLayer)]) == 1

    fg = rewrite_for_prediction(fg)

    assert not any(isinstance(node.op, BatchNormLayer) for node in fg.apply_nodes)
    assert (
        len([node.op for node in fg.apply_nodes if isinstance(node.op, PredictionBatchNormLayer)])
        == 1
    )
