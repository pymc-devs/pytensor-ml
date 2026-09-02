import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.graph.basic import equal_computations

from pytensor_ml.layers import BatchNorm, Dropout, Linear, PredictionBatchNormLayer, Sequential
from pytensor_ml.layers.dropout import DropoutLayer
from pytensor_ml.pytensorf import function, rewrite_for_prediction

floatX = pytensor.config.floatX


def dropout_layers_in(compiled):
    return [node for node in compiled.maker.fgraph.apply_nodes if isinstance(node.op, DropoutLayer)]


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


def test_a_dropout_that_keeps_everything_is_dropped_at_compile_time():
    """Nothing is sampled at ``p=0``, but the graph still records the layer the user asked for, so the
    saving is the compiler's to make."""
    X = pt.tensor("X", shape=(None, 6))
    built = Dropout("d", p=0.0, random_state=0)(X)

    assert any(
        isinstance(node.op, DropoutLayer)
        for node in pytensor.graph.FunctionGraph(outputs=[built], clone=False).apply_nodes
    )

    compiled = pytensor.function([X], built)
    X_np = np.random.default_rng(0).normal(size=(4, 6)).astype(floatX)

    assert dropout_layers_in(compiled) == []
    np.testing.assert_allclose(compiled(X_np), X_np)


def test_a_dropout_that_keeps_nothing_is_dropped_at_compile_time():
    """Keeping nothing scales the survivors by 1/0, which pytensor evaluates while canonicalizing even
    though the branch it sits on is never selected."""
    X = pt.tensor("X", shape=(None, 6))
    built = Dropout("d", p=1.0, random_state=0)(X)

    assert any(
        isinstance(node.op, DropoutLayer)
        for node in pytensor.graph.FunctionGraph(outputs=[built], clone=False).apply_nodes
    )

    compiled = pytensor.function([X], built)
    X_np = np.random.default_rng(0).normal(size=(4, 6)).astype(floatX)

    assert dropout_layers_in(compiled) == []
    np.testing.assert_allclose(compiled(X_np), np.zeros_like(X_np))


def test_an_ordinary_dropout_survives_compilation():
    """The rewrite reads the probability off the op, so being too eager here would silently switch
    dropout off in training."""
    X = pt.tensor("X", shape=(None, 6))
    compiled = pytensor.function([X], Dropout("d", p=0.5, random_state=0)(X))

    assert len(dropout_layers_in(compiled)) == 1


@pytest.mark.parametrize("p", [0.0, 1.0], ids=["keeps_everything", "keeps_nothing"])
def test_a_degenerate_dropout_is_dropped_through_the_project_compile_path(p):
    """:func:`~pytensor_ml.pytensorf.function` compiles with a mode of its own, which has to keep
    carrying pytensor's specializations or a rewrite registered into them stops reaching real models."""
    X = pt.tensor("X", shape=(None, 6))

    compiled = function([X], Dropout("d", p=p, random_state=0)(X))

    assert dropout_layers_in(compiled) == []
