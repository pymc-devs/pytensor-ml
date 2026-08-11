import importlib

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.graph.replace import vectorize_graph

import pytensor_ml.layers

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import BatchNorm2D, Dropout, Embedding, Input, LayerNorm, Linear, Sequential
from pytensor_ml.pytensorf import (
    collect_non_trainable_updates,
    collect_trainable_params,
    rewrite_for_prediction,
)
from pytensor_ml.state import CustomInitializer, NormalInitializer, initialize_params

floatX = pytensor.config.floatX

# The numpy references below sum in a different order than the graph, so the gap tracks the precision.
ATOL = 1e-6 if floatX == "float64" else 1e-5


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "pytensor_ml layers")))


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_linear_layer(bias, rng):
    X = pt.tensor("X", shape=(None, 6))
    linear = Linear(name="Linear_1", n_in=6, n_out=3, bias=bias)
    out = linear(X)

    X_in, *weights = out.owner.inputs
    [X_out] = out.owner.outputs

    assert out.owner.op.name == "Linear_1[(?,6) -> (?,3)]"

    expected_names = ["Linear_1_W", "Linear_1_b"] if bias else ["Linear_1_W"]
    assert [w.name for w in weights] == expected_names

    assert X_out.name == "Linear_1_output"

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    W_np = rng.normal(size=(6, 3)).astype(floatX)
    b_np = rng.normal(size=(3,)).astype(floatX)

    linear.W.set_value(W_np)
    if bias:
        linear.b.set_value(b_np)

    res = out.eval({X: X_np})
    expected = X_np @ W_np + b_np if bias else X_np @ W_np
    np.testing.assert_allclose(res, expected)


def test_sequential(rng):
    linear1 = Linear(name="Linear_1", n_in=6, n_out=3)
    linear2 = Linear(name="Linear_2", n_in=3, n_out=1)
    mlp = Sequential(linear1, linear2)

    X = pt.tensor("X", shape=(None, 6))
    out = mlp(X)
    assert out.type.shape == (None, 1)

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    W1_np = rng.normal(size=(6, 3)).astype(floatX)
    b1_np = rng.normal(size=(3,)).astype(floatX)
    W2_np = rng.normal(size=(3, 1)).astype(floatX)
    b2_np = rng.normal(size=(1,)).astype(floatX)

    linear1.W.set_value(W1_np)
    linear1.b.set_value(b1_np)
    linear2.W.set_value(W2_np)
    linear2.b.set_value(b2_np)

    f = pytensor.function([X], out)
    res = f(X_np)

    np.testing.assert_allclose(res, (X_np @ W1_np + b1_np) @ W2_np + b2_np)


def test_dropout(rng):
    X = pt.tensor("X", shape=(None, 6))
    dropout = Dropout(name="Dropout_1", p=1.0)
    out = dropout(X)

    X_np = rng.normal(size=(10, 6)).astype(floatX)

    res = out.eval({X: X_np})
    np.testing.assert_allclose(res, np.zeros_like(X_np))


def test_invalid_dropout_p_raises():
    with pytest.raises(
        ValueError, match=r"Dropout probability has to be between 0 and 1, but got -0\.1"
    ):
        Dropout(name=None, p=-0.1)

    with pytest.raises(
        ValueError, match=r"Dropout probability has to be between 0 and 1, but got 1\.1"
    ):
        Dropout(name=None, p=1.1)


def test_input_accepts_a_varying_dimension():
    """One compiled graph has to serve batches of different sizes, which is what an unknown dimension is
    for -- asserting the declared type alone would pass even if the batch axis were pinned."""
    X = Input("X", (None, 64))
    forward = pytensor.function([X], Linear("fc", 64, 8)(X))

    assert X.type.shape == (None, 64)
    assert forward(np.zeros((3, 64), dtype=floatX)).shape == (3, 8)
    assert forward(np.zeros((7, 64), dtype=floatX)).shape == (7, 8)


def test_embedding_forward(rng):
    n_embeddings, n_features = 10, 4
    embedding = Embedding("emb", n_embeddings, n_features)
    W_np = rng.normal(size=(n_embeddings, n_features)).astype(floatX)
    embedding.W.set_value(W_np)

    ids = Input("ids", (2, 3), dtype="int64")  # a batch of index rows
    out = embedding(ids)
    assert out.name == "emb_output"

    ids_np = np.array([[1, 2, 3], [4, 5, 6]])
    res = out.eval({ids: ids_np})
    np.testing.assert_allclose(res, W_np[ids_np])
    assert res.shape == (2, 3, n_features)


def test_embedding_table_is_trainable(rng):
    # The OpFromGraph marker must pass the gradient through to the selected rows -- and only
    # those rows -- so the table trains; the integer indices are non-differentiable.
    embedding = Embedding("emb", n_embeddings=6, n_features=3)
    embedding.W.set_value(rng.normal(size=(6, 3)).astype(floatX))
    ids = pt.lvector("ids")
    grad_fn = pytensor.function([ids], pytensor.grad((embedding(ids) ** 2).sum(), embedding.W))

    grad = grad_fn(np.array([1, 1, 4]))
    selected = np.zeros(6, dtype=bool)
    selected[[1, 4]] = True
    assert np.any(grad[selected] != 0)
    assert np.all(grad[~selected] == 0)


@pytest.mark.parametrize("n_in", [6, None], ids=["specified", "lazy"])
def test_batch_norm_2d_forward(n_in, rng):
    X = pt.tensor("X", shape=(None, 6))
    batch_norm = BatchNorm2D(name="BatchNorm_1", n_in=n_in)
    out = batch_norm(X)

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    scale_np = rng.normal(size=(6,)).astype(floatX) ** 2
    loc_np = rng.normal(size=(6,)).astype(floatX) ** 2
    batch_norm.scale.set_value(scale_np)
    batch_norm.loc.set_value(loc_np)

    res = out.eval({X: X_np})
    mean_np = X_np.mean(axis=0)
    var_np = X_np.var(axis=0)
    expected = (X_np - mean_np) / np.sqrt(var_np + batch_norm.epsilon) * scale_np + loc_np

    np.testing.assert_allclose(res, expected, rtol=1e-5)


# Rank > 2 is the transformer case (batch, seq, d_model): the last axis is normalized and the affine
# parameters broadcast over every leading dimension.
@pytest.mark.parametrize("batch_shape", [(10,), (2, 4)], ids=["2d", "3d"])
@pytest.mark.parametrize("n_in", [6, None], ids=["specified", "lazy"])
def test_layer_norm_forward(n_in, batch_shape, rng):
    X = pt.tensor("X", shape=(*(None,) * len(batch_shape), 6))
    layer_norm = LayerNorm(name="LayerNorm_1", n_in=n_in)
    out = layer_norm(X)
    assert out.name == "LayerNorm_1_output"

    X_np = rng.normal(size=(*batch_shape, 6)).astype(floatX)
    scale_np = rng.normal(size=(6,)).astype(floatX)
    loc_np = rng.normal(size=(6,)).astype(floatX)
    layer_norm.scale.set_value(scale_np)
    layer_norm.loc.set_value(loc_np)

    res = out.eval({X: X_np})
    mean_np = X_np.mean(axis=-1, keepdims=True)
    var_np = X_np.var(axis=-1, keepdims=True)
    expected = (X_np - mean_np) / np.sqrt(var_np + layer_norm.epsilon) * scale_np + loc_np
    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_layer_norm_prediction_matches_training(rng):
    # LayerNorm normalizes over per-sample statistics, identical in train and eval, so unlike
    # BatchNorm it needs no prediction rewrite: rewrite_for_prediction leaves its output unchanged.
    X = pt.tensor("X", shape=(None, 6))
    layer_norm = LayerNorm("ln", n_in=6)
    out = layer_norm(X)
    layer_norm.scale.set_value(rng.normal(size=6).astype(floatX))
    layer_norm.loc.set_value(rng.normal(size=6).astype(floatX))

    X_np = rng.normal(size=(10, 6)).astype(floatX)
    np.testing.assert_allclose(
        rewrite_for_prediction(out).eval({X: X_np}), out.eval({X: X_np}), rtol=1e-6
    )


def test_vectorize_graph_batches_independent_predictions(rng):
    # A model built for a single sample must vectorize over a batch through the OpFromGraph-based
    # layers (Linear, LayerNorm); the batched result must match looping the single-sample graph.
    x = pt.vector("x", shape=(4,))
    net = Sequential(Linear("fc1", 4, 8), ReLU(), LayerNorm("ln", n_in=8), Linear("fc2", 8, 3))
    out = net(x)
    for parameter in collect_trainable_params(out):
        parameter.set_value(rng.normal(size=parameter.get_value().shape).astype(floatX))

    X = pt.matrix("X", shape=(None, 4))
    f_single = pytensor.function([x], out)
    f_batch = pytensor.function([X], vectorize_graph(out, {x: X}))

    X_np = rng.normal(size=(5, 4)).astype(floatX)
    np.testing.assert_allclose(f_batch(X_np), np.stack([f_single(row) for row in X_np]), rtol=1e-5)


def test_layer_norm_no_affine_standardizes_each_row(rng):
    X = pt.tensor("X", shape=(None, 8))
    out = LayerNorm(name="LayerNorm_1", n_in=8, affine=False)(X)

    X_np = rng.normal(loc=3.0, scale=2.0, size=(10, 8)).astype(floatX)
    res = out.eval({X: X_np})

    np.testing.assert_allclose(res.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(res.var(axis=-1), 1.0, rtol=1e-3)


def test_batch_norm_2d_learns_population_stats(rng):
    population_mean, population_std = 3.2, 6.2
    X = pt.tensor("X", shape=(None, 32))
    batch_norm = BatchNorm2D(name="BatchNorm_1", n_in=32, momentum=0.05, epsilon=1e-8)
    X_normalized = batch_norm(X)

    loss = pt.square(X_normalized - X).mean()
    d_loss = pt.grad(loss, [batch_norm.loc, batch_norm.scale])

    learning_rate = 1e-1
    updates = {
        batch_norm.loc: batch_norm.loc - learning_rate * d_loss[0],
        batch_norm.scale: batch_norm.scale - learning_rate * d_loss[1],
        batch_norm.running_mean: batch_norm.new_running_mean,
        batch_norm.running_var: batch_norm.new_running_var,
    }
    train = pytensor.function([X], X_normalized, updates=updates)

    def sample_batch():
        return rng.normal(loc=population_mean, scale=population_std, size=(100, 32)).astype(floatX)

    for _ in range(500):
        data = sample_batch()
        # Read before stepping: the affine parameters this batch is normalized with are the ones the
        # updates are about to overwrite.
        scale, loc = batch_norm.scale.get_value(), batch_norm.loc.get_value()

        np.testing.assert_allclose(
            train(data),
            (data - data.mean(axis=0)) / np.sqrt(data.var(axis=0) + batch_norm.epsilon) * scale
            + loc,
            rtol=1e-4,
            atol=ATOL,
        )

    scale, loc = batch_norm.scale.get_value(), batch_norm.loc.get_value()
    running_mean = batch_norm.running_mean.get_value()
    running_var = batch_norm.running_var.get_value()

    np.testing.assert_allclose(loc, population_mean, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(scale, population_std, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(running_mean, population_mean, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(np.sqrt(running_var), population_std, rtol=1e-1, atol=1e-1)

    predict = pytensor.function([X], rewrite_for_prediction(X_normalized))
    data = sample_batch()

    np.testing.assert_allclose(
        predict(data),
        (data - running_mean) / np.sqrt(running_var + batch_norm.epsilon) * scale + loc,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "op_name, submodule",
    [
        ("LinearLayer", "linear"),
        ("EmbeddingLayer", "embedding"),
        ("DropoutLayer", "dropout"),
        ("BatchNormLayer", "norm"),
        ("NoRunningStatsBatchNormLayer", "norm"),
        ("PredictionBatchNormLayer", "norm"),
        ("LayerNormLayer", "norm"),
    ],
)
def test_marker_ops_stay_reachable_from_the_package(op_name, submodule):
    # deserialize_graph resolves an op's recorded import path with getattr on this package, so these
    # bindings are load-bearing rather than convenience re-exports.
    from_package = getattr(pytensor_ml.layers, op_name)
    from_submodule = getattr(importlib.import_module(f"pytensor_ml.layers.{submodule}"), op_name)

    assert from_package is from_submodule


def test_batch_norm_without_running_stats_normalizes_with_batch_statistics(rng):
    X = pt.tensor("X", shape=(None, 4))
    normalize = pytensor.function([X], BatchNorm2D("bn", n_in=4, track_running_stats=False)(X))

    out = normalize(rng.normal(loc=5.0, scale=3.0, size=(256, 4)).astype(floatX))

    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=0), 1.0, rtol=1e-3)


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no_affine"])
def test_batch_norm_running_stats_write_back_to_the_right_inputs(affine):
    # The update map indexes inputs positionally, and the affine parameters shift the running
    # statistics along by two when present.
    X = pt.tensor("X", shape=(None, 4))
    batch_norm = BatchNorm2D("bn", n_in=4, affine=affine)

    updates = collect_non_trainable_updates(batch_norm(X))

    assert updates == {
        batch_norm.running_mean: batch_norm.new_running_mean,
        batch_norm.running_var: batch_norm.new_running_var,
    }


def test_batch_norm_variants_agree_on_output_arity():
    X = pt.tensor("X", shape=(None, 4))
    tracked = BatchNorm2D("tracked", n_in=4)(X)
    untracked = BatchNorm2D("untracked", n_in=4, track_running_stats=False)(X)

    # Matching arity is what lets BatchNorm2D use one code path; the untracked variant reports the
    # batch statistics but must not write them anywhere.
    assert len(tracked.owner.outputs) == len(untracked.owner.outputs)
    assert collect_non_trainable_updates(untracked) == {}


# Every parameter a layer owns, and the value a network-wide scheme must leave it at. A scheme reaching a
# batch-norm scale would rescale a normalized activation by a random factor, defeating the layer; reaching a
# bias would undo the zero start these layers are documented to have.
FEATURES = pt.tensor("features", shape=(None, 4), dtype=floatX)
IDS = pt.tensor("ids", shape=(None, 4), dtype="int32")

DECLARED_BY_LAYERS = {
    "Linear": (lambda: Linear("fc", n_in=4, n_out=4), FEATURES, {"fc_W": None, "fc_b": 0.0}),
    "Embedding": (
        lambda: Embedding("emb", n_embeddings=6, n_features=4),
        IDS,
        {"emb_W": None},
    ),
    "BatchNorm2D": (lambda: BatchNorm2D("bn", n_in=4), FEATURES, {"bn_scale": 1.0, "bn_loc": 0.0}),
    "LayerNorm": (lambda: LayerNorm("ln", n_in=4), FEATURES, {"ln_scale": 1.0, "ln_loc": 0.0}),
}


@pytest.mark.parametrize("layer_name", sorted(DECLARED_BY_LAYERS), ids=sorted(DECLARED_BY_LAYERS))
def test_a_layer_declares_the_initializers_its_parameters_need(layer_name):
    """A parameter whose starting value is part of the layer's definition declares an initializer, so a
    network-wide scheme passes it by. The scheme here returns a sentinel, so any parameter it reaches is
    obvious and any declaration that stopped working shows up as that sentinel."""
    build, layer_input, expected = DECLARED_BY_LAYERS[layer_name]
    prediction = build()(layer_input)

    sentinel = 7.0
    reached_by_the_scheme = CustomInitializer(
        lambda shape, dtype, rng: np.full(shape, sentinel, dtype=dtype)
    )
    params = collect_trainable_params(prediction)
    values = initialize_params(params, scheme=reached_by_the_scheme, rng=0)

    assert {p.name for p in params} == set(expected)
    for param, value in zip(params, values):
        if expected[param.name] is None:
            np.testing.assert_allclose(value, sentinel)  # no declaration: the scheme applies
        else:
            np.testing.assert_allclose(value, expected[param.name])


# One case per keyword: the layer to build with it, the parameter it must reach, and the parameters it must
# leave alone. A keyword that hits the wrong parameter, or that quietly strips a sibling's declaration, is
# the failure this is aimed at.
INITIALIZER_KEYWORDS = {
    "Linear.weight": (
        lambda init: Linear("fc", n_in=4, n_out=4, weight_initializer=init),
        FEATURES,
        "fc_W",
        {"fc_b": 0.0},
    ),
    "Linear.bias": (
        lambda init: Linear("fc", n_in=4, n_out=4, bias_initializer=init),
        FEATURES,
        "fc_b",
        {
            "fc_W": 0.0
        },  # the scheme's value: a bias initializer reaching the weight would show up here
    ),
    "Embedding.weight": (
        lambda init: Embedding("emb", n_embeddings=6, n_features=4, weight_initializer=init),
        IDS,
        "emb_W",
        {},
    ),
    "BatchNorm2D.scale": (
        lambda init: BatchNorm2D("bn", n_in=4, scale_initializer=init),
        FEATURES,
        "bn_scale",
        {"bn_loc": 0.0},
    ),
    "BatchNorm2D.loc": (
        lambda init: BatchNorm2D("bn", n_in=4, loc_initializer=init),
        FEATURES,
        "bn_loc",
        {"bn_scale": 1.0},
    ),
    "LayerNorm.scale": (
        lambda init: LayerNorm("ln", n_in=4, scale_initializer=init),
        FEATURES,
        "ln_scale",
        {"ln_loc": 0.0},
    ),
    "LayerNorm.loc": (
        lambda init: LayerNorm("ln", n_in=4, loc_initializer=init),
        FEATURES,
        "ln_loc",
        {"ln_scale": 1.0},
    ),
}


@pytest.mark.parametrize("case", sorted(INITIALIZER_KEYWORDS), ids=sorted(INITIALIZER_KEYWORDS))
def test_an_initializer_keyword_reaches_only_the_parameter_it_names(case):
    build, layer_input, target, siblings = INITIALIZER_KEYWORDS[case]
    sentinel = 7.0
    layer = build(
        CustomInitializer(lambda shape, dtype, rng: np.full(shape, sentinel, dtype=dtype))
    )
    prediction = layer(layer_input)

    params = collect_trainable_params(prediction)
    values = dict(zip((p.name for p in params), initialize_params(params, scheme="zeros", rng=0)))

    np.testing.assert_allclose(values[target], sentinel)
    for name, expected in siblings.items():
        np.testing.assert_allclose(values[name], expected)


def test_a_bias_initializer_replaces_the_zero_declaration_rather_than_fighting_it():
    """The keyword becomes the declaration, so it survives a network-wide scheme the same way the zero it
    replaced did. torch draws biases from a fan-scaled uniform, and this is how you say that here."""
    layer = Linear("fc", n_in=4, n_out=4, bias_initializer=NormalInitializer(0.0, 1.0))
    prediction = layer(pt.tensor("features", shape=(None, 4), dtype=floatX))

    params = collect_trainable_params(prediction)
    values = dict(zip((p.name for p in params), initialize_params(params, scheme="zeros", rng=0)))

    assert not np.allclose(values["fc_b"], 0.0)


@pytest.mark.parametrize("layer_name", sorted(DECLARED_BY_LAYERS), ids=sorted(DECLARED_BY_LAYERS))
def test_a_layer_is_born_holding_a_draw_from_its_own_initializer(layer_name):
    """Creating a parameter and giving it a value are one event, as they are in torch, flax and keras. A
    weight left at zero passes no gradient below the last layer of a stack, so a network nobody remembered to
    initialize trains its output bias and nothing else."""
    build, layer_input, expected = DECLARED_BY_LAYERS[layer_name]
    prediction = build()(layer_input)

    for param in collect_trainable_params(prediction):
        declared = expected[param.name]
        value = param.get_value()
        if declared is None:
            assert len(np.unique(value)) > 1, (
                f"{param.name} was born holding one repeated value rather than a draw"
            )
        else:
            np.testing.assert_allclose(value, declared, err_msg=param.name)


def test_a_keyword_initializer_reaches_the_value_and_not_only_the_declaration():
    """The bias case: a keyword used to be recorded as a declaration while the value stayed zero, so it took
    effect only if someone happened to call initialize afterwards."""
    sentinel = 7.0
    layer = Linear(
        "fc",
        n_in=4,
        n_out=4,
        bias_initializer=CustomInitializer(
            lambda shape, dtype, rng: np.full(shape, sentinel, dtype=dtype)
        ),
    )

    np.testing.assert_allclose(layer.b.get_value(), sentinel)


def test_a_non_default_scheme_still_reaches_every_weight_matrix():
    """The trap in drawing at construction: recording the fallback as a declaration would outrank `scheme`
    and silently disable it library-wide, while every test that initializes with the default still passed."""
    X = pt.tensor("features", shape=(None, 4), dtype=floatX)
    prediction = Sequential(Linear("fc1", 4, 4), Linear("fc2", 4, 4))(X)
    params = collect_trainable_params(prediction)

    values = dict(zip((p.name for p in params), initialize_params(params, scheme="ones", rng=0)))

    np.testing.assert_allclose(values["fc1_W"], 1.0)
    np.testing.assert_allclose(values["fc2_W"], 1.0)
    np.testing.assert_allclose(values["fc1_b"], 0.0)  # its declaration still outranks the scheme


def test_construction_draws_do_not_leak_into_a_seeded_initialize():
    """Construction draws from fresh entropy, so two identically seeded runs must still agree: the seeded
    initialize has to overwrite every value rather than build on what construction happened to produce."""

    def seeded_values():
        X = pt.tensor("features", shape=(None, 4), dtype=floatX)
        prediction = Sequential(Linear("fc1", 4, 4), Linear("fc2", 4, 4))(X)
        params = collect_trainable_params(prediction)
        return dict(zip((p.name for p in params), initialize_params(params, rng=0)))

    first, second = seeded_values(), seeded_values()

    assert set(first) == set(second)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)
