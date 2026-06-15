import json

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.op import RandomVariable

from pytensor_ml.activations import GELU, LeakyReLU, ReLU, Sigmoid, Softmax, SoftPlus, Swish, Tanh
from pytensor_ml.attention import scaled_dot_product_attention
from pytensor_ml.json_serialize import (
    deserialize_graph,
    op_from_json,
    prop_to_json,
    serialize_graph,
)
from pytensor_ml.layers import (
    BatchNorm2D,
    Concatenate,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Sequential,
    Squeeze,
)
from pytensor_ml.params import collect_shared_variables, collect_trainable_params

floatX = pytensor.config.floatX

ALL_ACTIVATIONS = [
    ReLU(),
    LeakyReLU(),
    Tanh(),
    Sigmoid(),
    SoftPlus(),
    Softmax(),
    GELU(approximate=False),
    GELU(approximate=True),
    Swish(),
    Swish(beta=1.5),
]


def assert_outputs_roundtrip(data_inputs, outputs, data_values):
    """Serialize the graph of ``outputs`` to JSON and back, and check the rebuilt graph computes the same."""
    output_list = outputs if isinstance(outputs, list) else [outputs]
    shared = collect_shared_variables(output_list)
    # allow_nan=False enforces strict, portable JSON: inf/nan must go through sentinels, not the
    # non-standard Infinity/NaN tokens a lenient parser would emit.
    blob = json.dumps(serialize_graph([*data_inputs, *shared], output_list), allow_nan=False)
    rebuilt_inputs, rebuilt_outputs = deserialize_graph(json.loads(blob))

    original = pytensor.function(data_inputs, output_list)  # shared inputs captured
    restored = pytensor.function(rebuilt_inputs, rebuilt_outputs)  # every input explicit
    feed = [*data_values, *(s.get_value() for s in shared)]
    for got, want in zip(restored(*feed), original(*data_values)):
        np.testing.assert_allclose(got, want, rtol=1e-6)


def initialized_network(*layers, seed=0):
    rng = np.random.default_rng(seed)
    X = pt.matrix("X")
    output = Sequential(*layers)(X)
    for parameter in collect_trainable_params(output):
        parameter.set_value(rng.normal(size=parameter.get_value().shape))
    return X, output


def _activation_id(activation):
    if isinstance(activation, GELU) and activation.approximate:
        return "GELU_tanh"
    if isinstance(activation, Swish):
        return f"Swish_beta{activation.beta}"
    return type(activation).__name__


@pytest.mark.parametrize("activation", ALL_ACTIVATIONS, ids=_activation_id)
def test_every_activation_roundtrips(activation):
    X, output = initialized_network(Linear("fc1", 4, 8), activation, Linear("fc2", 8, 4))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(5, 4))])


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_linear_bias_variants_roundtrip(bias):
    X, output = initialized_network(Linear("fc", 4, 3, bias=bias))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(5, 4))])


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"affine": False}, {"track_running_stats": False}],
    ids=["default", "no_affine", "no_running_stats"],
)
def test_batchnorm_variants_roundtrip(kwargs):
    X, output = initialized_network(Linear("fc", 4, 4), BatchNorm2D("bn", n_in=4, **kwargs))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(8, 4))])


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no_affine"])
def test_layernorm_roundtrips(affine):
    X, output = initialized_network(Linear("fc", 4, 6), LayerNorm("ln", n_in=6, affine=affine))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(8, 4))])


def test_squeeze_roundtrips():
    X = pt.matrix("X")
    assert_outputs_roundtrip(
        [X], Squeeze(X[:, :1], axis=1), [np.random.default_rng(0).normal(size=(5, 4))]
    )


def test_concatenate_roundtrips():
    X = pt.matrix("X")
    assert_outputs_roundtrip(
        [X], Concatenate([X, X], axis=1), [np.random.default_rng(0).normal(size=(5, 4))]
    )


def test_specify_shape_with_unknown_dim_roundtrips():
    # An unspecified (None) dim in specify_shape is a NoneTypeT constant -- exactly what the grad and
    # canonicalize rewrites inside compile_train bake into a real model's graph.
    X = pt.matrix("X")
    output = pt.specify_shape(X, (None, 4))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(0).normal(size=(5, 4))])


def test_embedding_roundtrips():
    ids = pt.lmatrix("ids")
    embedding = Embedding("emb", n_embeddings=8, n_features=5)
    embedding.W.set_value(np.random.default_rng(0).normal(size=(8, 5)))
    assert_outputs_roundtrip([ids], embedding(ids), [np.array([[1, 2, 3], [4, 0, 7]])])


@pytest.mark.parametrize("scale", [None, 0.5], ids=["default_scale", "custom_scale"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
def test_attention_roundtrips(is_causal, scale):
    # The causal branch bakes a -inf constant into the graph, exercising the non-finite codec path.
    rng = np.random.default_rng(0)
    q, k, v = (pt.tensor(name, shape=(2, 2, 4, 3)) for name in "qkv")
    output = scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    values = [rng.normal(size=(2, 2, 4, 3)).astype(floatX) for _ in range(3)]
    assert_outputs_roundtrip([q, k, v], output, values)


def test_multi_output_network_roundtrips():
    X = pt.matrix("X")
    output = [Linear("head_a", 4, 2)(X), Linear("head_b", 4, 3)(X)]
    for parameter in collect_trainable_params(output):
        parameter.set_value(np.random.default_rng(0).normal(size=parameter.get_value().shape))
    assert_outputs_roundtrip([X], output, [np.random.default_rng(1).normal(size=(5, 4))])


def test_dropout_graph_with_rng_roundtrips():
    # Dropout introduces an RNG (RandomGeneratorType) shared variable and a bernoulli RandomVariable; the
    # bernoulli must survive reconstruction, not collapse to an identity pass-through.
    X = pt.matrix("X")
    output = Dropout(p=0.5, random_state=0)(X)
    blob = json.dumps(serialize_graph([X, *collect_shared_variables(output)], [output]))
    _, rebuilt_outputs = deserialize_graph(json.loads(blob))
    assert any(
        node.owner and isinstance(node.owner.op, RandomVariable)
        for node in ancestors(rebuilt_outputs)
    )


def test_scan_recurrent_loop_roundtrips():
    rng = np.random.default_rng(0)
    sequence = pt.matrix("sequence")
    W_rec = pytensor.shared(rng.normal(size=(3, 3)), name="W_rec")

    def step(x_t, hidden):
        return pt.tanh(x_t + hidden @ W_rec)

    hidden_seq = pytensor.scan(
        step, sequences=sequence, outputs_info=pt.zeros(3), return_updates=False
    )
    assert_outputs_roundtrip([sequence], hidden_seq[-1], [rng.normal(size=(6, 3))])


def test_unregistered_scalar_op_raises_loudly():
    with pytest.raises(NotImplementedError, match="Unregistered scalar op"):
        op_from_json({"family": "scalar", "type": "NoSuchScalarOp"})


def test_non_json_native_prop_raises_loudly():
    # The "no silent drop" guarantee: a prop the codec can't represent must raise, not vanish.
    with pytest.raises(TypeError, match="Unserializable op prop"):
        prop_to_json(object())
