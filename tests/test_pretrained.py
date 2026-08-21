import json

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import BatchNorm, Dropout, Embedding, Linear, Sequential
from pytensor_ml.params import NonTrainableParameter, TrainableParameter
from pytensor_ml.pretrained import from_pretrained, load_network, save_network, save_pretrained
from pytensor_ml.pytensorf import collect_shared_variables, collect_trainable_params
from pytensor_ml.state import (
    NormalInitializer,
    UnrecordedInitializer,
    initialize_params,
    initializer,
)
from tests.conftest import constant, he_normal

floatX = pytensor.config.floatX


def build_initialized_network(seed=0):
    rng = np.random.default_rng(seed)
    X = pt.matrix("X")
    output = Sequential(Linear("fc1", 4, 8), ReLU(), Linear("fc2", 8, 2))(X)
    for parameter in collect_trainable_params(output):
        value = rng.normal(size=parameter.get_value().shape)
        parameter.set_value(value.astype(parameter.type.dtype))
    return X, output


def predict(inputs, output, x_value):
    # Plain literals default to float64, which does not fit a float32 graph.
    return pytensor.function(inputs, output)(np.asarray(x_value, dtype=floatX))


def test_from_pretrained_restores_architecture_and_weights(tmp_path):
    X, output = build_initialized_network()
    x_value = np.random.default_rng(1).normal(size=(5, 4))
    expected = predict([X], output, x_value)

    save_pretrained(output, tmp_path)
    restored_inputs, restored_output = from_pretrained(tmp_path)

    np.testing.assert_allclose(
        predict(restored_inputs, restored_output, x_value), expected, rtol=1e-6
    )


def test_save_pretrained_writes_config_and_weights(tmp_path):
    X, output = build_initialized_network()
    save_pretrained(output, tmp_path)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model.safetensors").exists()


def test_load_network_restores_trainable_params_holding_a_draw(tmp_path):
    """A rebuilt parameter holds a value for the same reason a freshly constructed one does, and holds it
    under the same law: a weight drawn, a bias at the zero it declares. The values are not the saved ones --
    this restores architecture, and `load_state` fills them by name."""
    X, output = build_initialized_network()
    save_network(output, tmp_path / "config.json")

    _, restored_output = load_network(tmp_path / "config.json")
    params = {p.name: p for p in collect_trainable_params(restored_output)}

    assert set(params) == {"fc1_W", "fc1_b", "fc2_W", "fc2_b"}
    assert all(isinstance(p, TrainableParameter) for p in params.values())
    for name in ("fc1_W", "fc2_W"):
        assert len(np.unique(params[name].get_value())) > 1, name
    for name in ("fc1_b", "fc2_b"):
        np.testing.assert_array_equal(params[name].get_value(), 0)


def test_from_pretrained_rejects_huggingface_directory(tmp_path):
    # A HuggingFace config shares our filenames but is a hyperparameter sheet; auto-detect must not misparse.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert", "hidden_size": 768}))
    with pytest.raises(NotImplementedError, match="HuggingFace"):
        from_pretrained(tmp_path)


def test_from_pretrained_rejects_unrecognized_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"foo": 1}))
    with pytest.raises(ValueError, match="Unrecognized config"):
        from_pretrained(tmp_path)


def test_load_network_rejects_unstamped_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    with pytest.raises(ValueError, match="HuggingFace config"):
        load_network(tmp_path / "config.json")


def test_dropout_network_roundtrips_with_fresh_rng(tmp_path):
    X = pt.matrix("X")
    output = Sequential(Linear("fc", 4, 4), Dropout(p=0.5, random_state=0))(X)
    for parameter in collect_trainable_params(output):
        value = np.random.default_rng(0).normal(size=parameter.get_value().shape)
        parameter.set_value(value.astype(parameter.type.dtype))
    fc_weight = (
        next(v for v in collect_shared_variables(output) if v.name == "fc_W").get_value().copy()
    )

    save_pretrained(output, tmp_path)
    restored_inputs, restored_output = from_pretrained(tmp_path)  # fresh RNG by default

    # The weights round-trip through safetensors even with an RNG in the graph; the RNG itself is fresh.
    restored_weight = next(v for v in collect_shared_variables(restored_output) if v.name == "fc_W")
    np.testing.assert_array_equal(restored_weight.get_value(), fc_weight)
    predict(restored_inputs, restored_output, np.zeros((3, 4)))  # the rebuilt graph runs


def test_restore_rng_reproduces_dropout_draws(tmp_path):
    X = pt.matrix("X")
    output = Dropout(p=0.5, random_state=0)(X)
    save_pretrained(output, tmp_path)
    x_value = np.random.default_rng(1).normal(size=(6, 4))

    original = predict([X], output, x_value)
    fresh_inputs, fresh_output = from_pretrained(tmp_path)  # default: fresh RNG
    restored_inputs, restored_output = from_pretrained(tmp_path, restore_rng=True)

    np.testing.assert_array_equal(predict(restored_inputs, restored_output, x_value), original)
    assert not np.array_equal(predict(fresh_inputs, fresh_output, x_value), original)


def test_batchnorm_non_trainable_state_survives_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = pt.matrix("X")
    output = Sequential(Linear("fc", 4, 4), BatchNorm("bn", n_in=4))(X)
    running_mean = next(v for v in collect_shared_variables(output) if v.name == "bn_running_mean")
    running_mean.set_value(rng.normal(size=4).astype(running_mean.type.dtype))

    save_pretrained(output, tmp_path)
    _, restored_output = from_pretrained(tmp_path)

    restored_mean = next(
        v for v in collect_shared_variables(restored_output) if v.name == "bn_running_mean"
    )
    assert isinstance(restored_mean, NonTrainableParameter)
    np.testing.assert_array_equal(restored_mean.get_value(), running_mean.get_value())


def test_load_network_rejects_an_older_format_version(tmp_path):
    # Op classes are recorded by import path, so a config from another layout must fail here rather than
    # later inside class resolution.
    X = pt.tensor("X", shape=(None, 4))
    path = tmp_path / "config.json"
    save_network(Linear("fc", 4, 2)(X), path, inputs=[X])

    config = json.loads(path.read_text())
    config["format_version"] = 1
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="graph format version 1"):
        load_network(path)


def test_a_loaded_batch_norm_returns_to_its_identity_transform(tmp_path):
    """The regression the whole exercise is for. A batch norm scale is ones because the layer declares it,
    and a config that dropped the declaration gave the scale a fan-scaled draw -- and once `fans` started
    refusing 1-D shapes, an outright error."""
    X = pt.matrix("X")
    output = Sequential(Linear("fc", 4, 4), BatchNorm("norm", n_in=4))(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    parameters = collect_trainable_params(restored)
    for parameter, value in zip(parameters, initialize_params(parameters, rng=0)):
        parameter.set_value(value)

    by_name = {p.name: p for p in parameters}
    np.testing.assert_array_equal(by_name["norm_scale"].get_value(), 1)
    np.testing.assert_array_equal(by_name["norm_loc"].get_value(), 0)


def test_a_parameterized_initializer_keeps_its_arguments(tmp_path):
    """Recording the registry name alone would be lossy: 'normal' rebuilds at the default spread, so a table
    built for GPT-2 at 0.02 would come back at 0.01 and nothing would say so."""
    X = pt.imatrix("ids")
    output = Embedding("tok", 32, 8, weight_initializer=NormalInitializer(0.0, 0.02))(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    [table] = collect_trainable_params(restored)

    assert isinstance(table.initializer, NormalInitializer)
    assert (table.initializer.mean, table.initializer.std) == (0.0, 0.02)


def test_a_decorated_initializer_round_trips_with_its_parameters(tmp_path):
    """The reason `@initializer` exists: the parameters a closure would have captured are declared instead,
    so they can be written down. `constant` lives in conftest, at module level, which is what lets the
    config find the class again."""
    X = pt.matrix("X")
    output = Linear("fc", 4, 4, weight_initializer=constant(value=7.0))(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    weight = next(p for p in collect_trainable_params(restored) if p.name == "fc_W")

    assert isinstance(weight.initializer, constant)
    assert weight.initializer.value == 7.0
    np.testing.assert_array_equal(weight.get_value(), 7.0)  # and the rebuilt parameter drew from it


def test_an_initializer_defined_locally_reports_what_was_lost(tmp_path):
    """An import cannot reach a class defined inside a function, so the config records only its name. Saving
    and loading still work, since restoring saved values needs no law; the redraw is what needs it back."""

    @initializer
    def local_constant(rng, shape, value):
        return np.full(shape, value)

    X = pt.matrix("X")
    output = Linear("fc", 4, 4, weight_initializer=local_constant(value=3.0))(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    weight = next(p for p in collect_trainable_params(restored) if p.name == "fc_W")

    assert isinstance(weight.initializer, UnrecordedInitializer)
    with pytest.raises(ValueError, match="local_constant, which the saved config could not record"):
        initialize_params([weight], rng=0)


def test_a_loaded_network_initializes_exactly_like_the_one_it_was_saved_from(tmp_path):
    """The whole point, stated as one equality. Same seed, same values, parameter for parameter -- which also
    pins that the rebuilt parameters come back in the saved order, since one generator draws them in
    sequence and a permutation would hand each the wrong draw."""
    X = pt.matrix("X")
    original = Sequential(
        Linear("fc1", 4, 8),
        ReLU(),
        BatchNorm("norm", n_in=8),
        Linear("fc2", 8, 2, weight_initializer=NormalInitializer(0.0, 0.02)),
    )(X)
    save_network(original, tmp_path / "config.json")
    _, restored = load_network(tmp_path / "config.json")

    def seeded(output):
        parameters = collect_trainable_params(output)
        values = initialize_params(parameters, rng=1234)
        return {p.name: value for p, value in zip(parameters, values)}

    from_original, from_loaded = seeded(original), seeded(restored)

    assert list(from_original) == list(from_loaded)  # same order, not merely the same names
    for name, value in from_original.items():
        np.testing.assert_array_equal(value, from_loaded[name], err_msg=name)


@initializer
def arange_fill(rng, shape, start):
    """A second decorated initializer, distinguishable from `constant` at a glance."""
    return np.arange(start, start + int(np.prod(shape))).reshape(shape)


def test_several_decorated_initializers_keep_their_own_class_and_parameters(tmp_path):
    """Props live on the instance and the class is shared, so two instances of one decorated initializer must
    not collide, and two different ones must not be confused for each other."""
    X = pt.matrix("X")
    output = Sequential(
        Linear("fc1", 4, 4, weight_initializer=constant(value=7.0)),
        Linear("fc2", 4, 4, weight_initializer=constant(value=-2.0)),
        Linear("fc3", 4, 4, weight_initializer=arange_fill(start=100.0)),
    )(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    weights = {p.name: p for p in collect_trainable_params(restored) if p.name.endswith("_W")}

    assert isinstance(weights["fc1_W"].initializer, constant)
    assert isinstance(weights["fc2_W"].initializer, constant)
    assert isinstance(weights["fc3_W"].initializer, arange_fill)

    assert weights["fc1_W"].initializer.value == 7.0
    assert weights["fc2_W"].initializer.value == -2.0
    assert weights["fc3_W"].initializer.start == 100.0

    np.testing.assert_array_equal(weights["fc1_W"].get_value(), 7.0)
    np.testing.assert_array_equal(weights["fc2_W"].get_value(), -2.0)
    assert weights["fc3_W"].get_value().min() == 100.0


def test_an_initializer_with_no_parameters_round_trips(tmp_path):
    """Serializability comes from the recorded parameters, and this has none -- the fans are computed inside
    the sampler from the shape it is handed, which the config never sees. So a scaled initializer written by
    hand survives a round trip on the strength of its import path alone."""
    X = pt.matrix("X")
    output = Linear("fc", 16, 4, weight_initializer=he_normal())(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    weight = next(p for p in collect_trainable_params(restored) if p.name == "fc_W")

    assert isinstance(weight.initializer, he_normal)
    assert weight.initializer.__props__ == ()
    # Redrawn from the restored class, and scaled by the fan-in the layer's shape implies.
    [value] = initialize_params([weight], rng=0)
    assert value.std() == pytest.approx(np.sqrt(2.0 / 16), rel=0.25)
