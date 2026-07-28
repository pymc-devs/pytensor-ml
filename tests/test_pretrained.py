import json

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import BatchNorm2D, Dropout, Linear, Sequential
from pytensor_ml.params import NonTrainableParameter, TrainableParameter
from pytensor_ml.pretrained import from_pretrained, load_network, save_network, save_pretrained
from pytensor_ml.pytensorf import collect_shared_variables, collect_trainable_params


def build_initialized_network(seed=0):
    rng = np.random.default_rng(seed)
    X = pt.matrix("X")
    output = Sequential(Linear("fc1", 4, 8), ReLU(), Linear("fc2", 8, 2))(X)
    for parameter in collect_trainable_params(output):
        parameter.set_value(rng.normal(size=parameter.get_value().shape))
    return X, output


def predict(inputs, output, x_value):
    return pytensor.function(inputs, output)(x_value)


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


def test_load_network_restores_zero_initialized_trainable_params(tmp_path):
    X, output = build_initialized_network()
    save_network(output, tmp_path / "config.json")

    _, restored_output = load_network(tmp_path / "config.json")
    params = collect_trainable_params(restored_output)

    assert len(params) == 4  # two Linear layers, weight + bias each
    assert all(isinstance(p, TrainableParameter) for p in params)
    assert all(np.all(p.get_value() == 0) for p in params)  # architecture only, weights unset


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
        parameter.set_value(np.random.default_rng(0).normal(size=parameter.get_value().shape))
    fc_weight = (
        next(v for v in collect_shared_variables(output) if v.name == "fc_W").get_value().copy()
    )

    save_pretrained(output, tmp_path)
    restored_inputs, restored_output = from_pretrained(tmp_path)  # fresh RNG by default

    # The weights round-trip through safetensors even with an RNG in the graph; the RNG itself is fresh.
    restored_weight = next(v for v in collect_shared_variables(restored_output) if v.name == "fc_W")
    np.testing.assert_array_equal(restored_weight.get_value(), fc_weight)
    pytensor.function(restored_inputs, restored_output)(np.zeros((3, 4)))  # the rebuilt graph runs


def test_restore_rng_reproduces_dropout_draws(tmp_path):
    X = pt.matrix("X")
    output = Dropout(p=0.5, random_state=0)(X)
    save_pretrained(output, tmp_path)
    x_value = np.random.default_rng(1).normal(size=(6, 4))

    original = pytensor.function([X], output)(x_value)
    fresh_inputs, fresh_output = from_pretrained(tmp_path)  # default: fresh RNG
    restored_inputs, restored_output = from_pretrained(tmp_path, restore_rng=True)

    np.testing.assert_array_equal(
        pytensor.function(restored_inputs, restored_output)(x_value), original
    )
    assert not np.array_equal(pytensor.function(fresh_inputs, fresh_output)(x_value), original)


def test_batchnorm_non_trainable_state_survives_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = pt.matrix("X")
    output = Sequential(Linear("fc", 4, 4), BatchNorm2D("bn", n_in=4))(X)
    running_mean = next(v for v in collect_shared_variables(output) if v.name == "bn_running_mean")
    running_mean.set_value(rng.normal(size=4))

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
