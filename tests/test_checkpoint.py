import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from safetensors.numpy import load_file

from pytensor_ml.checkpoint import load_state, save_state
from pytensor_ml.layers import Linear, Sequential
from pytensor_ml.loss import SquaredError, supervised_loss
from pytensor_ml.optim import adam
from pytensor_ml.pytensorf import collect_trainable_params, compile_predict, function
from pytensor_ml.state import initialize_params

floatX = pytensor.config.floatX


def shared(value, name, dtype="float64"):
    return pytensor.shared(np.asarray(value, dtype=dtype), name=name)


def downcast_in_place(variable, dtype):
    """Narrow a variable's stored value past the container's filtering, as a 32-bit backend's update does."""
    variable.container.storage[0] = np.asarray(variable.get_value(), dtype=dtype)


def build_trained_step(seed: int = 0):
    """Build a small network, take a few Adam steps, and return its params and optimizer state."""
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc1", n_in=4, n_out=8), Linear("fc2", n_in=8, n_out=2))(X)
    parameters = collect_trainable_params(prediction)
    rng = np.random.default_rng(seed)
    for parameter, value in zip(parameters, initialize_params(parameters, rng=rng)):
        parameter.set_value(value)

    loss, target = supervised_loss(prediction, SquaredError(), ndim_out=2)
    updates = adam(learning_rate=1e-2)(loss, parameters)
    state = [variable for variable in updates if variable not in set(parameters)]
    step = function([X, target], loss, updates=updates)

    X_value = rng.normal(size=(16, 4)).astype(floatX)
    target_value = rng.normal(size=(16, 2)).astype(floatX)
    for _ in range(3):
        step(X_value, target_value)
    return parameters, state, step, (X_value, target_value)


def test_round_trip_restores_parameters_and_optimizer_state(tmp_path):
    parameters, state, step, batch = build_trained_step()
    checkpointed = parameters + state
    path = tmp_path / "checkpoint.safetensors"
    save_state(checkpointed, path)
    snapshot = {variable.name: variable.get_value().copy() for variable in checkpointed}

    for _ in range(5):  # drive params and optimizer state away from the snapshot
        step(*batch)
    assert any(
        not np.allclose(variable.get_value(), snapshot[variable.name]) for variable in checkpointed
    )

    load_state(checkpointed, path)
    for variable in checkpointed:
        np.testing.assert_array_equal(variable.get_value(), snapshot[variable.name])


def test_step_counter_round_trips_exactly(tmp_path):
    # The step counter is a rank-0 int64 array; the round-trip must preserve its rank, dtype, and value
    # (np.ascontiguousarray would silently promote it to shape (1,) on save -- see _as_saveable_array).
    _, state, _, _ = build_trained_step()
    counter = next(variable for variable in state if variable.name == "adam/step_count")
    saved = counter.get_value()
    assert saved.shape == () and saved.dtype == np.int64

    path = tmp_path / "checkpoint.safetensors"
    save_state(state, path)
    counter.set_value(np.asarray(saved + 100, dtype="int64"))
    load_state(state, path)

    restored = counter.get_value()
    assert restored.shape == ()
    assert restored.dtype == np.int64
    np.testing.assert_array_equal(restored, saved)


def test_non_contiguous_value_round_trips_without_reordering(tmp_path):
    # safetensors serializes an array's raw buffer: handed an F-ordered value it silently writes the
    # elements in the wrong order rather than raising, so the save must copy to C order first.
    variable = shared(np.arange(6, dtype="float64").reshape(2, 3).T, "w")
    assert not variable.get_value(borrow=True).flags["C_CONTIGUOUS"]

    path = tmp_path / "checkpoint.safetensors"
    save_state([variable], path)
    variable.set_value(np.zeros((3, 2)))
    load_state([variable], path)

    np.testing.assert_array_equal(variable.get_value(), np.arange(6).reshape(2, 3).T)


@pytest.mark.parametrize(
    "mode, backend_module", [("JAX", "jax"), ("MLX", "mlx.core")], ids=["jax", "mlx"]
)
def test_round_trip_after_compiled_backend_update(mode, backend_module, tmp_path):
    # A function compiled for a JIT backend stores that backend's array type in the shared variables it
    # updates, so both sides of a checkpoint taken after training see no numpy at all.
    pytest.importorskip(backend_module)
    weight = shared([1.0, 2.0], "w", dtype="float32")
    counter = shared(0, "step", dtype="int64")  # int64 as the optimizer rules declare it
    pytensor.function([], [], updates={weight: weight * 2, counter: counter + 1}, mode=mode)()
    assert not isinstance(weight.get_value(), np.ndarray)  # the premise: no longer a numpy value

    path = tmp_path / "checkpoint.safetensors"
    save_state([weight, counter], path)
    weight.set_value(np.zeros(2, dtype="float32"))
    counter.set_value(np.asarray(0, dtype="int64"))
    load_state([weight, counter], path)

    np.testing.assert_array_equal(weight.get_value(), [2.0, 4.0])
    np.testing.assert_array_equal(counter.get_value(), 1)


def test_name_map_loads_under_renamed_keys(tmp_path):
    saved = shared([1.0, 2.0], "saved/weight")
    path = tmp_path / "checkpoint.safetensors"
    save_state([saved], path)

    target = shared([0.0, 0.0], "model/weight")
    load_state([target], path, name_map={"model/weight": "saved/weight"})
    np.testing.assert_array_equal(target.get_value(), [1.0, 2.0])


def test_load_rejects_missing_and_unexpected_keys(tmp_path):
    save_state([shared([1.0], "a"), shared([2.0], "b")], tmp_path / "c.safetensors")
    with pytest.raises(ValueError) as error:
        load_state([shared([0.0], "a"), shared([0.0], "c")], tmp_path / "c.safetensors")
    message = str(error.value)
    assert "missing" in message and "'c'" in message  # target 'c' absent from the archive
    assert "unexpected" in message and "'b'" in message  # archived 'b' has no target


def test_load_rejects_shape_mismatch(tmp_path):
    save_state([shared([1.0, 2.0, 3.0], "w")], tmp_path / "c.safetensors")
    target = shared([0.0, 0.0], "w")
    with pytest.raises(ValueError, match="shape"):
        load_state([target], tmp_path / "c.safetensors")
    np.testing.assert_array_equal(target.get_value(), [0.0, 0.0])


def test_load_rejects_dtype_mismatch(tmp_path):
    # Same shape, different dtype: a real footgun when loading a lower-precision checkpoint into fp64 params.
    save_state([shared([1, 2, 3], "w", dtype="int64")], tmp_path / "c.safetensors")
    target = shared([0.0, 0.0, 0.0], "w")
    with pytest.raises(ValueError, match="archive has int64"):
        load_state([target], tmp_path / "c.safetensors")
    np.testing.assert_array_equal(target.get_value(), [0.0, 0.0, 0.0])


def test_round_trip_survives_backend_narrowing_the_stored_dtype(tmp_path):
    counter = shared(2, "adam/step_count", dtype="int64")
    downcast_in_place(counter, "int32")
    path = tmp_path / "c.safetensors"
    save_state([counter], path)

    # Declared dtype, so the archive ports to a run at the other precision.
    assert load_file(path)["adam/step_count"].dtype == np.dtype("int64")

    fresh = shared(0, "adam/step_count", dtype="int64")
    load_state([fresh], path)
    np.testing.assert_array_equal(fresh.get_value(), 2)


def test_load_accepts_target_whose_stored_dtype_was_narrowed(tmp_path):
    # Mirror case: a clean archive into a live counter the backend already narrowed.
    save_state([shared(7, "adam/step_count", dtype="int64")], tmp_path / "c.safetensors")
    target = shared(0, "adam/step_count", dtype="int64")
    downcast_in_place(target, "int32")

    load_state([target], tmp_path / "c.safetensors")
    np.testing.assert_array_equal(target.get_value(), 7)
    assert np.asarray(target.get_value()).dtype == np.dtype("int64")


def test_load_leaves_all_targets_untouched_when_one_fails(tmp_path):
    save_state([shared([1.0], "a"), shared([2.0, 2.0], "b")], tmp_path / "c.safetensors")
    good = shared([0.0], "a")
    bad = shared([0.0], "b")  # wrong shape -> whole load must abort
    with pytest.raises(ValueError, match="shape"):
        load_state([good, bad], tmp_path / "c.safetensors")
    np.testing.assert_array_equal(good.get_value(), [0.0])


def test_save_rejects_unnamed_variable(tmp_path):
    with pytest.raises(ValueError, match="unnamed"):
        save_state([shared([1.0], None)], tmp_path / "c.safetensors")


def test_save_rejects_duplicate_names(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        save_state([shared([1.0], "w"), shared([2.0], "w")], tmp_path / "c.safetensors")


def test_load_rejects_non_injective_name_map(tmp_path):
    save_state([shared([1.0], "x")], tmp_path / "c.safetensors")
    with pytest.raises(ValueError, match="not injective"):
        load_state(
            [shared([0.0], "a"), shared([0.0], "b")],
            tmp_path / "c.safetensors",
            name_map={"a": "x", "b": "x"},
        )


def test_parameters_keep_identity_across_compilation():
    # The checkpoint handle is the parameter objects themselves. Compiling prediction and training graphs
    # must not clone them, or a checkpoint taken from collect_trainable_params would address dead copies.
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc1", n_in=4, n_out=8), Linear("fc2", n_in=8, n_out=2))(X)
    before = collect_trainable_params(prediction)

    compile_predict(prediction, inputs=[X])
    loss, target = supervised_loss(prediction, SquaredError(), ndim_out=2)
    function([X, target], loss, updates=adam(1e-2)(loss, before))

    after = collect_trainable_params(prediction)
    assert all(b is a for b, a in zip(before, after, strict=True))
