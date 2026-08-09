from typing import get_args

import numpy as np
import pytensor
import pytest

from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import collect_trainable_params
from pytensor_ml.state import (
    _INITIALIZERS,
    CustomInitializer,
    InitializationScheme,
    OneInitializer,
    ZeroInitializer,
    initialize_params,
)


def test_scheme_names_match_the_initializer_registry():
    assert set(get_args(InitializationScheme)) == set(_INITIALIZERS)


class TestInitializeParams:
    @pytest.mark.parametrize("scheme", sorted(_INITIALIZERS))
    def test_values_match_parameter_shapes_and_dtypes(self, scheme, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)

        values = initialize_params(params, scheme=scheme, rng=np.random.default_rng(0))

        assert len(values) == len(params)
        for param, val in zip(params, values):
            assert val.shape == param.get_value().shape
            assert str(val.dtype) == str(param.get_value().dtype)

    def test_xavier_normal_scales_by_fan_in_plus_fan_out(self, simple_network):
        X, y = simple_network
        weight = next(p for p in collect_trainable_params(y) if p.name == "fc1_W")
        fan_sum = sum(weight.get_value().shape)

        [value] = initialize_params([weight], scheme="xavier_normal", rng=np.random.default_rng(42))

        assert value.std() == pytest.approx(np.sqrt(2.0 / fan_sum), rel=0.05)
        # Xavier uniform targets this same variance; only a normal draw crosses its hard bound.
        assert np.abs(value).max() > np.sqrt(6.0 / fan_sum)

    @pytest.mark.parametrize("scheme", ["zeros", ZeroInitializer()], ids=["by_name", "by_instance"])
    def test_zeros(self, scheme, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)

        values = initialize_params(params, scheme=scheme)

        assert len(values) == len(params)
        for val in values:
            np.testing.assert_array_equal(val, 0)

    def test_reproducible_with_seed(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)

        values1 = initialize_params(params, rng=np.random.default_rng(123))
        values2 = initialize_params(params, rng=np.random.default_rng(123))

        for v1, v2 in zip(values1, values2):
            np.testing.assert_array_equal(v1, v2)

    def test_parameters_do_not_all_receive_the_same_draws(self):
        # The seed must be an int: passing a Generator masks the regression this guards.
        first = trainable(np.zeros((4, 4), dtype="float64"), "first")
        second = trainable(np.zeros((4, 4), dtype="float64"), "second")

        values = initialize_params([first, second], scheme="unit_uniform", rng=0)

        assert not np.array_equal(values[0], values[1])

    def test_accepts_a_custom_initializer(self):
        params = [
            trainable(np.zeros((4, 4), dtype="float64"), "first"),
            trainable(np.zeros((4, 2), dtype="float64"), "second"),
        ]
        constant = CustomInitializer(lambda shape, dtype, rng: np.full(shape, 7.0, dtype=dtype))

        values = initialize_params(params, scheme=constant)

        assert len(values) == len(params)
        for val in values:
            np.testing.assert_array_equal(val, 7.0)


class TestDeclaredInitializers:
    def test_a_declared_initializer_wins_over_the_scheme(self):
        # Starting from zeros, so the assertion only holds if the declared initializer actually ran.
        scale = trainable(np.zeros(4, dtype="float64"), "scale", initializer=OneInitializer())

        [value] = initialize_params([scale], scheme="xavier_normal", rng=0)

        np.testing.assert_array_equal(value, 1)

    def test_undeclared_parameters_still_follow_the_scheme(self):
        weight = trainable(np.zeros((4, 4), dtype="float64"), "weight")
        bias = trainable(np.zeros(4, dtype="float64"), "bias", initializer=ZeroInitializer())

        weight_value, bias_value = initialize_params([weight, bias], scheme="unit_uniform", rng=0)

        assert weight_value.min() > 0
        np.testing.assert_array_equal(bias_value, 0)

    def test_shared_variables_without_the_marker_class_follow_the_scheme(self):
        state = pytensor.shared(np.zeros(4, dtype="float64"), name="state")

        [value] = initialize_params([state], scheme="unit_uniform", rng=0)

        assert value.min() > 0

    def test_calling_an_initializer_overrides_a_declaration(self):
        scale = trainable(np.ones(3, dtype="float64"), "scale", initializer=OneInitializer())

        ZeroInitializer()(scale)

        np.testing.assert_array_equal(scale.get_value(), 0)


def test_calling_an_initializer_assigns_the_parameter_in_place():
    param = trainable(np.ones(3, dtype="float64"), "w")

    returned = ZeroInitializer()(param)

    np.testing.assert_array_equal(param.get_value(), 0)
    assert returned is param
