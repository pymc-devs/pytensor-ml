from typing import get_args

import numpy as np
import pytensor
import pytest

from pytensor_ml.layers import Linear
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import collect_trainable_params
from pytensor_ml.state import (
    _INITIALIZERS,
    CustomInitializer,
    InitializationScheme,
    OneInitializer,
    XavierNormalInitializer,
    XavierUniformInitializer,
    ZeroInitializer,
    fans,
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


def test_a_convolution_kernel_folds_its_receptive_field_into_both_fans():
    """Summing the shape is the fan computation only for a matrix. For an ``(in, out, kH, kW)`` kernel every
    input channel reaches an output at each of the kH*kW offsets, so leaving the receptive field out of the
    fans overstates the spread: 0.258 where the correct scale for this shape is 0.096."""
    kernel_shape = (8, 16, 3, 3)  # asymmetric, so the orientation of the two fans is pinned as well
    fan_in, fan_out = fans(kernel_shape)
    assert (fan_in, fan_out) == (8 * 9, 16 * 9)

    value = XavierNormalInitializer().sample(kernel_shape, "float64", np.random.default_rng(0))

    assert value.std() == pytest.approx(np.sqrt(2.0 / (fan_in + fan_out)), rel=0.05)


@pytest.mark.parametrize(
    "shape", [(768, 768), (50257, 768), (4, 7)], ids=["square", "embedding", "small"]
)
def test_a_weight_matrix_draws_exactly_as_it_did_before(shape):
    """No parameter that already existed may move: a matrix has no dimensions past the second, so the sum of
    its fans is the sum of its shape, which is what the draw was scaled by before."""
    fan_in, fan_out = fans(shape)
    assert fan_in + fan_out == sum(shape)

    with_fans = XavierNormalInitializer().sample(shape, "float64", np.random.default_rng(7))
    with_shape_sum = np.random.default_rng(7).normal(0, np.sqrt(2.0 / sum(shape)), size=shape)

    np.testing.assert_array_equal(with_fans, with_shape_sum)


def test_the_fan_in_is_the_dimension_the_layers_treat_as_input():
    """Weights here are ``(n_in, n_out)``, the transpose of torch's layout, so the leading dimension is the
    fan-in. Xavier only reads the sum and cannot tell the difference; anything scaling by fan-in alone can."""
    layer = Linear("fc", n_in=4, n_out=7)

    fan_in, fan_out = fans(layer.W.get_value().shape)

    assert (fan_in, fan_out) == (4, 7)


@pytest.mark.parametrize(
    "initializer",
    [XavierNormalInitializer(), XavierUniformInitializer()],
    ids=["normal", "uniform"],
)
def test_a_fan_scaled_initializer_rejects_a_parameter_with_no_fans(initializer):
    """A bias or a norm scale has no fan-in and fan-out, and scaling by the length of a vector is a number
    with no meaning behind it. Such parameters declare their own initializer; reaching one with a scheme
    instead should say so rather than draw something arbitrary."""
    with pytest.raises(ValueError, match="at least two dimensions"):
        initializer.sample((768,), "float64", np.random.default_rng(0))
