import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor.gradient import disconnected_grad, zero_grad

from pytensor_ml.layers import Linear
from pytensor_ml.params import NonTrainableParameter, TrainableParameter, step_counter
from pytensor_ml.pytensorf import (
    collect_clock_updates,
    collect_data_inputs,
    collect_differentiable_params,
    collect_graph_inputs,
    collect_non_trainable_params,
    collect_non_trainable_updates,
    collect_shared_variables,
    collect_step_counters,
    collect_trainable_params,
)

FC_PARAMS = {"fc1_W", "fc1_b", "fc2_W", "fc2_b"}
BN_AFFINE = {"bn1_loc", "bn1_scale"}
BN_RUNNING_STATS = {"bn1_running_mean", "bn1_running_var"}


def names(variables):
    return {variable.name for variable in variables}


class TestCollectGraphInputs:
    def test_returns_only_the_data_input(self, simple_network):
        X, y = simple_network
        assert collect_graph_inputs(y) == [X]

    def test_accepts_a_single_variable_or_a_list(self, simple_network):
        _, y = simple_network
        assert collect_graph_inputs(y) == collect_graph_inputs([y])


class TestCollectSharedVariables:
    def test_simple_network(self, simple_network):
        _, y = simple_network
        assert names(collect_shared_variables(y)) == FC_PARAMS

    def test_batchnorm_includes_affine_and_running_stats(self, network_with_batchnorm):
        _, y = network_with_batchnorm
        assert names(collect_shared_variables(y)) == FC_PARAMS | BN_AFFINE | BN_RUNNING_STATS


class TestCollectTrainableParams:
    def test_simple_network(self, simple_network):
        _, y = simple_network
        params = collect_trainable_params(y)

        assert names(params) == FC_PARAMS
        assert all(isinstance(param, TrainableParameter) for param in params)

    def test_batchnorm_excludes_running_stats(self, network_with_batchnorm):
        _, y = network_with_batchnorm
        assert names(collect_trainable_params(y)) == FC_PARAMS | BN_AFFINE


class TestCollectDifferentiableParams:
    def test_matches_trainable_params_without_stop_gradients(self, simple_network):
        _, y = simple_network
        assert collect_differentiable_params(y) == collect_trainable_params(y)

    @pytest.mark.parametrize(
        "stop_gradient", [disconnected_grad, zero_grad], ids=["disconnected", "zero"]
    )
    def test_excludes_a_detached_network(self, stop_gradient):
        X = pt.tensor("X", shape=(None, 4))
        online, target = Linear("online", 4, 2)(X), Linear("target", 4, 2)(X)
        loss = ((online - stop_gradient(target)) ** 2).sum()

        assert names(collect_trainable_params(loss)) == {
            "online_W",
            "online_b",
            "target_W",
            "target_b",
        }
        assert names(collect_differentiable_params(loss)) == {"online_W", "online_b"}

    def test_keeps_a_parameter_that_is_also_reached_on_a_live_path(self):
        X = pt.tensor("X", shape=(None, 4))
        prediction = Linear("fc", 4, 2)(X)
        loss = ((prediction - disconnected_grad(prediction)) ** 2).sum() + prediction.sum()

        assert names(collect_differentiable_params(loss)) == {"fc_W", "fc_b"}


class TestCollectNonTrainableParams:
    def test_simple_network_has_none(self, simple_network):
        _, y = simple_network
        assert collect_non_trainable_params(y) == []

    def test_batchnorm_running_stats(self, network_with_batchnorm):
        _, y = network_with_batchnorm
        params = collect_non_trainable_params(y)

        assert names(params) == BN_RUNNING_STATS
        assert all(isinstance(param, NonTrainableParameter) for param in params)


class TestCollectNonTrainableUpdates:
    def test_simple_network_has_none(self, simple_network):
        _, y = simple_network
        assert collect_non_trainable_updates(y) == {}

    def test_batchnorm_updates_its_running_stats(self, network_with_batchnorm):
        _, y = network_with_batchnorm
        assert names(collect_non_trainable_updates(y)) == BN_RUNNING_STATS

    def test_dropout_has_none(self, network_with_dropout):
        _, y = network_with_dropout
        assert collect_non_trainable_updates(y) == {}


def test_collect_data_inputs_excludes_every_shared_variable(network_with_batchnorm):
    X, y = network_with_batchnorm
    assert collect_data_inputs(y) == [X]


class TestCollectStepCounters:
    def test_a_network_has_no_clock(self, simple_network):
        _, y = simple_network
        assert collect_step_counters(y) == []

    def test_a_clock_read_several_times_is_collected_once(self):
        clock = step_counter()
        assert collect_step_counters([clock * 2, clock + 1, clock]) == [clock]

    def test_a_clock_is_never_collected_as_a_parameter(self, simple_network):
        """An optimizer differentiating the loss with respect to the clock would fail outright."""
        _, y = simple_network
        clock = step_counter()

        assert clock not in collect_trainable_params([y, clock])
        assert clock not in collect_non_trainable_params([y, clock])


class TestCollectClockUpdates:
    def test_a_network_has_no_clock_updates(self, simple_network):
        _, y = simple_network
        assert collect_clock_updates(y) == {}

    def test_one_advance_however_many_readers(self):
        """The point of holding a clock by object: three readers, one advance."""
        clock = step_counter()
        updates = collect_clock_updates([clock * 2, clock + 5, pt.exp(clock.astype("float64"))])

        assert list(updates) == [clock]
        assert updates[clock].eval() == 1

    def test_a_clock_nobody_reads_is_not_collected(self):
        """An unread clock is not a graph input, so it costs nothing to allocate one and not use it."""
        unread = step_counter(name="unread_clock")
        read = step_counter(name="read_clock")

        assert list(collect_clock_updates([read + 1])) == [read]
        assert unread not in collect_clock_updates([read + 1])

    def test_a_scaled_reader_still_advances_it_once(self):
        """Scaling at the read site is what replaces a stride parameter: coarser time, same tick."""
        clock = step_counter()
        updates = collect_clock_updates([clock // 2, clock * 4])

        assert list(updates) == [clock]
        assert updates[clock].eval() == 1

    def test_clocks_that_agree_are_both_collected(self):
        first, second = step_counter(name="first"), step_counter(name="second")
        assert set(collect_clock_updates([first + second])) == {first, second}

    def test_clocks_that_disagree_raise(self):
        """Two clocks counting the same steps can only disagree if state was restored inconsistently."""
        restored, fresh = step_counter(name="restored"), step_counter(name="fresh")
        restored.set_value(np.asarray(120, dtype="int64"))

        with pytest.raises(ValueError, match="hold different step counts"):
            collect_clock_updates([restored + fresh])

    def test_accepts_a_single_variable_or_a_list(self):
        clock = step_counter()
        assert list(collect_clock_updates(clock + 1)) == list(collect_clock_updates([clock + 1]))
