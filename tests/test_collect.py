from pytensor_ml.params import NonTrainableParameter, TrainableParameter
from pytensor_ml.pytensorf import (
    collect_data_inputs,
    collect_graph_inputs,
    collect_non_trainable_params,
    collect_non_trainable_updates,
    collect_shared_variables,
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
