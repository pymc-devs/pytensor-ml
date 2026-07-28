from pytensor_ml.params import NonTrainableParameter, TrainableParameter
from pytensor_ml.pytensorf import (
    collect_data_inputs,
    collect_graph_inputs,
    collect_non_trainable_params,
    collect_non_trainable_updates,
    collect_shared_variables,
    collect_trainable_params,
)


class TestCollectGraphInputs:
    def test_simple_network(self, simple_network):
        X, y = simple_network
        inputs = collect_graph_inputs(y)
        # Only X (SharedVariables are excluded)
        assert len(inputs) == 1
        assert X in inputs

    def test_accepts_single_variable(self, simple_network):
        X, y = simple_network
        inputs_single = collect_graph_inputs(y)
        inputs_list = collect_graph_inputs([y])
        assert inputs_single == inputs_list


class TestCollectSharedVariables:
    def test_simple_network(self, simple_network):
        X, y = simple_network
        shared_vars = collect_shared_variables(y)
        # 2 weights + 2 biases = 4 SharedVariables
        assert len(shared_vars) == 4
        assert X not in shared_vars

    def test_batchnorm_includes_all_shared(self, network_with_batchnorm):
        X, y = network_with_batchnorm
        shared_vars = collect_shared_variables(y)
        # fc1: W, b; bn1: loc, scale, running_mean, running_var; fc2: W, b = 8
        assert len(shared_vars) == 8


class TestCollectTrainableParams:
    def test_simple_network(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)
        # 2 weights + 2 biases = 4 params (all TrainableParameter)
        assert len(params) == 4
        assert all(isinstance(p, TrainableParameter) for p in params)

    def test_batchnorm_excludes_running_stats(self, network_with_batchnorm):
        X, y = network_with_batchnorm
        params = collect_trainable_params(y)
        # fc1: W, b; bn1: loc, scale; fc2: W, b = 6 params
        # running_mean and running_var are NonTrainableParameter
        assert len(params) == 6
        param_names = {p.name for p in params}
        assert "bn1_running_mean" not in param_names
        assert "bn1_running_var" not in param_names


class TestCollectNonTrainableParams:
    def test_simple_network_no_non_trainable(self, simple_network):
        _, y = simple_network
        non_trainable = collect_non_trainable_params(y)
        assert len(non_trainable) == 0

    def test_batchnorm_has_non_trainable(self, network_with_batchnorm):
        _, y = network_with_batchnorm
        non_trainable = collect_non_trainable_params(y)
        assert len(non_trainable) == 2
        assert all(isinstance(p, NonTrainableParameter) for p in non_trainable)
        names = {p.name for p in non_trainable}
        assert "bn1_running_mean" in names
        assert "bn1_running_var" in names


class TestCollectNonTrainableUpdates:
    def test_simple_network_no_updates(self, simple_network):
        _, y = simple_network
        updates = collect_non_trainable_updates(y)
        assert updates == {}

    def test_batchnorm_has_running_stat_updates(self, network_with_batchnorm):
        _, y = network_with_batchnorm
        updates = collect_non_trainable_updates(y)
        assert len(updates) == 2
        old_names = {v.name for v in updates.keys()}
        assert "bn1_running_mean" in old_names
        assert "bn1_running_var" in old_names

    def test_dropout_no_updates(self, network_with_dropout):
        _, y = network_with_dropout
        updates = collect_non_trainable_updates(y)
        assert updates == {}


class TestCollectDataInputs:
    def test_simple_network(self, simple_network):
        X, y = simple_network
        data_inputs = collect_data_inputs(y)
        assert data_inputs == [X]

    def test_batchnorm_network(self, network_with_batchnorm):
        X, y = network_with_batchnorm
        data_inputs = collect_data_inputs(y)
        assert data_inputs == [X]
