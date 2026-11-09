import numpy as np
import pytest

from pytensor.graph.traversal import ancestors

from pytensor_ml.activations import Tanh
from pytensor_ml.layers import Input, Linear, Sequential
from pytensor_ml.pytensorf import collect_trainable_params

pm = pytest.importorskip("pymc")

from pytensor_ml.pymc import replace_with_model_variables  # noqa: E402


def test_replace_with_model_variables_swaps_in_priors_and_data_by_name():
    X = Input("X", shape=(None, 2))
    logits = Sequential(
        Linear("hidden", n_in=2, n_out=3), Tanh(), Linear("output", n_in=3, n_out=1)
    )(X)[:, 0]

    with pm.Model() as model:
        X_data = pm.Data("X", np.ones((4, 2)))
        priors = [
            pm.Normal("hidden_W", shape=(2, 3)),
            pm.Normal("hidden_b", shape=(3,)),
            pm.Normal("output_W", shape=(3, 1)),
            pm.Normal("output_b", shape=(1,)),
        ]
        model_logits = replace_with_model_variables(logits)
        pm.Bernoulli("y", logit_p=model_logits, observed=np.array([0, 1, 1, 0]))

    graph = set(ancestors([model_logits]))
    assert collect_trainable_params(model_logits) == []
    assert X not in graph
    assert X_data in graph and all(prior in graph for prior in priors)
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_replace_with_model_variables_names_every_input_the_model_lacks():
    """Writing the priors is filling in names and shapes, so the error lists all of them at once."""
    X = Input("X", shape=(None, 2))
    prediction = Linear("fc", n_in=2, n_out=1)(X)

    with pm.Model():
        pm.Normal("fc_W", shape=(2, 1))
        with pytest.raises(
            ValueError, match=r"2 of the network's inputs: .*fc_b with shape \(1,\)"
        ):
            replace_with_model_variables(prediction)
