import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor import config
from sklearn.datasets import load_digits, make_regression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from pytensor_ml.activations import LeakyReLU
from pytensor_ml.layers import BatchNorm2D, Linear, Sequential
from pytensor_ml.loss import CrossEntropy, SquaredError, supervised_loss
from pytensor_ml.optim import adam, compile_train, sgd
from pytensor_ml.optim.base import state_for
from pytensor_ml.params import collect_non_trainable_params, collect_trainable_params, trainable
from pytensor_ml.state import initialize_params
from pytensor_ml.util import DataLoader


@pytest.fixture
def classification_data():
    features, labels = load_digits(return_X_y=True)
    onehot_labels = OneHotEncoder().fit_transform(labels[:, None]).toarray()
    return MinMaxScaler().fit_transform(features), onehot_labels


@pytest.fixture
def regression_data():
    features, target = make_regression(n_samples=1000, n_features=64, noise=10, random_state=0)
    return StandardScaler().fit_transform(features), StandardScaler().fit_transform(target[:, None])


def build_network(n_out: int) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("hidden1", n_in=64, n_out=256),
        LeakyReLU(),
        Linear("hidden2", n_in=256, n_out=128),
        LeakyReLU(),
        Linear("output", n_in=128, n_out=n_out),
    )
    return X, network(X)


def initialize(parameters, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for parameter, value in zip(parameters, initialize_params(parameters, rng=rng)):
        parameter.set_value(value)


def train(step, data, n_steps: int = 50, batch_size: int = 512) -> list[float]:
    dataloader = DataLoader(*data, batch_size=batch_size)
    return [float(step(*dataloader())) for _ in range(n_steps)]


@pytest.mark.parametrize(
    "rule", [sgd(learning_rate=1e-2), adam(learning_rate=1e-2)], ids=["sgd", "adam"]
)
def test_trains_classifier(classification_data, rule):
    X, prediction = build_network(n_out=10)
    parameters = collect_trainable_params(prediction)
    initialize(parameters)
    loss, target = supervised_loss(
        prediction, CrossEntropy(expect_onehot_labels=True, expect_logits=True), ndim_out=2
    )
    step = compile_train(loss, rule, parameters=parameters, inputs=[X, target])

    history = train(step, classification_data)
    assert history[-1] < history[0]


@pytest.mark.parametrize(
    "rule", [sgd(learning_rate=1e-2), adam(learning_rate=1e-2)], ids=["sgd", "adam"]
)
def test_trains_regressor(regression_data, rule):
    X, prediction = build_network(n_out=1)
    parameters = collect_trainable_params(prediction)
    initialize(parameters)
    loss, target = supervised_loss(prediction, SquaredError(), ndim_out=2)
    step = compile_train(loss, rule, parameters=parameters, inputs=[X, target])

    history = train(step, regression_data)
    assert history[-1] < history[0]


def test_supervised_loss_builds_target_and_scalar_loss():
    X = pt.tensor("X", shape=(None, 64))
    prediction = Linear("output", n_in=64, n_out=10)(X)
    loss, target = supervised_loss(
        prediction, CrossEntropy(expect_onehot_labels=True, expect_logits=True), ndim_out=2
    )
    assert target.type.shape == (None, 10)
    assert loss.type.ndim == 0


def test_optimizer_state_is_reachable_from_the_updates_dict():
    # The updates dict is the checkpoint handle: optimizer state is exactly the keys that are not parameters,
    # held by object identity. No wrapper is needed to retain them.
    X = pt.tensor("X", shape=(None, 4))
    prediction = Linear("output", n_in=4, n_out=2)(X)
    parameters = collect_trainable_params(prediction)
    loss, _ = supervised_loss(prediction, SquaredError(), ndim_out=2)

    updates = adam(learning_rate=1e-2)(loss, parameters)
    state = [variable for variable in updates if variable not in set(parameters)]

    # One shared step counter plus a first and second moment per parameter.
    assert len(state) == 1 + 2 * len(parameters)
    assert "adam/step_count" in {variable.name for variable in state}


def test_compile_train_infers_parameters_and_inputs():
    X = pt.tensor("X", shape=(None, 4))
    prediction = Linear("output", n_in=4, n_out=2)(X)
    initialize(collect_trainable_params(prediction))
    loss, _ = supervised_loss(prediction, SquaredError(), ndim_out=2)

    step = compile_train(loss, sgd(1e-2))  # parameters and inputs collected from the graph

    assert callable(step)


def test_state_for_requires_named_parameter():
    # An unnamed parameter would leave every state buffer named by its bare slot, so distinct parameters
    # would silently share state at serialization boundaries.
    anonymous = trainable(np.zeros(2))
    with pytest.raises(ValueError, match="unnamed parameter"):
        state_for(anonymous, "adam/first_moment")


def test_compile_train_rejects_duplicate_parameter_names():
    # Two parameters sharing a name give their optimizer state colliding names; compile_train refuses to
    # build a training step whose checkpointed state cannot be told apart.
    first = trainable(np.zeros(1), name="dup")
    second = trainable(np.zeros(1), name="dup")
    loss = ((first + second) ** 2).sum()
    with pytest.raises(ValueError, match="share the name"):
        compile_train(loss, adam(1e-3), parameters=[first, second], inputs=[])


def test_compile_train_includes_non_trainable_updates():
    # compile_train merges batch-norm running-stat updates that a bare gradient rule would omit.
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc", n_in=4, n_out=4), BatchNorm2D("bn", n_in=4))(X)
    parameters = collect_trainable_params(prediction)
    initialize(parameters)
    loss, target = supervised_loss(prediction, SquaredError(), ndim_out=2)
    step = compile_train(loss, sgd(1e-2), parameters=parameters, inputs=[X, target])

    running_mean = next(
        p for p in collect_non_trainable_params(prediction) if "running_mean" in p.name
    )
    before = running_mean.get_value().copy()

    rng = np.random.default_rng(0)
    step(rng.normal(size=(16, 4)).astype(config.floatX), np.zeros((16, 4), dtype=config.floatX))

    assert not np.allclose(running_mean.get_value(), before)
