import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import BatchNorm2D, Dropout, Linear, Sequential
from pytensor_ml.state import fans, initializer


@initializer
def he_normal(rng, shape):
    """A fan-scaled draw, written the way a user would: the fans come from `fans(shape)` rather than being
    handed over. Takes no parameters at all, so what a config records of it is the class alone."""
    fan_in, _ = fans(shape)
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)


@initializer
def constant(rng, shape, value):
    """Fill every element with ``value``. A draw no other initializer produces, so a parameter holding it
    says which initializer reached it. Defined here rather than inline so it also survives a round trip
    through a saved config, which a locally defined one cannot."""
    return np.full(shape, value)


@pytest.fixture
def simple_network():
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("fc1", 64, 32),
        ReLU(),
        Linear("fc2", 32, 10),
    )
    y = network(X)
    return X, y


@pytest.fixture
def network_with_batchnorm():
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("fc1", 64, 32),
        BatchNorm2D("bn1", n_in=32),
        ReLU(),
        Linear("fc2", 32, 10),
    )
    y = network(X)
    return X, y


@pytest.fixture
def network_with_dropout():
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("fc1", 64, 32),
        Dropout(p=0.5),
        ReLU(),
        Linear("fc2", 32, 10),
    )
    y = network(X)
    return X, y
