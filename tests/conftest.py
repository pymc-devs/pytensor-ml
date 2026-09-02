import logging

import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import BatchNorm, Dropout, Linear, Sequential
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
        Linear("fc1", n_in=64, n_out=32),
        ReLU(),
        Linear("fc2", n_in=32, n_out=10),
    )
    y = network(X)
    return X, y


@pytest.fixture
def network_with_batchnorm():
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("fc1", n_in=64, n_out=32),
        BatchNorm("bn1", n_in=32),
        ReLU(),
        Linear("fc2", n_in=32, n_out=10),
    )
    y = network(X)
    return X, y


@pytest.fixture
def network_with_dropout():
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("fc1", n_in=64, n_out=32),
        Dropout(p=0.5),
        ReLU(),
        Linear("fc2", n_in=32, n_out=10),
    )
    y = network(X)
    return X, y


@pytest.fixture(autouse=True)
def fail_on_swallowed_rewrite_errors(caplog):
    """Fail if a node rewriter raised while a test was rewriting a graph.

    Pytensor catches exceptions from a node rewriter, reports them through ``logger.error``, and leaves
    the graph untouched. Nothing else notices: ``filterwarnings = ["error"]`` only sees warnings, and a
    rewrite that crashes on the first node it touches looks exactly like one that correctly declined to
    match -- so a scan keeps the dropout an inference graph is supposed to drop, and the test still
    passes.
    """
    yield

    failures = [
        record.getMessage()
        for record in caplog.get_records("call")
        if record.name.startswith("pytensor.graph.rewriting") and record.levelno >= logging.ERROR
    ]

    assert not failures, "a node rewriter raised and pytensor swallowed it:\n" + "\n".join(failures)
