import numpy as np
import pytensor.tensor as pt

from pytensor import config

import pytensor_ml.model

from pytensor_ml.layers import BatchNorm2D, Linear, Sequential
from pytensor_ml.loss import SquaredError
from pytensor_ml.model import Model
from pytensor_ml.optim import sgd


class TestModelPredict:
    def test_matches_a_hand_computed_forward_pass(self):
        X = pt.tensor("X", shape=(None, 6))
        fc1 = Linear("fc1", n_in=6, n_out=3)
        fc2 = Linear("fc2", n_in=3, n_out=1)
        model = Model(X, Sequential(fc1, fc2)(X)).initialize(seed=42)

        X_test = np.random.default_rng(0).normal(size=(10, 6)).astype(config.floatX)
        hidden = X_test @ fc1.W.get_value() + fc1.b.get_value()
        expected = hidden @ fc2.W.get_value() + fc2.b.get_value()

        result = model.predict(X_test)

        assert result.shape == (10, 1)
        assert result.dtype == config.floatX
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_normalizes_with_running_stats_not_batch_stats(self):
        X = pt.tensor("X", shape=(None, 4))
        fc1 = Linear("fc1", n_in=4, n_out=4)
        bn = BatchNorm2D("bn1", n_in=4)
        model = Model(X, Sequential(fc1, bn)(X)).initialize(seed=42)

        bn.running_mean.set_value(np.array([1.0, 2.0, 3.0, 4.0], dtype=config.floatX))
        bn.running_var.set_value(np.ones(4, dtype=config.floatX))

        rng = np.random.default_rng(0)
        # Two batch sizes: batch statistics would differ between them, running statistics cannot.
        for n_rows in (5, 20):
            X_test = rng.normal(size=(n_rows, 4)).astype(config.floatX)
            fc_out = X_test @ fc1.W.get_value() + fc1.b.get_value()
            standardized = (fc_out - bn.running_mean.get_value()) / np.sqrt(
                bn.running_var.get_value() + bn.epsilon
            )
            expected = standardized * bn.scale.get_value() + bn.loc.get_value()

            np.testing.assert_allclose(model.predict(X_test), expected, rtol=1e-5)

    def test_compiles_once_and_reuses_the_function(self, monkeypatch):
        X = pt.tensor("X", shape=(None, 4))
        model = Model(X, Linear("fc1", n_in=4, n_out=2)(X)).initialize(seed=42)

        uncounted_compile_predict = pytensor_ml.model.compile_predict
        compile_count = 0

        def counting_compile_predict(*args, **kwargs):
            nonlocal compile_count
            compile_count += 1
            return uncounted_compile_predict(*args, **kwargs)

        monkeypatch.setattr(pytensor_ml.model, "compile_predict", counting_compile_predict)

        X_test = np.random.default_rng(0).normal(size=(5, 4)).astype(config.floatX)
        first = model.predict(X_test)
        second = model.predict(X_test)

        assert compile_count == 1
        np.testing.assert_array_equal(first, second)


def test_compile_train_accepts_a_prebuilt_loss():
    # An autoencoder reconstructs its own input, so there is no target separate from X and the supervised
    # path cannot express it. The step takes one argument, not two.
    X = pt.tensor("X", shape=(None, 4))
    reconstruction = Sequential(Linear("enc", n_in=4, n_out=2), Linear("dec", n_in=2, n_out=4))(X)
    model = Model(X, reconstruction).initialize(seed=0)

    step = model.compile_train(sgd(learning_rate=1e-2), loss=SquaredError()(X, reconstruction))

    X_batch = np.random.default_rng(0).normal(size=(64, 4)).astype(config.floatX)
    history = [float(step(X_batch)) for _ in range(50)]

    assert history[-1] < history[0]


def test_compile_train_reduces_loss():
    X = pt.tensor("X", shape=(None, 4))
    y = Sequential(Linear("fc1", n_in=4, n_out=8), Linear("fc2", n_in=8, n_out=1))(X)
    model = Model(X, y).initialize(seed=0)

    step = model.compile_train(sgd(learning_rate=1e-2), SquaredError(), ndim_out=2)

    rng = np.random.default_rng(0)
    X_batch = rng.normal(size=(64, 4)).astype(config.floatX)
    target = rng.normal(size=(64, 1)).astype(config.floatX)

    history = [float(step(X_batch, target)) for _ in range(50)]

    assert history[-1] < history[0]
