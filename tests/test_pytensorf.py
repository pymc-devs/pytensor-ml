import numpy as np
import pytensor.tensor as pt

from pytensor import config

from pytensor_ml.layers import Dropout, Linear, Sequential
from pytensor_ml.params import collect_trainable_params
from pytensor_ml.pytensorf import compile_predict
from pytensor_ml.state import initialize_params


def test_compile_predict_removes_dropout():
    # The inference rewrite drops Dropout, so repeated calls are deterministic; without it they would differ.
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc", n_in=4, n_out=4), Dropout(p=0.5))(X)
    parameters = collect_trainable_params(prediction)
    for parameter, value in zip(
        parameters, initialize_params(parameters, rng=np.random.default_rng(0))
    ):
        parameter.set_value(value)

    predict = compile_predict(prediction, inputs=[X])
    X_values = np.random.default_rng(0).normal(size=(8, 4)).astype(config.floatX)
    np.testing.assert_allclose(predict(X_values), predict(X_values))
