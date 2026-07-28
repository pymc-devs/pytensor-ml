import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor import config
from pytensor.compile import Mode
from scipy.special import erf

from pytensor_ml.activations import GELU, LeakyReLU, ReLU, Sigmoid, SoftPlus, Swish, Tanh
from pytensor_ml.layers import Linear, Sequential
from pytensor_ml.loss import CrossEntropy, supervised_loss
from pytensor_ml.optim import adam, compile_train
from pytensor_ml.params import collect_trainable_params
from pytensor_ml.state import initialize_params

# Fastest compile mode. Test network is tiny so no optimizations needed
FAST_MODE = Mode(linker="py", optimizer="fast_compile")

# One-hot XOR: not linearly separable, so a single-hidden-layer network can only fit it if the activation
# supplies a working nonlinearity end to end.
XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=config.floatX)
XOR_Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=config.floatX)

# A confidently fit XOR drives cross-entropy toward zero; a linear/identity activation is stuck near ln(2).
# Threshold between them, with margin: every real activation crosses it within ~80 steps.
XOR_LOSS_THRESHOLD = 0.1
XOR_MAX_STEPS = 200

HIDDEN_ACTIVATIONS = [
    ReLU(),
    LeakyReLU(),
    Tanh(),
    Sigmoid(),
    SoftPlus(),
    GELU(approximate=False),
    GELU(approximate=True),
    Swish(),
]


def _activation_id(activation):
    if isinstance(activation, GELU) and activation.approximate:
        return "GELU_tanh"
    return type(activation).__name__


@pytest.mark.parametrize("negative_slope", [0.01, 0.1])
def test_leaky_relu_matches_reference(negative_slope):
    # The XOR test only checks that an activation supplies a working nonlinearity, which a sign-flipped
    # negative branch still does. Pin the actual values.
    x = pt.vector("x")
    values = np.linspace(-6, 6, 101).astype(config.floatX)

    f = pytensor.function([x], LeakyReLU(negative_slope=negative_slope)(x), mode=FAST_MODE)

    expected = np.where(values > 0, values, negative_slope * values)
    np.testing.assert_allclose(f(values), expected, rtol=1e-6)


def test_gelu_and_approx_match_erf_reference():
    x = pt.vector("x")
    values = np.linspace(-6, 6, 101)
    reference = 0.5 * values * (1 + erf(values / np.sqrt(2)))

    exact = pytensor.function([x], GELU(approximate=False)(x), mode=FAST_MODE)
    approx = pytensor.function([x], GELU(approximate=True)(x), mode=FAST_MODE)

    np.testing.assert_allclose(exact(values), reference, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(approx(values), reference, atol=1e-3)


@pytest.mark.parametrize("activation", HIDDEN_ACTIVATIONS, ids=_activation_id)
def test_activation_lets_a_network_learn_xor(activation):
    X = pt.matrix("X")
    output = Sequential(Linear("fc1", 2, 8), activation, Linear("fc2", 8, 2))(X)
    parameters = collect_trainable_params(output)
    for parameter, value in zip(
        parameters, initialize_params(parameters, rng=np.random.default_rng(0))
    ):
        parameter.set_value(value)
    loss, target = supervised_loss(
        output, CrossEntropy(expect_onehot_labels=True, expect_logits=True), ndim_out=2
    )
    step = compile_train(
        loss,
        adam(learning_rate=0.05),
        parameters=parameters,
        inputs=[X, target],
        compile_kwargs={"mode": FAST_MODE},
    )

    # Stop the moment the loss drops below threshold so CI pays only for the steps actually needed, and fail
    # loudly if a broken (gradient-killing) activation never gets there.
    for _ in range(XOR_MAX_STEPS):
        if float(step(XOR_X, XOR_Y)) < XOR_LOSS_THRESHOLD:
            return
    pytest.fail("network never confidently learned XOR (loss stayed >= threshold)")
