import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.mode import Mode
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.op import RandomVariable

from pytensor_ml.activations import ReLU
from pytensor_ml.params import collect_trainable_params
from pytensor_ml.transformer import FeedForward, TransformerBlock

floatX = pytensor.config.floatX
# The python linker evaluates the same graph without a slow numba compile of these block-sized graphs.
FAST = Mode(linker="py", optimizer="fast_compile")


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(sum(map(ord, "Transformer")))


def randomize(output, rng):
    for parameter in collect_trainable_params(output):
        parameter.set_value(rng.normal(size=parameter.get_value().shape).astype(floatX))


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def zero(parameter):
    parameter.set_value(np.zeros_like(parameter.get_value()))


@pytest.mark.parametrize(
    "kwargs, expected_hidden",
    [
        ({}, 32),
        ({"mlp_ratio": 2}, 16),
        ({"hidden_dim": 20}, 20),
        ({"hidden_dim": 20, "mlp_ratio": 2}, 20),
    ],
    ids=["default_4x", "ratio_2x", "explicit_hidden", "explicit_overrides_ratio"],
)
def test_feedforward_hidden_dim(kwargs, expected_hidden):
    ff = FeedForward("ff", d_model=8, **kwargs)
    assert ff.hidden_dim == expected_hidden
    assert ff.fc_in.W.get_value().shape == (8, expected_hidden)
    assert ff.fc_out.W.get_value().shape == (expected_hidden, 8)


@pytest.mark.parametrize(
    "activation, reference",
    [(None, gelu_tanh), (ReLU(), lambda x: np.maximum(x, 0.0))],
    ids=["default_gelu", "relu"],
)
def test_feedforward_matches_manual(activation, reference, rng):
    ff = FeedForward("ff", d_model=6, mlp_ratio=2, activation=activation)
    X = pt.tensor("X", shape=(2, 4, 6))
    out = ff(X)
    randomize(out, rng)

    X_np = rng.normal(size=(2, 4, 6)).astype(floatX)
    result = out.eval({X: X_np}, mode=FAST)

    hidden = reference(X_np @ ff.fc_in.W.get_value() + ff.fc_in.b.get_value())
    expected = hidden @ ff.fc_out.W.get_value() + ff.fc_out.b.get_value()
    np.testing.assert_allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize("norm_first", [True, False], ids=["pre_norm", "post_norm"])
def test_block_preserves_shape(norm_first, rng):
    block = TransformerBlock("blk", d_model=8, n_head=2, norm_first=norm_first, is_causal=True)
    X = pt.tensor("X", shape=(None, None, 8))
    out = block(X)
    assert out.type.shape == (None, None, 8)
    assert out.name == "blk_output"

    randomize(out, rng)
    result = out.eval({X: rng.normal(size=(2, 5, 8)).astype(floatX)}, mode=FAST)
    assert result.shape == (2, 5, 8)


def test_prenorm_block_is_residual_identity(rng):
    """With both sublayer output projections zeroed, a pre-norm block is the identity on its input."""
    block = TransformerBlock("blk", d_model=8, n_head=2, norm_first=True)
    X = pt.tensor("X", shape=(2, 5, 8))
    out = block(X)
    randomize(out, rng)
    for parameter in (
        block.attn.out_proj.W,
        block.attn.out_proj.b,
        block.ff.fc_out.W,
        block.ff.fc_out.b,
    ):
        zero(parameter)

    X_np = rng.normal(size=(2, 5, 8)).astype(floatX)
    np.testing.assert_allclose(out.eval({X: X_np}, mode=FAST), X_np, atol=1e-5)


def test_block_gradient_reaches_every_parameter(rng):
    block = TransformerBlock("blk", d_model=8, n_head=2, is_causal=True)
    X = pt.tensor("X", shape=(2, 4, 8))
    out = block(X)
    randomize(out, rng)
    parameters = collect_trainable_params(out)

    grads = pytensor.function([X], pt.grad(out.sum(), parameters), mode=FAST)(
        rng.normal(size=(2, 4, 8)).astype(floatX)
    )
    for parameter, grad in zip(parameters, grads):
        assert np.abs(grad).sum() > 0, f"no gradient into {parameter.name}"


@pytest.mark.parametrize("dropout", [0.0, 0.1], ids=["no_dropout", "dropout"])
def test_block_dropout_adds_rng_only_when_enabled(dropout):
    block = TransformerBlock("blk", d_model=8, n_head=2, dropout=dropout)
    out = block(pt.tensor("X", shape=(2, 4, 8)))
    has_rng = any(
        node.owner and isinstance(node.owner.op, RandomVariable) for node in ancestors([out])
    )
    assert has_rng == (dropout > 0)


def test_block_forwards_attention_config(rng):
    """The block plumbs is_causal and the mask into its attention. The masking and causal-severance
    math itself is covered by the attention suite, so this only checks the wiring."""
    block = TransformerBlock("blk", d_model=8, n_head=2, is_causal=True)
    assert block.attn.is_causal is True

    X = pt.tensor("X", shape=(1, 4, 8))
    out = block(X)
    randomize(out, rng)
    mask = np.zeros((1, 2, 4, 4), dtype=floatX)
    mask[:, :, :, 2] = -np.inf  # forbid key 2 for every query
    masked_out = block(X, mask)  # same weights, with a mask

    X_np = rng.normal(size=(1, 4, 8)).astype(floatX)
    assert not np.allclose(out.eval({X: X_np}, mode=FAST), masked_out.eval({X: X_np}, mode=FAST))
