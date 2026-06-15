from functools import partial

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("mlx.core")

from pytensor_ml.attention import scaled_dot_product_attention
from tests.dispatch.mlx.test_basic import compare_mlx_and_py

floatX = pytensor.config.floatX
# mlx computes in float32 on Metal, so compare against the float64 py oracle with a loose tolerance.
assert_close = partial(np.testing.assert_allclose, atol=1e-3, rtol=1e-3)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(sum(map(ord, "MLX Attention")))


def symbolic_like(*arrays):
    return [pt.tensor(name, shape=a.shape) for name, a in zip("qkv", arrays)]


@pytest.mark.parametrize("scale", [None, 0.5], ids=["default_scale", "custom_scale"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
def test_attention_matches_py(is_causal, scale, rng):
    q, k, v = (rng.normal(size=(2, 3, 5, 4)).astype(floatX) for _ in range(3))
    q_sym, k_sym, v_sym = symbolic_like(q, k, v)
    out = scaled_dot_product_attention(q_sym, k_sym, v_sym, is_causal=is_causal, scale=scale)
    compare_mlx_and_py([q_sym, k_sym, v_sym], out, [q, k, v], assert_fn=assert_close)


def test_grouped_query(rng):
    q = rng.normal(size=(2, 6, 4, 5)).astype(floatX)
    k = rng.normal(size=(2, 2, 4, 5)).astype(floatX)
    v = rng.normal(size=(2, 2, 4, 5)).astype(floatX)
    q_sym, k_sym, v_sym = symbolic_like(q, k, v)
    out = scaled_dot_product_attention(q_sym, k_sym, v_sym, is_causal=True)
    compare_mlx_and_py([q_sym, k_sym, v_sym], out, [q, k, v], assert_fn=assert_close)


@pytest.mark.parametrize("is_causal", [False, True], ids=["mask_only", "mask_and_causal"])
def test_additive_mask(is_causal, rng):
    q, k, v = (rng.normal(size=(2, 3, 5, 4)).astype(floatX) for _ in range(3))
    mask = np.where(rng.normal(size=(2, 3, 5, 5)) > 0, 0.0, -np.inf).astype(floatX)
    # Keep the first key valid for every query so no row is fully masked -- a query whose entire causal
    # window is masked is ill-posed (softmax of all-masked) and produces NaN in the py oracle.
    mask[:, :, :, 0] = 0.0
    q_sym, k_sym, v_sym = symbolic_like(q, k, v)
    out = scaled_dot_product_attention(
        q_sym, k_sym, v_sym, mask=pt.as_tensor(mask), is_causal=is_causal
    )
    compare_mlx_and_py([q_sym, k_sym, v_sym], out, [q, k, v], assert_fn=assert_close)


def test_dispatch_auto_registers():
    """Loading MLX's dispatch must auto-register ours -- no `import pytensor_ml.dispatch.mlx` needed."""
    from pytensor.link.mlx.dispatch import mlx_funcify

    from pytensor_ml.attention import AttentionLayer

    assert AttentionLayer in mlx_funcify.registry
