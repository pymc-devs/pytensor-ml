import jax
import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify

from pytensor_ml.layers.attention import AttentionLayer


@jax_funcify.register(AttentionLayer)
def jax_funcify_AttentionLayer(op, node=None, **kwargs):
    """Dispatch the attention marker to ``jax.nn.dot_product_attention`` (XLA/cuDNN flash kernel)."""
    is_causal = op.is_causal
    scale = op.scale

    def attention(q, k, v, mask=None):
        # jax expects (batch, seq, head, dim); our convention is (batch, head, seq, dim). The additive
        # mask is (batch, head, seq, seq) in both, so only q/k/v are transposed. is_causal composes with
        # bias, so a combined causal + padding mask needs no manual construction here.
        q = jnp.swapaxes(q, -3, -2)
        k = jnp.swapaxes(k, -3, -2)
        v = jnp.swapaxes(v, -3, -2)
        out = jax.nn.dot_product_attention(q, k, v, bias=mask, scale=scale, is_causal=is_causal)
        return jnp.swapaxes(out, -3, -2)

    return attention
