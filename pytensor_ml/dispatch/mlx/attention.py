import math

import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify

from pytensor_ml.layers.attention import AttentionLayer


@mlx_funcify.register(AttentionLayer)
def mlx_funcify_AttentionLayer(op, node=None, **kwargs):
    """Dispatch the attention marker to ``mx.fast.scaled_dot_product_attention`` (fused Metal kernel)."""
    is_causal = op.is_causal
    scale = op.scale

    def attention(q, k, v, mask=None):
        # mlx uses our (batch, head, seq, dim) convention, so no transpose. scale must be a concrete
        # float. mlx takes either the string "causal" or an additive mask, not both, so fold the causal
        # triangle into the additive mask when a padding mask is also present.
        scale_value = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        if is_causal and mask is not None:
            q_len, kv_len = q.shape[-2], k.shape[-2]
            rows = mx.arange(q_len).reshape(q_len, 1)
            cols = mx.arange(kv_len).reshape(1, kv_len)
            mask = mask + mx.where(cols <= rows + (kv_len - q_len), 0.0, -float("inf"))
        elif is_causal:
            mask = "causal"
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=float(scale_value), mask=mask)

    return attention
