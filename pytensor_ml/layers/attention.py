import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import Layer, UnaryLayerOp
from pytensor_ml.layers.linear import Linear
from pytensor_ml.state import Initializer


class AttentionLayer(UnaryLayerOp):
    __props__ = ("is_causal", "scale")

    def build_inner_graph(self, q, k, v, mask=None):
        return [_sdpa_graph(q, k, v, mask, self.is_causal, self.scale)]


def _repeat_kv(x: TensorVariable, n_head: int | None, n_kv_head: int | None) -> TensorVariable:
    """Broadcast ``n_kv_head`` key/value heads up to ``n_head`` query heads (grouped-query attention).

    Each key/value head serves a contiguous group of ``n_head // n_kv_head`` query heads, matching the
    ``repeat_interleave`` convention used by every grouped-query model. When the head counts are equal this
    is a no-op and adds nothing to the graph -- the dense self-attention path stays clean.
    """
    if n_head is None or n_kv_head is None:
        raise ValueError("Attention requires statically known head counts on q and k.")
    if n_head == n_kv_head:
        return x
    return pt.repeat(x, n_head // n_kv_head, axis=-3)


def _sdpa_graph(
    q: TensorVariable,
    k: TensorVariable,
    v: TensorVariable,
    mask: TensorVariable | None,
    is_causal: bool,
    scale: float | None,
) -> TensorVariable:
    k = _repeat_kv(k, q.type.shape[-3], k.type.shape[-3])
    v = _repeat_kv(v, q.type.shape[-3], v.type.shape[-3])

    if scale is None:
        scale_t = 1.0 / pt.sqrt(q.shape[-1].astype(config.floatX))
    else:
        scale_t = pt.as_tensor(scale, dtype=config.floatX)

    scores = (q @ k.swapaxes(-1, -2)) * scale_t

    if is_causal:
        sq, sk = q.shape[-2], k.shape[-2]
        # Position i may attend to j <= i, aligned bottom-right so a partial query block (the decoding
        # case) sees the whole prefix. Reduces to a plain lower triangle when sq == sk.
        q_idx = pt.arange(sq)[:, None]
        k_idx = pt.arange(sk)[None, :]
        causal = pt.where(k_idx <= q_idx + (sk - sq), 0.0, -np.inf).astype(config.floatX)
        scores = scores + causal

    if mask is not None:
        scores = scores + mask

    return pt.special.softmax(scores, axis=-1) @ v


def scaled_dot_product_attention(
    q: TensorVariable,
    k: TensorVariable,
    v: TensorVariable,
    *,
    mask: TensorVariable | None = None,
    is_causal: bool = False,
    scale: float | None = None,
) -> TensorVariable:
    r"""
    Scaled dot-product attention over the trailing ``(head, sequence, feature)`` axes.

    Compute

    .. math::

        \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}} + M\right) V,

    where the softmax is taken over the key axis and :math:`M` is the combined causal and/or additive
    mask. This is the position-agnostic attention kernel: rotary or other positional schemes act on
    ``q`` and ``k`` before they reach this function.

    The query and key/value head counts may differ (grouped-query attention): key/value heads are
    broadcast up to the query head count. The query/key feature size and the value feature size are
    independent -- only the query and key feature sizes must match.

    Parameters
    ----------
    q : TensorVariable
        Queries, shape ``(..., n_head, q_len, qk_dim)``.
    k : TensorVariable
        Keys, shape ``(..., n_kv_head, kv_len, qk_dim)``. ``n_head`` must be a multiple of
        ``n_kv_head``.
    v : TensorVariable
        Values, shape ``(..., n_kv_head, kv_len, v_dim)``.
    mask : TensorVariable, optional
        Additive mask broadcast onto the attention scores of shape ``(..., n_head, q_len, kv_len)``,
        e.g. ``0`` for attended positions and a large negative value for masked ones. Combined with the
        causal mask when both are requested. Default is None.
    is_causal : bool, optional
        Apply a causal mask so each position attends only to itself and earlier positions. Default is
        False.
    scale : float, optional
        Softmax temperature applied to the scores. Defaults to :math:`1/\sqrt{d_k}` when None.

    Returns
    -------
    TensorVariable
        Attention output, shape ``(..., n_head, q_len, v_dim)``.
    """
    q, k, v = (pt.as_tensor(t).copy() for t in (q, k, v))
    inputs = [q, k, v]

    if mask is not None:
        mask = pt.as_tensor(mask)
        inputs.append(mask)

    op = AttentionLayer(
        name="ScaledDotProductAttention",
        is_causal=is_causal,
        scale=scale,
    )
    result = op(*inputs)
    result.name = "attention_output"
    return result


class MultiheadAttention(Layer):
    r"""
    Multi-head self-attention.

    Project the input to per-head queries, keys, and values, apply
    :func:`scaled_dot_product_attention`, then project the concatenated heads back to the model
    dimension. Supports grouped-query attention through ``n_kv_head``.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "MultiheadAttention" when None.
    n_embd : int
        Model dimension of the input and output.
    n_head : int
        Number of query heads. Must divide ``n_embd`` evenly.
    n_kv_head : int, optional
        Number of key/value heads, for grouped-query attention. Must divide ``n_head`` evenly. Defaults
        to ``n_head`` (standard multi-head attention).
    bias : bool, optional
        Include bias terms in the projections. Default is True.
    is_causal : bool, optional
        Apply a causal mask in the attention. Default is False.
    out_proj_initializer : Initializer, optional
        How the output projection's weight is drawn. The three input projections are unaffected, which is
        what a scaling applied only to the projection writing back into a residual stream needs. Xavier
        normal when omitted, as for any other weight.
    """

    def __init__(
        self,
        name: str | None,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        bias: bool = True,
        is_causal: bool = False,
        *,
        out_proj_initializer: Initializer | None = None,
    ):
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        n_kv_head = n_kv_head if n_kv_head is not None else n_head
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})")

        self.name = name if name else "MultiheadAttention"
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.is_causal = is_causal

        self.q_proj = Linear(f"{self.name}_q_proj", n_embd, n_head * self.head_dim, bias)
        self.k_proj = Linear(f"{self.name}_k_proj", n_embd, n_kv_head * self.head_dim, bias)
        self.v_proj = Linear(f"{self.name}_v_proj", n_embd, n_kv_head * self.head_dim, bias)
        self.out_proj = Linear(
            f"{self.name}_out_proj",
            n_head * self.head_dim,
            n_embd,
            bias,
            weight_initializer=out_proj_initializer,
        )

    def _split_heads(self, x: pt.TensorVariable, n_head: int) -> pt.TensorVariable:
        # (..., seq, n_head * head_dim) -> (..., n_head, seq, head_dim). split_dims keeps the static
        # shape a plain reshape would erase, which lets the numba backend vectorize attention (e.g. under
        # vectorize_graph) instead of falling back to object mode.
        return pt.split_dims(x, shape=(n_head, self.head_dim), axis=-1).swapaxes(-3, -2)

    def __call__(self, x: pt.TensorLike, mask: pt.TensorLike | None = None) -> pt.TensorVariable:
        x = pt.as_tensor(x)

        q = self._split_heads(self.q_proj(x), self.n_head)
        k = self._split_heads(self.k_proj(x), self.n_kv_head)
        v = self._split_heads(self.v_proj(x), self.n_kv_head)

        if mask is not None:
            mask = pt.as_tensor(mask)
        attn = scaled_dot_product_attention(q, k, v, mask=mask, is_causal=self.is_causal)

        # (..., n_head, seq, head_dim) -> (..., seq, n_head * head_dim)
        attn = pt.join_dims(attn.swapaxes(-3, -2), start_axis=-2, n_axes=2)

        out = self.out_proj(attn)
        out.name = f"{self.name}_output"
        return out


class CausalSelfAttention(MultiheadAttention):
    r"""
    Causal multi-head self-attention.

    A :class:`MultiheadAttention` with ``is_causal=True``: each position attends only to itself and
    earlier positions. This is the attention used in GPT-style decoder blocks.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "CausalSelfAttention" when None.
    n_embd : int
        Model dimension of the input and output.
    n_head : int
        Number of query heads. Must divide ``n_embd`` evenly.
    n_kv_head : int, optional
        Number of key/value heads, for grouped-query attention. Defaults to ``n_head``.
    bias : bool, optional
        Include bias terms in the projections. Default is True.
    out_proj_initializer : Initializer, optional
        How the output projection's weight is drawn. See :class:`MultiheadAttention`.
    """

    def __init__(
        self,
        name: str | None,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        bias: bool = True,
        *,
        out_proj_initializer: Initializer | None = None,
    ):
        super().__init__(
            name if name else "CausalSelfAttention",
            n_embd,
            n_head,
            n_kv_head=n_kv_head,
            bias=bias,
            is_causal=True,
            out_proj_initializer=out_proj_initializer,
        )


__all__ = [
    "CausalSelfAttention",
    "MultiheadAttention",
    "scaled_dot_product_attention",
]
