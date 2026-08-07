from typing import Literal, get_args

import numpy as np
import pytensor.tensor as pt

from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import Layer, UnaryLayerOp

Pairing = Literal["half", "adjacent"]
Scaling = Literal["none", "linear", "ntk"]

_PAIRINGS = get_args(Pairing)
_SCALINGS = get_args(Scaling)


def _validate_options(pairing: str, scaling: str, scaling_factor: float) -> None:
    """Reject unknown options where they are given, rather than letting a typo silently select the
    default behaviour inside the inner graph."""
    if pairing not in _PAIRINGS:
        raise ValueError(f"pairing must be one of {_PAIRINGS}, got {pairing!r}")
    if scaling not in _SCALINGS:
        raise ValueError(f"scaling must be one of {_SCALINGS}, got {scaling!r}")
    if scaling != "none" and scaling_factor <= 0:
        raise ValueError(f"scaling_factor must be positive, got {scaling_factor}")


def _resolve_head_dim(x: TensorVariable) -> int:
    """The rotated feature size, which must be static and even.

    Static because the frequencies are materialized as a constant at graph-build time, and even
    because the rotation acts on two-dimensional subspaces.
    """
    if not np.issubdtype(np.dtype(x.dtype), np.floating):
        raise ValueError(
            f"RotaryEmbedding rotates a floating-point tensor, but got dtype {x.dtype}. Cast the "
            f"input first."
        )

    head_dim = x.type.shape[-1]
    if head_dim is None:
        raise ValueError(
            "RotaryEmbedding needs a statically known last dimension to build its frequencies. "
            "Give the input a static feature dimension, e.g. pt.tensor(shape=(None, n_head, None, "
            "head_dim))."
        )
    if head_dim % 2:
        raise ValueError(f"RotaryEmbedding needs an even head dimension, got {head_dim}.")

    return head_dim


def _align_to_head_axes(
    angles: TensorVariable, x: TensorVariable, position_ids: TensorVariable
) -> TensorVariable:
    """Insert ``x``'s head axes into the angle tensor so the sequence axes line up.

    ``position_ids`` describes tokens: its last axis is the sequence and any leading axes are batch
    axes. ``x`` carries its head axes *between* those two, so the angles need matching placeholder
    axes. How many follows from the ranks, which is why the caller does not pass an unsqueeze position:
    ``(seq,)`` broadcasts over everything, and ``(batch, seq)`` lines up against
    ``(batch, n_head, seq, head_dim)`` without the caller reshaping. Deriving it also keeps the
    inserted axes statically 1, which is what pytensor requires before it will broadcast them -- a
    hand-built ``(batch, 1, seq)`` input whose middle axis is merely unknown is rejected at runtime.
    """
    n_head_axes = (x.type.ndim - 1) - position_ids.type.ndim
    if n_head_axes < 0:
        raise ValueError(
            f"position_ids has {position_ids.type.ndim} dimensions, more than the "
            f"{x.type.ndim - 1} non-feature dimensions of x. Its last axis is the sequence and any "
            f"leading axes are batch axes, so it can never be wider than x minus the feature axis."
        )
    if n_head_axes == 0:
        return angles

    return angles[(Ellipsis, *(None,) * n_head_axes, slice(None), slice(None))]


def _inverse_frequencies(
    head_dim: int, base: float, scaling: str, scaling_factor: float
) -> np.ndarray:
    r"""
    Angular frequencies :math:`\theta_i = \mathrm{base}^{-2i/d}` for each rotated pair.

    Computed in numpy at graph-build time: ``base``, ``head_dim`` and the scaling are all known then,
    so the frequencies enter the graph as one constant instead of a subgraph of ``arange`` and
    ``power``.

    Parameters
    ----------
    head_dim : int
        Size of the rotated feature axis. There are ``head_dim // 2`` frequencies.
    base : float
        Geometric base of the frequency ladder, :math:`\theta` in the RoFormer paper.
    scaling : str
        ``"none"``, ``"linear"`` for position interpolation, or ``"ntk"`` for NTK-aware scaling.
    scaling_factor : float
        Context-extension factor for the scaled variants; ignored when ``scaling="none"``.

    Returns
    -------
    ndarray
        Frequencies of shape ``(head_dim // 2,)``, in float64.
    """
    if scaling == "ntk":
        if head_dim <= 2:
            raise ValueError(
                f"NTK-aware scaling rescales the base by scaling_factor ** (d / (d - 2)), which is "
                f"undefined for head_dim <= 2; got {head_dim}."
            )
        # Static NTK-aware scaling: stretch the frequency ladder itself rather than the positions, so
        # the highest frequencies are left almost untouched. This is the exponent HuggingFace's
        # `_compute_dynamic_ntk_parameters` uses; the dynamic form, which recomputes the base from the
        # running sequence length, needs a runtime length and is deliberately not offered here.
        base = base * scaling_factor ** (head_dim / (head_dim - 2))

    inverse_frequencies = 1.0 / base ** (np.arange(0, head_dim, 2, dtype="float64") / head_dim)

    if scaling == "linear":
        # Position interpolation: dividing the frequencies is algebraically the same as dividing the
        # positions, because the angle is bilinear in the two.
        inverse_frequencies = inverse_frequencies / scaling_factor

    return inverse_frequencies


def _split_pairs(
    x: TensorVariable, pairing: str, half: int
) -> tuple[TensorVariable, TensorVariable]:
    """Split the feature axis into the two halves of every rotated pair.

    The two conventions differ only in which channels are paired, but they are not interchangeable:
    weights trained under one produce nonsense under the other.

    ``"half"`` pairs channel ``i`` with ``i + d/2``, matching HuggingFace's ``rotate_half`` (Llama,
    Qwen, Mistral, Gemma), GPT-NeoX and flax's ``RoPE``. ``"adjacent"`` pairs ``2i`` with ``2i + 1``,
    matching the original RoFormer formulation, GPT-J and ``torchtune``.
    """
    if pairing == "half":
        return x[..., :half], x[..., half:]

    return x[..., 0::2], x[..., 1::2]


def _join_pairs(
    x: TensorVariable, first: TensorVariable, second: TensorVariable, pairing: str
) -> TensorVariable:
    """Reassemble the feature axis, inverting :func:`_split_pairs` for the same ``pairing``.

    Both branches stay on ops every backend lowers, and both preserve the static feature size that
    lets the backends vectorize the rotation. Notably ``split_dims``/``join_dims`` would read more
    naturally for the interleaved case but have no JAX or MLX conversion, so a graph using them
    compiles only in the default backend.
    """
    if pairing == "half":
        return pt.concatenate([first, second], axis=-1)

    # Strided writes into a copy of x rather than an interleaving reshape. Every channel is
    # overwritten -- the even ones by `first`, the odd ones by `second` -- so no value of x survives,
    # and both were read before either write.
    rotated = pt.set_subtensor(x[..., 0::2], first)
    return pt.set_subtensor(rotated[..., 1::2], second)


class RotaryEmbeddingLayer(UnaryLayerOp):
    __props__ = ("base", "pairing", "scaling", "scaling_factor")

    def build_inner_graph(self, x, position_ids):
        _validate_options(self.pairing, self.scaling, self.scaling_factor)
        head_dim = _resolve_head_dim(x)

        inverse_frequencies = _inverse_frequencies(
            head_dim, self.base, self.scaling, self.scaling_factor
        )
        # Angles, and therefore cos/sin, are computed in x's own dtype: floatX is float32 or float64
        # here, so this already is the "compute in fp32" path that the fp16/bf16 frameworks upcast to.
        angles = position_ids[..., None].astype(x.dtype) * pt.constant(
            inverse_frequencies.astype(x.dtype), name="inverse_frequencies"
        )
        angles = _align_to_head_axes(angles, x, position_ids)
        cos, sin = pt.cos(angles), pt.sin(angles)

        first, second = _split_pairs(x, self.pairing, head_dim // 2)
        rotated = _join_pairs(
            x, first * cos - second * sin, second * cos + first * sin, self.pairing
        )

        return [rotated]


def rotary_embedding(
    x: pt.TensorLike,
    position_ids: pt.TensorLike,
    *,
    base: float = 10_000.0,
    pairing: Pairing = "half",
    scaling: Scaling = "none",
    scaling_factor: float = 1.0,
) -> TensorVariable:
    r"""
    Rotary position embedding (RoPE) applied to the trailing feature axis.

    Rotate each two-dimensional subspace of the feature axis by an angle proportional to the token's
    position:

    .. math::

        \begin{pmatrix} x'_a \\ x'_b \end{pmatrix} =
        \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\
                        \sin m\theta_i & \cos m\theta_i \end{pmatrix}
        \begin{pmatrix} x_a \\ x_b \end{pmatrix},
        \qquad \theta_i = \mathrm{base}^{-2i/d},

    where :math:`m` is the position and :math:`(a, b)` is the ``i``-th channel pair. Because a
    rotation is orthogonal, the dot product of a rotated query and a rotated key depends only on the
    *difference* of their positions -- that relative-position property is the point of the scheme, and
    it is what makes RoPE usable for incremental decoding: a token rotated by its absolute position
    keeps the same relationship to every earlier token no matter when it was computed.

    Apply this to queries and keys before :func:`~pytensor_ml.layers.attention.scaled_dot_product_attention`,
    never to values.

    Parameters
    ----------
    x : TensorLike
        Tensor whose last axis is rotated, typically queries or keys of shape
        ``(..., n_head, seq, head_dim)``. ``head_dim`` must be static and even.
    position_ids : TensorLike
        Position of each token, shape ``(..., seq)``: the last axis is the sequence and any leading
        axes are batch axes. ``x``'s head axes are inserted for you, so both ``(seq,)`` -- shared by
        the whole batch -- and ``(batch, seq)`` work against ``(batch, n_head, seq, head_dim)``.
        Positions are explicit rather than an implied ``0..seq-1`` so that the same graph serves a full
        sequence and a single decode step appended to a cached prefix.
    base : float, optional
        Geometric base of the frequency ladder (:math:`\theta` in some formulations). Default is
        10000.0, the value used by RoFormer, Llama, GPT-NeoX and torchtune.
    pairing : str, optional
        Which channels form each rotated pair. ``"half"`` (default) pairs ``i`` with ``i + d/2``, as
        HuggingFace's ``rotate_half``, GPT-NeoX and flax's ``RoPE`` do; ``"adjacent"`` pairs ``2i``
        with ``2i + 1``, as the original RoFormer, GPT-J and torchtune do. The two are a permutation of
        the feature axis apart, so this must match whatever the weights were trained with.
    scaling : str, optional
        Context-extension scheme. ``"none"`` (default) for plain RoPE, ``"linear"`` for position
        interpolation (Chen et al. 2023), or ``"ntk"`` for static NTK-aware scaling, which rescales the
        base by ``scaling_factor ** (head_dim / (head_dim - 2))``.
    scaling_factor : float, optional
        Extension factor for the scaled variants, ignored when ``scaling="none"``. Default is 1.0,
        which is the identity for both variants.

    Returns
    -------
    TensorVariable
        ``x`` with its last axis rotated, same shape and dtype.
    """
    _validate_options(pairing, scaling, scaling_factor)

    x = pt.as_tensor(x)
    position_ids = pt.as_tensor(position_ids)

    result = RotaryEmbeddingLayer(
        name="RotaryEmbedding",
        base=base,
        pairing=pairing,
        scaling=scaling,
        scaling_factor=scaling_factor,
    )(x, position_ids)
    result.name = "rotary_embedding_output"

    return result


class RotaryEmbedding(Layer):
    r"""
    Rotary position embeddings as a reusable, configured layer.

    Holds the frequency configuration so that queries and keys are rotated identically -- they must be,
    since attention compares them -- and calls :func:`rotary_embedding` to build the graph. The layer
    has no parameters of its own.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's output. Defaults to "RotaryEmbedding" when None.
    base : float, optional
        Geometric base of the frequency ladder. Default is 10000.0.
    pairing : str, optional
        ``"half"`` or ``"adjacent"``. Default is ``"half"``.
    scaling : str, optional
        ``"none"``, ``"linear"``, or ``"ntk"``. Default is ``"none"``.
    scaling_factor : float, optional
        Extension factor for the scaled variants. Default is 1.0.

    See Also
    --------
    rotary_embedding : The functional form, documenting the conventions in full.
    """

    def __init__(
        self,
        name: str | None = None,
        base: float = 10_000.0,
        pairing: Pairing = "half",
        scaling: Scaling = "none",
        scaling_factor: float = 1.0,
    ):
        _validate_options(pairing, scaling, scaling_factor)

        self.name = name if name else "RotaryEmbedding"
        self.base = base
        self.pairing = pairing
        self.scaling = scaling
        self.scaling_factor = scaling_factor

    # Positions are a required second input, which the one-tensor Layer.__call__ signature does not
    # describe. Widening the base class would loosen it for every layer that really is unary.
    def __call__(  # type: ignore[override]
        self, x: pt.TensorLike, position_ids: pt.TensorLike
    ) -> TensorVariable:
        out = rotary_embedding(
            x,
            position_ids,
            base=self.base,
            pairing=self.pairing,
            scaling=self.scaling,
            scaling_factor=self.scaling_factor,
        )
        out.name = f"{self.name}_output"

        return out


__all__ = [
    "RotaryEmbedding",
    "rotary_embedding",
]
