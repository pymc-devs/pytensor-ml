from typing import Literal, get_args

import pytensor.tensor as pt

from pytensor.scalar import upcast
from pytensor.tensor.type import float_dtypes
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import UnaryLayerOp, _resolve_layer_name

Pairing = Literal["half", "adjacent"]
Scaling = Literal["none", "linear", "ntk"]

_PAIRINGS = get_args(Pairing)
_SCALINGS = get_args(Scaling)


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

    where :math:`m` is the position and :math:`(a, b)` is the :math:`i`-th channel pair
    :cite:p:`su2021roformer`.
    Query-key dot products depend only on relative position.

    Apply to queries and keys before
    :func:`~pytensor_ml.layers.attention.scaled_dot_product_attention`, never to values.

    Parameters
    ----------
    x : TensorLike
        Tensor whose last axis is rotated, typically queries or keys of shape
        ``(..., n_head, seq, head_dim)``. ``head_dim`` must be even. For JAX with ``pairing="half"``
        and for MLX, declare ``head_dim`` in the input's static shape.
    position_ids : TensorLike
        Token positions, shape ``(..., seq)``. Head axes are inserted so ``(seq,)`` and
        ``(batch, seq)`` both broadcast over ``(batch, n_head, seq, head_dim)``.
    base : float, optional
        Geometric base of the frequency ladder. Default is 10000.0.
    pairing : str, optional
        Pair channels across halves (``"half"`` [1]_) or consecutively (``"adjacent"`` [2]_). Must
        match the trained weights. Default is ``"half"``.
    scaling : str, optional
        Context-extension scheme: ``"none"``, ``"linear"`` for position interpolation
        :cite:p:`chen2023interpolation`, or ``"ntk"`` for static NTK-aware scaling. Default is
        ``"none"``.
    scaling_factor : float, optional
        Context-extension factor, ignored when ``scaling="none"``. Default is 1.0.

    Returns
    -------
    rotated : TensorVariable
        ``x`` with its last axis rotated, same shape and dtype.

    Examples
    --------
    Rotate a token at its absolute position in a decoded sequence:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import rotary_embedding

        token = np.ones((1, 8), dtype="float32")
        rotated = rotary_embedding(token, position_ids=np.array([12])).eval()

    References
    ----------
    .. [1] https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py#L109-L113
    .. [2] https://github.com/meta-pytorch/torchtune/blob/v0.6.1/torchtune/modules/position_embeddings.py#L99-L113
    """
    x = pt.as_tensor(x)
    position_ids = pt.as_tensor(position_ids)

    _validate_options(pairing=pairing, scaling=scaling, base=base, scaling_factor=scaling_factor)
    _validate_inputs(x=x, position_ids=position_ids)

    rotated = RotaryEmbeddingLayer(
        name="RotaryEmbedding",
        base=float(base),
        pairing=pairing,
        scaling=scaling,
        scaling_factor=float(scaling_factor),
    )(x, position_ids)
    rotated.name = "rotary_embedding_output"

    return rotated


class RotaryEmbedding:
    r"""
    Rotary position embeddings as a configured layer.

    Share one frequency configuration between queries and keys :cite:p:`su2021roformer`. The layer
    has no learned parameters.

    Parameters
    ----------
    name : str or None, optional
        Name prefix for the layer's output. Defaults to "RotaryEmbedding" when None.
    base : float, optional
        Geometric base of the frequency ladder. Default is 10000.0.
    pairing : str, optional
        ``"half"`` or ``"adjacent"``. Default is ``"half"``.
    scaling : str, optional
        ``"none"``, ``"linear"``, or ``"ntk"``. Default is ``"none"``.
    scaling_factor : float, optional
        Extension factor for the scaled variants. Default is 1.0.

    Examples
    --------
    Apply the same rotation to queries and keys before causal attention:

    .. code-block:: python

        import pytensor.tensor as pt

        from pytensor_ml.layers import Input, RotaryEmbedding, scaled_dot_product_attention

        queries = Input("queries", shape=(None, 4, None, 8))
        keys = Input("keys", shape=(None, 4, None, 8))
        values = Input("values", shape=(None, 4, None, 8))
        positions = pt.lvector("positions")
        rope = RotaryEmbedding("rope", pairing="half")
        output = scaled_dot_product_attention(
            rope(queries, positions), rope(keys, positions), values, is_causal=True
        )
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        base: float = 10_000.0,
        pairing: Pairing = "half",
        scaling: Scaling = "none",
        scaling_factor: float = 1.0,
    ):
        _validate_options(
            pairing=pairing, scaling=scaling, base=base, scaling_factor=scaling_factor
        )

        self.name = _resolve_layer_name(name, type(self).__name__, "base")
        self.base = base
        self.pairing = pairing
        self.scaling = scaling
        self.scaling_factor = scaling_factor

    def __call__(self, x: pt.TensorLike, position_ids: pt.TensorLike) -> TensorVariable:
        """
        Rotate a tensor using this layer's configuration.

        Parameters
        ----------
        x : TensorLike
            Tensor whose last axis is rotated, as in :func:`rotary_embedding`.
        position_ids : TensorLike
            Token positions, as in :func:`rotary_embedding`.

        Returns
        -------
        rotated : TensorVariable
            ``x`` with its last axis rotated, same shape and dtype.
        """
        rotated = rotary_embedding(
            x=x,
            position_ids=position_ids,
            base=self.base,
            pairing=self.pairing,
            scaling=self.scaling,
            scaling_factor=self.scaling_factor,
        )
        rotated.name = f"{self.name}_output"

        return rotated


class RotaryEmbeddingLayer(UnaryLayerOp):
    __props__ = ("base", "pairing", "scaling", "scaling_factor")

    def build_inner_graph(self, x, position_ids):
        if position_ids.type.ndim > 0:
            # The static check cannot see a length only known at run time, and a length of one would
            # then broadcast: every token gets the same rotation, or the sequence silently grows.
            position_ids = pt.specify_shape(position_ids, shape=(..., x.shape[-2]))

        static_head_dim = x.type.shape[-1]
        head_dim = x.shape[-1] if static_head_dim is None else static_head_dim
        dtype = x.type.dtype

        # float16 cannot count past 2048 exactly or hold a base past 65504, so the angles are built
        # at float32 or wider and only their cosines and sines return to the input's dtype.
        angle_dtype = upcast(dtype, "float32")
        angular_frequencies = _angular_frequencies(
            head_dim=head_dim,
            dtype=angle_dtype,
            base=self.base,
            scaling=self.scaling,
            scaling_factor=self.scaling_factor,
        )
        angles = position_ids[..., None].astype(angle_dtype) * angular_frequencies
        angles = _add_head_axes(angles=angles, x=x, position_ids=position_ids)
        cos = pt.cos(angles).astype(dtype)
        sin = pt.sin(angles).astype(dtype)

        first, second = _split_pairs(x=x, pairing=self.pairing, head_dim=head_dim)
        rotated = _join_pairs(
            x=x,
            first=first * cos - second * sin,
            second=second * cos + first * sin,
            pairing=self.pairing,
        )

        return [rotated]


def _validate_options(pairing: str, scaling: str, base: float, scaling_factor: float) -> None:
    """Reject unknown pairing and scaling options, and frequencies that are not positive."""
    if pairing not in _PAIRINGS:
        raise ValueError(f"pairing must be one of {_PAIRINGS}, got {pairing!r}")
    if scaling not in _SCALINGS:
        raise ValueError(f"scaling must be one of {_SCALINGS}, got {scaling!r}")
    if not base > 0:
        raise ValueError(f"base must be positive, got {base}. Llama uses 10000.0.")
    if scaling != "none" and not scaling_factor > 0:
        raise ValueError(f"scaling_factor must be positive, got {scaling_factor}")


def _validate_inputs(x: TensorVariable, position_ids: TensorVariable) -> None:
    """Reject inputs whose dtype or static shape cannot be rotated."""
    if x.type.dtype not in float_dtypes:
        raise ValueError(
            f"RotaryEmbedding needs a floating-point input, got dtype {x.type.dtype}. Cast x to a "
            "float dtype first."
        )

    n_position_axes = position_ids.type.ndim
    n_non_feature_axes = x.type.ndim - 1
    if n_position_axes == 0 and n_non_feature_axes > 0:
        raise ValueError(
            "position_ids is a scalar, but x has a sequence axis. Pass a vector with one position "
            "per token, such as a one-element vector for a single token."
        )
    if n_position_axes > n_non_feature_axes:
        raise ValueError(
            f"position_ids has {n_position_axes} dimensions, more than the {n_non_feature_axes} "
            "non-feature dimensions of x. Drop the extra axes from position_ids."
        )

    head_dim = x.type.shape[-1]
    if head_dim is not None and head_dim % 2:
        raise ValueError(
            f"RotaryEmbedding rotates channel pairs, so head_dim must be even, got {head_dim}. "
            "Project x to an even width first."
        )

    if n_position_axes == 0:
        return

    n_positions = position_ids.type.shape[-1]
    n_tokens = x.type.shape[-2]
    if n_positions is not None and n_tokens is not None and n_positions != n_tokens:
        raise ValueError(
            f"position_ids has {n_positions} entries for a sequence of {n_tokens} tokens. Pass one "
            "position per token."
        )


def _add_head_axes(
    angles: TensorVariable, x: TensorVariable, position_ids: TensorVariable
) -> TensorVariable:
    """Insert the head axes of ``x``, which ``position_ids`` lacks, before the sequence axis."""
    n_head_axes = x.type.ndim - 1 - position_ids.type.ndim

    return pt.expand_dims(angles, axis=tuple(range(-2 - n_head_axes, -2)))


def _angular_frequencies(
    head_dim: int | TensorVariable, dtype: str, base: float, scaling: str, scaling_factor: float
) -> TensorVariable:
    r"""
    Angular frequencies :math:`\theta_i = \mathrm{base}^{-2i/d}`, one per rotated pair.

    Returns
    -------
    angular_frequencies : TensorVariable
        Shape ``(head_dim // 2,)``, dtype ``dtype``. Folds to a constant when ``head_dim`` is
        static.
    """
    dimension = pt.cast(head_dim, dtype)
    frequency_base = pt.as_tensor(base, dtype=dtype)
    if scaling == "ntk":
        if isinstance(head_dim, int) and head_dim <= 2:
            raise ValueError(
                "NTK scaling rescales the base by scaling_factor ** (d / (d - 2)), which is "
                f"undefined for head_dim <= 2, got {head_dim}. Use scaling='linear' or a wider "
                "head_dim."
            )
        # Static NTK scaling changes the base independently of the sequence length.
        factor = pt.as_tensor(scaling_factor, dtype=dtype)
        frequency_base = frequency_base * factor ** (dimension / (dimension - 2))

    exponent = pt.arange(0, head_dim, 2, dtype=dtype) / dimension
    angular_frequencies = frequency_base**-exponent

    if scaling == "linear":
        angular_frequencies = angular_frequencies / pt.as_tensor(scaling_factor, dtype=dtype)

    return angular_frequencies


def _split_pairs(
    x: TensorVariable, pairing: str, head_dim: int | TensorVariable
) -> tuple[TensorVariable, TensorVariable]:
    """Split the feature axis into the two members of every rotated pair.

    The conventions are a permutation of the feature axis apart and are not interchangeable: weights
    trained under one produce nonsense under the other.

    ``"half"`` pairs channel ``i`` with ``i + d/2``, and ``"adjacent"`` pairs ``2i`` with
    ``2i + 1``.
    """
    if pairing == "half":
        half = head_dim // 2
        first, second = x[..., :half], x[..., half:]
    else:
        first, second = x[..., 0::2], x[..., 1::2]

    return first, second


def _join_pairs(
    x: TensorVariable, first: TensorVariable, second: TensorVariable, pairing: str
) -> TensorVariable:
    """Reassemble the feature axis, inverting :func:`_split_pairs` for the same ``pairing``."""
    if pairing == "half":
        rotated = pt.concatenate([first, second], axis=-1)
    else:
        rotated = x[..., 0::2].set(first)
        rotated = rotated[..., 1::2].set(second)

    return rotated


__all__ = [
    "RotaryEmbedding",
    "rotary_embedding",
]
