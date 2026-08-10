import numpy as np
import pytensor.tensor as pt

from pytensor import config

from pytensor_ml.base import Layer, UnaryLayerOp
from pytensor_ml.params import trainable
from pytensor_ml.state import Initializer


class EmbeddingLayer(UnaryLayerOp):
    __props__ = ("n_embeddings", "n_features")

    def build_inner_graph(self, ids, W):
        return [W[ids]]


class Embedding(Layer):
    r"""
    Lookup-table embedding.

    Map each integer index to a learned row of the ``(n_embeddings, n_features)`` table,
    appending a trailing feature axis of size ``n_features`` while preserving the shape of the
    index input.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "Embedding" when None.
    n_embeddings : int
        Number of rows in the table -- the number of distinct indices it can map.
    n_features : int
        Size of each embedding row.
    weight_initializer : Initializer, optional
        How the table is drawn, in place of whatever scheme :meth:`~pytensor_ml.model.Model.initialize` is
        given. Left to the scheme when omitted. A fan-scaled scheme puts the vocabulary size in the
        denominator, which is correct Xavier and much tighter than the ``NormalInitializer(0.0, 0.02)`` that
        reference implementations of GPT-2 use, so this is the keyword to reach for when matching one.
    """

    def __init__(
        self,
        name: str | None,
        n_embeddings: int,
        n_features: int,
        *,
        weight_initializer: Initializer | None = None,
    ):
        self.name = name if name else "Embedding"
        self.n_embeddings = n_embeddings
        self.n_features = n_features

        W_value = np.zeros((n_embeddings, n_features), dtype=config.floatX)
        self.W = trainable(W_value, f"{self.name}_W", initializer=weight_initializer)

    def __call__(self, ids: pt.TensorLike) -> pt.TensorVariable:
        ids = pt.as_tensor(ids)

        out = EmbeddingLayer(
            name=self.name,
            n_embeddings=self.n_embeddings,
            n_features=self.n_features,
        )(ids, self.W)
        out.name = f"{self.name}_output"

        return out
