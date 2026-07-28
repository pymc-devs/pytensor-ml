import numpy as np
import pytensor.tensor as pt

from pytensor import config

from pytensor_ml.base import Layer, UnaryLayerOp
from pytensor_ml.params import trainable


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
    """

    def __init__(self, name: str | None, n_embeddings: int, n_features: int):
        self.name = name if name else "Embedding"
        self.n_embeddings = n_embeddings
        self.n_features = n_features

        W_value = np.zeros((n_embeddings, n_features), dtype=config.floatX)
        self.W = trainable(W_value, f"{self.name}_W")

    def __call__(self, ids: pt.TensorLike) -> pt.TensorVariable:
        ids = pt.as_tensor(ids)

        out = EmbeddingLayer(
            name=self.name,
            n_embeddings=self.n_embeddings,
            n_features=self.n_features,
        )(ids, self.W)
        out.name = f"{self.name}_output"

        return out
