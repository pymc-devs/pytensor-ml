from typing import Any

import numpy as np
import pytensor.tensor as pt
import pytensor.tensor.random as ptr

from pytensor import config
from pytensor.compile.sharedvalue import shared

from pytensor_ml.base import Layer, UnaryLayerOp


class DropoutLayer(UnaryLayerOp):
    __props__ = ("p",)

    def build_inner_graph(self, X, mask):
        return [pt.where(mask, ift=X / (1 - self.p), iff=0)]


class Dropout(Layer):
    def __init__(self, name: str | None = None, p: float = 0.5, random_state: Any | None = None):
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.name = name if name else "Dropout"
        self.p = p
        self.rng = shared(np.random.default_rng(random_state))

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)
        p = pt.as_tensor(self.p, dtype=config.floatX)
        _, mask = ptr.bernoulli(p=1 - p, size=X.shape, rng=self.rng, return_next_rng=True)
        mask = mask.astype(config.floatX)

        X_masked = DropoutLayer(
            name=f"{self.name}[p = {self.p}]",
            p=self.p,
        )(X, mask)
        X_masked.name = f"{self.name}_output"

        return X_masked
