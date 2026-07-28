from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import pytensor.tensor as pt
import pytensor.tensor.random as ptr

from pytensor import config
from pytensor.compile.sharedvalue import shared

from pytensor_ml.params import (
    NonTrainableParameter,
    TrainableParameter,
    non_trainable,
    trainable,
)
from pytensor_ml.pytensorf import LayerOp, UnaryLayerOp


def shape_to_str(shape):
    inner = ",".join([str(st_dim) if st_dim is not None else "?" for st_dim in shape])
    return f"({inner})"


class Layer(ABC):
    @abstractmethod
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable: ...


class LinearLayer(UnaryLayerOp):
    __props__ = ("n_in", "n_out", "bias")

    def build_inner_graph(self, X, W, b=None):
        res = X @ W
        if self.bias:
            res = res + b
        return [res]


class Linear(Layer):
    def __init__(self, name: str | None, n_in: int, n_out: int, bias: bool = True):
        self.name = name if name else "Linear"
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias

        W_value = np.zeros((n_in, n_out), dtype=config.floatX)
        self.W = trainable(W_value, f"{self.name}_W")

        if self.bias:
            b_value = np.zeros(n_out, dtype=config.floatX)
            self.b = trainable(b_value, f"{self.name}_b")

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)

        inputs = [X, self.W]
        if self.bias:
            inputs.append(self.b)

        input_shape = shape_to_str(X.type.shape)
        output_shape = shape_to_str((X @ self.W).type.shape)

        ofg = LinearLayer(
            name=f"{self.name}[{input_shape} -> {output_shape}]",
            n_in=self.n_in,
            n_out=self.n_out,
            bias=self.bias,
        )
        out = ofg(*inputs)
        out.name = f"{self.name}_output"

        return out


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


def _batch_normalize(X, epsilon):
    mu = X.mean(axis=0)
    sigma_sq = X.var(axis=0)
    return (X - mu) / pt.sqrt(sigma_sq + epsilon), mu, sigma_sq


class BatchNormLayer(LayerOp):
    __props__ = ("n_in", "epsilon", "momentum", "affine")

    def update_map(self):
        # Outputs 1 and 2 (the new running mean and variance) update inputs 3 and 4 (the old ones).
        return {1: 3, 2: 4}

    def build_inner_graph(self, X, *rest):
        if self.affine:
            loc, scale, running_mean, running_var = rest
        else:
            running_mean, running_var = rest

        X_normalized, mu, sigma_sq = _batch_normalize(X, self.epsilon)
        X_rescaled = X_normalized * scale + loc if self.affine else X_normalized

        new_running_mean = self.momentum * mu + (1 - self.momentum) * running_mean
        new_running_var = self.momentum * sigma_sq + (1 - self.momentum) * running_var

        return [X_rescaled, new_running_mean, new_running_var]


class NoRunningStatsBatchNormLayer(UnaryLayerOp):
    __props__ = ("n_in", "epsilon", "affine")

    def build_inner_graph(self, X, *rest):
        X_normalized, _, _ = _batch_normalize(X, self.epsilon)
        if self.affine:
            loc, scale = rest
            return [X_normalized * scale + loc]
        return [X_normalized]


class PredictionBatchNormLayer(UnaryLayerOp):
    __props__ = ("n_in", "epsilon", "affine")

    def build_inner_graph(self, X, loc, scale, running_mean, running_var):
        res = (X - running_mean) / pt.sqrt(running_var + self.epsilon)
        return [loc + res * scale]


class BatchNorm2D(Layer):
    r"""
    Batch normalization over the batch axis.

    Standardize each feature across the batch, then optionally apply a learned affine transform:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta,

    where the mean and (biased) variance are taken over the batch (first) axis. During training the
    batch statistics are used and the running mean and variance are updated toward them as
    :math:`(1 - m)\,r + m\,b` from each batch statistic :math:`b`.

    Parameters
    ----------
    name : str, optional
        Name used as a prefix for the layer's parameters. Default is "BatchNorm".
    n_in : int, optional
        Size of the feature axis. Inferred from the input's last dimension on the first call when
        omitted.
    epsilon : float, optional
        Constant :math:`\epsilon` added to the variance for numerical stability. Default is 1e-5.
    momentum : float, optional
        Weight :math:`m` of the current batch statistic in the running-average update. Default is
        0.1.
    affine : bool, optional
        Apply the learned scale :math:`\gamma` and shift :math:`\beta`. Default is True.
    track_running_stats : bool, optional
        Maintain running mean and variance for use at prediction time. Default is True.

    Notes
    -----
    Batch normalization is not symmetric between training and prediction, so inference needs
    special handling. A plain forward pass -- calling the layer, or compiling with
    :func:`function` -- normalizes with the *current batch's* mean and variance, making each
    output depend on the other samples in the batch. That is what you want while training, but
    wrong at inference, where an example must normalize the same way no matter what it happens to
    be batched with. Compile prediction graphs with :func:`compile_predict`, which applies
    :func:`rewrite_for_prediction` to substitute the accumulated running statistics for the batch
    statistics.
    """

    def __init__(
        self,
        name: str | None = None,
        n_in: int | None = None,
        epsilon: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        self.name = name if name else "BatchNorm"
        self.n_in = n_in
        self.epsilon = epsilon
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.scale: TrainableParameter | None = None
        self.loc: TrainableParameter | None = None
        self.running_mean: NonTrainableParameter | None = None
        self.running_var: NonTrainableParameter | None = None

        self.initialized = False
        self._initialize_params(None)

    def _initialize_params(self, X: pt.TensorVariable | None):
        if self.initialized:
            return

        if self.n_in is None and X is None:
            return

        if X is not None:
            n_in = X.type.shape[-1]
        else:
            n_in = self.n_in

        if self.affine:
            scale_value = np.ones(n_in, dtype=config.floatX)
            loc_value = np.zeros(n_in, dtype=config.floatX)
            self.scale = trainable(scale_value, f"{self.name}_scale")
            self.loc = trainable(loc_value, f"{self.name}_loc")

        if self.track_running_stats:
            running_mean_value = np.zeros(n_in, dtype=config.floatX)
            running_var_value = np.ones(n_in, dtype=config.floatX)
            self.running_mean = non_trainable(running_mean_value, f"{self.name}_running_mean")
            self.running_var = non_trainable(running_var_value, f"{self.name}_running_var")

        self.initialized = True

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)
        inputs = [X]

        self._initialize_params(X)

        if self.affine:
            assert self.scale is not None and self.loc is not None
            inputs.extend([self.loc, self.scale])

        if self.track_running_stats:
            assert self.running_mean is not None and self.running_var is not None

            batch_norm_op: LayerOp = BatchNormLayer(
                name=self.name,
                n_in=self.n_in,
                epsilon=self.epsilon,
                momentum=self.momentum,
                affine=self.affine,
            )

            X_transformed, self.new_running_mean, self.new_running_var = batch_norm_op(
                *inputs, self.running_mean, self.running_var
            )

        else:
            batch_norm_op = NoRunningStatsBatchNormLayer(
                name=self.name,
                n_in=self.n_in,
                epsilon=self.epsilon,
                affine=self.affine,
            )

            X_transformed = batch_norm_op(*inputs)

        # BatchNormLayer is multi-output; narrow the normalized tensor for the single-tensor return.
        assert isinstance(X_transformed, pt.TensorVariable)
        X_transformed.name = f"{self.name}_output"

        return X_transformed


class LayerNormLayer(UnaryLayerOp):
    __props__ = ("n_in", "epsilon", "affine")

    def build_inner_graph(self, X, *rest):
        mu = X.mean(axis=-1, keepdims=True)
        # Biased variance (ddof=0), matching torch.nn.LayerNorm; pretrained weights assume it. Do
        # not "correct" to the unbiased estimator -- it would diverge from every pretrained model.
        sigma_sq = X.var(axis=-1, keepdims=True)
        X_normalized = (X - mu) / pt.sqrt(sigma_sq + self.epsilon)

        if self.affine:
            scale, loc = rest
            return [X_normalized * scale + loc]
        return [X_normalized]


class LayerNorm(Layer):
    r"""
    Layer normalization over the last (feature) axis.

    Standardize each sample independently across its features, then optionally apply a learned
    affine transform:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta,

    where the mean and (biased) variance are taken over the last axis. The statistics depend only on
    the current sample, so there are no running statistics and no train/eval distinction.

    Parameters
    ----------
    name : str, optional
        Name used as a prefix for the layer's parameters. Default is "LayerNorm".
    n_in : int, optional
        Size of the normalized feature axis. Inferred from the input's last dimension on the first
        call when omitted.
    epsilon : float, optional
        Constant :math:`\epsilon` added to the variance for numerical stability. Default is 1e-5.
    affine : bool, optional
        Apply the learned scale :math:`\gamma` and shift :math:`\beta`. Default is True.
    """

    def __init__(
        self,
        name: str | None = None,
        n_in: int | None = None,
        epsilon: float = 1e-5,
        affine: bool = True,
    ):
        self.name = name if name else "LayerNorm"
        self.n_in = n_in
        self.epsilon = epsilon
        self.affine = affine

        self.scale: TrainableParameter | None = None
        self.loc: TrainableParameter | None = None

        self.initialized = False
        self._initialize_params(None)

    def _initialize_params(self, X: pt.TensorVariable | None):
        if self.initialized:
            return

        if self.n_in is None and X is None:
            return

        n_in = X.type.shape[-1] if X is not None else self.n_in

        if self.affine:
            self.scale = trainable(np.ones(n_in, dtype=config.floatX), f"{self.name}_scale")
            self.loc = trainable(np.zeros(n_in, dtype=config.floatX), f"{self.name}_loc")

        self.initialized = True

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)
        self._initialize_params(X)

        inputs = [X]
        if self.affine:
            assert self.scale is not None and self.loc is not None
            inputs.extend([self.scale, self.loc])

        X_transformed = LayerNormLayer(
            name=self.name,
            n_in=self.n_in,
            epsilon=self.epsilon,
            affine=self.affine,
        )(*inputs)
        X_transformed.name = f"{self.name}_output"

        return X_transformed


def Input(name: str, shape: tuple[int, ...], dtype: str | None = None) -> pt.TensorVariable:
    """
    Create a named symbolic input tensor with a fully static shape.

    Parameters
    ----------
    name : str
        Name of the input variable.
    shape : tuple of int
        Static size of each dimension. Raise ``ValueError`` if any entry is not an integer.
    dtype : str or None
        Data type of the input. Defaults to ``floatX`` when None.
    """
    if not all(isinstance(dim, int) for dim in shape):
        raise ValueError("All dimensions must be integers")

    return pt.tensor(name=name, shape=shape, dtype=dtype)


def Sequential(*layers: Callable) -> Callable:
    def forward(x: pt.TensorLike) -> pt.TensorLike:
        for layer in layers:
            x = layer(x)
        return x

    return forward


Squeeze = pt.squeeze
Concatenate = pt.concatenate


__all__ = [
    "BatchNorm2D",
    "Concatenate",
    "Dropout",
    "Embedding",
    "Input",
    "LayerNorm",
    "Linear",
    "Sequential",
    "Squeeze",
]
