import numpy as np

from pytensor.compile import Function
from pytensor.printing import debugprint
from pytensor.tensor.variable import TensorVariable

from pytensor_ml import optim
from pytensor_ml.loss import Loss, supervised_loss
from pytensor_ml.params import TrainableParameter
from pytensor_ml.pytensorf import collect_trainable_params, compile_predict
from pytensor_ml.state import InitializationSchemeLike, initialize_params


class Model:
    """A network's input and output, with conveniences to initialize its weights, train, and run inference."""

    def __init__(self, X: TensorVariable, y: TensorVariable, compile_kwargs: dict | None = None):
        self.X = X
        self.y = y
        self._compile_kwargs = compile_kwargs or {}
        self._predict_fn: Function | None = None

    @property
    def weights(self) -> list[TrainableParameter]:
        return collect_trainable_params(self.y)

    def initialize(
        self,
        scheme: InitializationSchemeLike = "xavier_normal",
        seed: int | np.random.Generator | None = None,
    ) -> "Model":
        """
        Initialize the trainable weights in place and return self.

        Parameters
        ----------
        scheme : str or Initializer
            Initialization scheme for the weights: the name of a built-in scheme, or an
            :class:`~pytensor_ml.state.Initializer` instance. Default 'xavier_normal'.
        seed : int or numpy Generator, optional
            Seed for reproducible initialization.
        """
        parameters = self.weights
        for parameter, value in zip(parameters, initialize_params(parameters, scheme, rng=seed)):
            parameter.set_value(value)
        return self

    def compile_train(
        self,
        rule: optim.UpdateRule,
        loss_fn: Loss,
        ndim_out: int = 1,
        compile_kwargs: dict | None = None,
    ) -> Function:
        """
        Compile a one-step training function against a supervised target.

        Builds a target placeholder and loss from the model output with :func:`supervised_loss`, then compiles
        a step over the model's weights. The returned function is called as ``step(X_batch, target_batch)`` and
        returns the loss, applying every update in place.

        Parameters
        ----------
        rule : UpdateRule
            A configured optimizer, e.g. ``adam(1e-3)``.
        loss_fn : Loss
            Callable ``(target, prediction) -> scalar loss``.
        ndim_out : int
            Number of leading output dimensions the target shares. Default 1.
        compile_kwargs : dict, optional
            Keyword arguments forwarded to the function compiler. Defaults to the model's own compile kwargs.
        """
        loss, target = supervised_loss(self.y, loss_fn, ndim_out)
        return optim.compile_train(
            loss,
            rule,
            parameters=self.weights,
            inputs=[self.X, target],
            compile_kwargs=compile_kwargs or self._compile_kwargs,
        )

    def predict(self, X_values: np.ndarray) -> np.ndarray:
        if self._predict_fn is None:
            self._predict_fn = compile_predict(
                self.y, inputs=[self.X], compile_kwargs=self._compile_kwargs
            )

        return np.asarray(self._predict_fn(X_values))

    def __str__(self):
        return debugprint(self.y, file="str")
