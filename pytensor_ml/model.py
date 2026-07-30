from collections.abc import Sequence

import numpy as np

from pytensor.compile import Function
from pytensor.graph.basic import Variable
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
        """The trainable parameters, in graph-input order rather than construction order.

        ``Sequential(Linear("fc1", ...), ReLU(), Linear("fc2", ...))`` yields
        ``[fc2_b, fc2_W, fc1_b, fc1_W]``, so ``weights[0]`` is not the first layer's weight matrix. The
        order is stable, which is all :meth:`initialize` needs to zip values onto the right parameters.
        """
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
        loss_fn: Loss | None = None,
        ndim_out: int = 1,
        compile_kwargs: dict | None = None,
        *,
        loss: TensorVariable | None = None,
        inputs: Sequence[Variable] | None = None,
        extra_outputs: Sequence[Variable] | None = None,
    ) -> Function:
        """
        Compile a one-step training function, either against a supervised target or a prebuilt loss.

        Given ``loss_fn``, builds a target placeholder from the model output with :func:`supervised_loss` and
        the step is called as ``step(X_batch, target_batch)``. Given ``loss`` instead, trains that graph
        directly, which is what an autoencoder or a language-model objective needs -- neither has a target
        separate from its input. Either way the step applies every update in place.

        Parameters
        ----------
        rule : UpdateRule
            A configured optimizer, e.g. ``adam(1e-3)``.
        loss_fn : Loss, optional
            Callable ``(target, prediction) -> scalar loss``. Mutually exclusive with ``loss``.
        ndim_out : int
            Number of leading output dimensions the target shares. Supervised path only. Default 1.
        compile_kwargs : dict, optional
            Keyword arguments forwarded to the function compiler. Defaults to the model's own compile kwargs.
        loss : TensorVariable, optional
            A scalar loss graph built over this model's output. Mutually exclusive with ``loss_fn``.
        inputs : sequence of Variable, optional
            Data inputs of the step, in call order. Collected from ``loss`` when omitted. Belongs to the
            prebuilt path; the supervised path derives its own.
        extra_outputs : sequence of Variable, optional
            Diagnostics to return alongside the loss, as in :func:`~pytensor_ml.optim.compile_train`.

        Returns
        -------
        step : Function
            The compiled one-step training function. Returns the loss alone, or ``(loss, *extra_outputs)``
            when diagnostics were requested.
        """
        if loss_fn is not None:
            if loss is not None or inputs is not None:
                raise ValueError(
                    "loss and inputs belong to the prebuilt path; omit them with loss_fn."
                )
            loss, target = supervised_loss(self.y, loss_fn, ndim_out)
            inputs = [self.X, target]
        elif loss is None:
            raise ValueError("Pass either loss_fn for a supervised target, or a prebuilt loss.")

        return optim.compile_train(
            loss,
            rule,
            parameters=self.weights,
            inputs=inputs,
            extra_outputs=extra_outputs,
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
