from collections.abc import Sequence

import pymc as pm

from pytensor.graph.basic import Variable

from pytensor_ml.pytensorf import collect_graph_inputs, collect_trainable_params, rebuild


def replace_with_model_variables(
    outputs: Variable | Sequence[Variable], model: pm.Model | None = None
) -> Variable | list[Variable]:
    """
    Swap a network's weights and data inputs for the PyMC model variables of the same names.

    Every trainable weight and every data input the outputs read must have a model variable of the
    same name: a prior for a weight, and usually a ``pm.Data`` container for an input. Running
    statistics and other non-trainable state are left as they are.

    Parameters
    ----------
    outputs : Variable or sequence of Variable
        The network's outputs, which are left untouched.
    model : pymc.Model, optional
        The model to take variables from. Defaults to the model of the enclosing ``with`` block.

    Returns
    -------
    replaced : Variable or list of Variable
        The outputs computed from the model's variables, matching the form of ``outputs``.

    Examples
    --------
    Declare a prior for each weight under the weight's own name, and the data under the input's name:

    .. code-block:: python

        import numpy as np
        import pymc as pm

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.pymc import replace_with_model_variables

        X = Input("X", shape=(None, 3))
        prediction = Linear("fc", n_in=3, n_out=1)(X)[:, 0]

        with pm.Model():
            pm.Data("X", np.zeros((10, 3)))
            pm.Normal("fc_W", shape=(3, 1))
            pm.Normal("fc_b", shape=(1,))

            mu = replace_with_model_variables(prediction)
            pm.Normal("y", mu=mu, sigma=1.0, observed=np.zeros(10))
    """
    model = pm.modelcontext(model)
    weights = collect_trainable_params(outputs)
    data_inputs = collect_graph_inputs(outputs)

    missing = [
        variable for variable in [*weights, *data_inputs] if variable.name not in model.named_vars
    ]
    if missing:
        expected = ", ".join(
            f"{variable.name} with shape {_shape_of(variable)}" for variable in missing
        )
        raise ValueError(
            f"The model has no variable for {len(missing)} of the network's inputs: {expected}. "
            f"Declare each under that name, a prior for a weight and data for an input."
        )

    return rebuild(
        outputs, {variable: model[variable.name] for variable in [*weights, *data_inputs]}
    )


def _shape_of(variable: Variable) -> tuple[int | None, ...]:
    if hasattr(variable, "get_value"):
        return variable.get_value(borrow=True).shape
    return variable.type.shape
