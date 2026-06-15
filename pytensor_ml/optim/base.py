from collections.abc import Callable, Sequence

import numpy as np
import pytensor

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.gradient import grad
from pytensor.tensor import TensorVariable
from pytensor.tensor.sharedvar import TensorSharedVariable

from pytensor_ml.pytensorf import rewrite_pregrad

# An optimizer parameter is a shared tensor variable: it can index the updates dict (it is a shared variable),
# take part in tensor arithmetic, and expose its concrete value via ``get_value``. Optimizer state buffers share
# the same type. Aliased for readable signatures.
type Parameter = TensorSharedVariable

# A mapping from a shared variable to the symbolic expression for its next value. This is pytensor's native
# `updates` contract, and it is the single currency every rule and transform in this package speaks: it carries
# the next parameter values *and* the next optimizer-state values in one identity-keyed dict.
Updates = dict[SharedVariable, TensorVariable]

# A chainable transform reads an updates dict and returns a new one, working in "step space" (updates[p] - p).
Transform = Callable[[Updates, Sequence[Parameter]], Updates]

# A configured optimizer: give it a loss (or precomputed gradients) and the parameters, get back an updates dict.
UpdateRule = Callable[[TensorVariable | Sequence[TensorVariable], Sequence[Parameter]], Updates]


def get_gradients(
    loss_or_gradients: TensorVariable | Sequence[TensorVariable],
    parameters: Sequence[Parameter],
) -> list[TensorVariable]:
    """
    Return gradients of the loss with respect to ``parameters``, or pass through precomputed gradients.

    Parameters
    ----------
    loss_or_gradients : TensorVariable or sequence of TensorVariable
        Either a scalar loss to differentiate, or an already-computed list of gradients, one per parameter.
    parameters : sequence of shared tensor variable
        Parameters to differentiate with respect to.

    Returns
    -------
    list of TensorVariable
        One gradient per parameter, in the order of ``parameters``.
    """
    if isinstance(loss_or_gradients, list | tuple):
        gradients = list(loss_or_gradients)
        if len(gradients) != len(parameters):
            raise ValueError(f"Got {len(gradients)} gradients for {len(parameters)} parameters.")
        return gradients
    return grad(rewrite_pregrad(loss_or_gradients), list(parameters))


def state_for(parameter: Parameter, slot: str, fill_value: float = 0.0) -> Parameter:
    """
    Allocate an optimizer-state shared variable shaped and typed like ``parameter``.

    The variable is named ``"{parameter.name}/{slot}"`` so it can be matched by name at serialization
    boundaries. The name is never used to *find* the variable at runtime — callers hold the returned object
    directly.

    Parameters
    ----------
    parameter : shared tensor variable
        The parameter this state accompanies. Its value's shape and dtype define the state's.
    slot : str
        A short role tag for the slot, e.g. ``"adam/first_moment"`` or ``"trace/velocity"``.
    fill_value : float
        Constant to initialize the state with. Default 0.0.

    Returns
    -------
    shared tensor variable
        A freshly allocated state variable.
    """
    value = parameter.get_value(borrow=True)
    name = f"{parameter.name}/{slot}" if parameter.name is not None else slot
    return pytensor.shared(np.full_like(value, fill_value), name=name)


def counter(name: str) -> Parameter:
    """Allocate an int64 step counter shared variable initialized to zero."""
    return pytensor.shared(np.asarray(0, dtype="int64"), name=name)


def chain(*transforms: Transform) -> Transform:
    """
    Compose transforms left to right into a single transform.

    The returned transform threads the updates dict through each argument in turn, giving the optax-style
    ``chain(clip_by_global_norm(...), trace(...), scale(...))`` surface over the underlying pure functions.

    Parameters
    ----------
    *transforms : Transform
        Transforms to apply in order.

    Returns
    -------
    Transform
        A transform that applies each input transform in sequence.
    """

    def combined(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        for transform in transforms:
            updates = transform(updates, parameters)
        return updates

    return combined
