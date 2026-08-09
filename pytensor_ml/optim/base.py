from collections.abc import Callable, Sequence
from contextvars import ContextVar
from functools import wraps

import numpy as np
import pytensor

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.gradient import DisconnectedInputError, DisconnectedType, grad
from pytensor.tensor import TensorVariable
from pytensor.tensor.sharedvar import TensorSharedVariable

from pytensor_ml.pytensorf import rewrite_pregrad

type Parameter = TensorSharedVariable

# What every rule accepts first: either a scalar loss to differentiate, or gradients already computed.
type LossOrGradients = TensorVariable | Sequence[TensorVariable]

# Pytensor's native `updates` contract, and the single currency every rule and transform here speaks: it
# carries the next parameter values *and* the next optimizer-state values in one identity-keyed dict.
Updates = dict[SharedVariable, TensorVariable]

# A chainable transform reads an updates dict and returns a new one, working in "step space" (updates[p] - p).
Transform = Callable[[Updates, Sequence[Parameter]], Updates]

UpdateRule = Callable[[LossOrGradients, Sequence[Parameter]], Updates]

# A learning-rate schedule: symbolic step count in, scalar learning rate out.
type Schedule = Callable[[TensorVariable], TensorVariable]

# A rate a rule multiplies into its step: a baked-in constant, a shared variable to steer from Python with
# `set_value` or to substitute a schedule into, or any scalar graph, which is what a schedule reading a
# training clock produces.
type Rate = float | Parameter | TensorVariable

# What an optimizer alias accepts as its rate, adding a schedule that drives it on-graph.
type LearningRate = Rate | Schedule


def get_gradients(
    loss_or_gradients: LossOrGradients,
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

    Raises
    ------
    DisconnectedInputError
        If the loss has no gradient with respect to one of ``parameters``, naming the ones it cannot reach.
    """
    if isinstance(loss_or_gradients, list | tuple):
        gradients = list(loss_or_gradients)
        if len(gradients) != len(parameters):
            raise ValueError(f"Got {len(gradients)} gradients for {len(parameters)} parameters.")
        return gradients

    loss = rewrite_pregrad(loss_or_gradients)
    try:
        return grad(loss, list(parameters))  # type: ignore[return-value]
    except DisconnectedInputError as error:
        unreachable = _unreachable_parameter_names(loss, parameters)
        if not unreachable:
            raise
        raise DisconnectedInputError(
            f"The loss has no gradient with respect to {unreachable}. Leave them out of `parameters`, or "
            "check that the loss is meant to depend on them: a term that differentiates away, such as an "
            "output bias under a second derivative, is the usual cause."
        ) from error


def _unreachable_parameter_names(
    loss: TensorVariable, parameters: Sequence[Parameter]
) -> list[str]:
    """Name the parameters the loss carries no gradient signal to, which pytensor's own error omits."""
    gradients = grad(
        loss, list(parameters), disconnected_inputs="ignore", return_disconnected="disconnected"
    )
    return [
        parameter.name or str(parameter)
        for parameter, gradient in zip(parameters, gradients)  # type: ignore[arg-type]
        if isinstance(gradient.type, DisconnectedType)
    ]


# A per-parameter slot, or the bare name of a rule-wide counter.
type _StateKey = tuple[Parameter, str] | str

# Bound by reuses_state for the duration of one rule invocation; None means "allocate fresh".
_state_buffers: ContextVar[dict[_StateKey, Parameter] | None] = ContextVar(
    "optimizer_state_buffers", default=None
)


def reuses_state[**P, R](builds_updates: Callable[P, R]) -> Callable[P, R]:
    """
    Give ``builds_updates`` a private set of optimizer-state buffers, reused on every invocation.

    A configured rule such as ``adam(1e-3)`` reads as a value, so it is natural to compile two training
    functions from one. Without this, each invocation allocates fresh momentum under the *same* derived
    name: the two steps then share parameters but not optimizer state, which is silently wrong at runtime
    and raises only later when both are checkpointed together. The same holds for a composed transform,
    whose state is likewise allocated per call.

    The buffers are keyed per wrapped callable rather than globally so two independently configured
    optimizers stay independent. They are bound dynamically because :func:`state_for` is reached several
    call layers below, and threading a cache down would touch every one of them. Nesting is safe: an inner
    scope restores the outer one on exit, so a rule's own buffers and its enclosing chain's coexist.

    Parameters
    ----------
    builds_updates : callable
        A rule or transform to wrap. Its buffers live as long as the wrapper does.
    """
    buffers: dict[_StateKey, Parameter] = {}

    @wraps(builds_updates)
    def with_persistent_state(*args: P.args, **kwargs: P.kwargs) -> R:
        token = _state_buffers.set(buffers)
        try:
            return builds_updates(*args, **kwargs)
        finally:
            _state_buffers.reset(token)

    return with_persistent_state


def _reuse_or_allocate(key: _StateKey, allocate: Callable[[], Parameter]) -> Parameter:
    buffers = _state_buffers.get()
    if buffers is None:
        return allocate()
    if key not in buffers:
        buffers[key] = allocate()
    return buffers[key]


def state_for(parameter: Parameter, slot: str, fill_value: float = 0.0) -> Parameter:
    """
    Return the optimizer-state shared variable shaped and typed like ``parameter``.

    The variable is named ``"{parameter.name}/{slot}"`` so it can be matched by name at serialization
    boundaries. The name is never used to *find* the variable at runtime — callers hold the returned object
    directly, and reuse within a rule is keyed on the parameter object, so two same-named parameters still
    get distinct buffers and collide loudly at save time rather than silently sharing.

    Allocates unless the enclosing rule was wrapped in :func:`reuses_state` and already holds this slot.

    Parameters
    ----------
    parameter : shared tensor variable
        The parameter this state accompanies. Its value's shape and dtype define the state's.
    slot : str
        A short role tag for the slot, e.g. ``"adam/first_moment"`` or ``"trace/velocity"``.
    fill_value : float
        Constant to initialize the state with. Default 0.0.
    """
    if parameter.name is None:
        raise ValueError(
            f"Cannot allocate optimizer state {slot!r} for an unnamed parameter. Stateful optimizers rely on "
            "parameter names to identify their state at serialization boundaries; give the parameter a name."
        )

    def allocate() -> Parameter:
        value = parameter.get_value(borrow=True)
        return pytensor.shared(np.full_like(value, fill_value), name=f"{parameter.name}/{slot}")

    return _reuse_or_allocate((parameter, slot), allocate)


def counter(name: str) -> Parameter:
    """Return an int64 step counter shared variable initialized to zero, reused across invocations of a
    rule wrapped in :func:`reuses_state` so its step count keeps advancing."""
    return _reuse_or_allocate(
        name, lambda: pytensor.shared(np.asarray(0, dtype="int64"), name=name)
    )


def scalar_state(name: str, fill_value: float = 0.0) -> Parameter:
    """
    Return a floatX scalar shared variable, reused across invocations of a rule wrapped in
    :func:`reuses_state`.

    Parameters
    ----------
    name : str
        Name of the variable, used to match it at serialization boundaries.
    fill_value : float
        Value to initialize it with. Default 0.0.
    """
    return _reuse_or_allocate(
        name,
        lambda: pytensor.shared(np.asarray(fill_value, dtype=pytensor.config.floatX), name=name),
    )


def require_unique_state_names(updates: Updates) -> None:
    """
    Raise if two distinct shared variables in ``updates`` share a name.

    Optimizer state is matched by name at serialization boundaries, so two buffers with the same name would
    silently alias each other on save or restore. Runtime is unaffected — the updates dict is keyed by object
    identity — so this guards only the serialization contract.

    Parameters
    ----------
    updates : Updates
        The assembled updates dict whose shared-variable keys are checked.
    """
    seen: set[str] = set()
    for variable in updates:
        name = variable.name
        if name is None:
            continue
        if name in seen:
            raise ValueError(
                f"Two distinct shared variables share the name {name!r}. Optimizer state is matched by name "
                "at serialization boundaries and would collide; ensure parameters have unique names."
            )
        seen.add(name)


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

    @reuses_state
    def combined(updates: Updates, parameters: Sequence[Parameter]) -> Updates:
        for transform in transforms:
            updates = transform(updates, parameters)
        return updates

    return combined
