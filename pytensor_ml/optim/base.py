from collections.abc import Callable, Sequence
from contextvars import ContextVar
from functools import wraps
from typing import overload

import numpy as np
import pytensor

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.gradient import DisconnectedInputError, grad
from pytensor.graph.basic import Variable
from pytensor.graph.op import io_connection_pattern
from pytensor.tensor import TensorVariable
from pytensor.tensor.sharedvar import TensorSharedVariable

from pytensor_ml.params import step_counter
from pytensor_ml.pytensorf import rewrite_pregrad

type Parameter = TensorSharedVariable

# What every rule accepts first: either a scalar loss to differentiate, or gradients already computed.
type LossOrGradients = TensorVariable | Sequence[TensorVariable]

# Pytensor's native `updates` contract, and the single currency every rule and transform here speaks: it
# carries the next parameter values *and* the next optimizer-state values in one identity-keyed dict.
Updates = dict[SharedVariable, TensorVariable]

Transform = Callable[[Updates, Sequence[Parameter]], Updates]
"""
A chainable step transformer, reading an updates dict and returning a new one.

Transforms work in step space -- ``updates[parameter] - parameter`` -- so one can be written without
knowing which rule produced the step it is adjusting.

Examples
--------
Write one as a plain function and :func:`chain` accepts it wherever a built-in transform goes. The
updates dict also carries optimizer state and training clocks, so touch only the entries for
``parameters`` -- rewriting the rest would halve a clock's advance as readily as a step:

.. code-block:: python

    import numpy as np

    from pytensor_ml.layers import Input, Linear
    from pytensor_ml.loss import SquaredError, supervised_loss
    from pytensor_ml.optim import adam, chain, compile_train


    def halve_every_step(updates, parameters):
        halved = dict(updates)
        for parameter in parameters:
            halved[parameter] = parameter + 0.5 * (updates[parameter] - parameter)
        return halved


    X = Input("X", shape=(None, 4))
    loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError(), ndim_out=2)

    step = compile_train(loss, chain(adam(1e-3), halve_every_step))
    loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
"""

UpdateRule = Callable[[LossOrGradients, Sequence[Parameter]], Updates]
"""
What every optimizer is: a callable taking a loss (or gradients) and the parameters, returning the
updates dict that moves them.

Examples
--------
Anything matching the signature is a rule, so a hand-written one composes with the rest of the module:

.. code-block:: python

    import numpy as np

    from pytensor_ml.layers import Input, Linear
    from pytensor_ml.loss import SquaredError, supervised_loss
    from pytensor_ml.optim import chain, clip_by_global_norm, compile_train, get_gradients


    def plain_descent(loss_or_gradients, parameters):
        gradients = get_gradients(loss_or_gradients, parameters)
        return {p: p - 0.01 * gradient for p, gradient in zip(parameters, gradients)}


    X = Input("X", shape=(None, 4))
    loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError(), ndim_out=2)

    step = compile_train(loss, chain(plain_descent, clip_by_global_norm(1.0)))
    loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
"""

type Schedule = Callable[[TensorVariable], TensorVariable]
"""
A learning-rate schedule: symbolic step count in, scalar learning rate out.

Examples
--------
The built-in schedules return one, and any callable of the same shape works in their place:

.. code-block:: python

    import pytensor.tensor as pt

    from pytensor_ml.optim import adam


    def inverse_square_root(step_count):
        return 3e-4 / pt.sqrt(pt.maximum(step_count, 1))


    rule = adam(learning_rate=inverse_square_root)
"""

type Rate = float | Parameter | TensorVariable
"""
A rate a rule multiplies into its step.

Either a baked-in constant, a shared variable to steer from Python with ``set_value`` or to substitute a
schedule into, or any scalar graph, which is what a schedule reading a training clock produces.

Examples
--------
A shared scalar is the form to reach for when the rate has to change mid-run without recompiling:

.. code-block:: python

    import numpy as np

    from pytensor_ml.layers import Input, Linear
    from pytensor_ml.loss import SquaredError, supervised_loss
    from pytensor_ml.optim import compile_train, scalar_state, sgd

    rate = scalar_state("rate", fill_value=0.1)

    X = Input("X", shape=(None, 4))
    loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError(), ndim_out=2)

    step = compile_train(loss, sgd(learning_rate=rate))
    loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))

    rate.set_value(np.array(0.01, dtype=rate.dtype))
"""

type LearningRate = Rate | Schedule
"""
What an optimizer alias accepts as its rate, adding a schedule that drives it on-graph.

Examples
--------
Every alias takes either form, so a constant can be swapped for a schedule without touching anything
else:

.. code-block:: python

    from pytensor_ml.optim import adam, cosine_schedule

    fixed = adam(learning_rate=3e-4)
    scheduled = adam(learning_rate=cosine_schedule(3e-4, total_steps=10_000))
"""


def to_floatx(value: Rate) -> Rate:
    """
    Return ``value`` at the current ``floatX``, casting only a variable stored at something else.

    A shared variable carries whatever dtype it was allocated with, which need not be the ``floatX`` the
    graph is built under -- restoring a checkpoint into a differently configured session is the ordinary
    way to get there. A learning rate is where it bites: a float64 rate in a float32 graph makes an update
    pytensor refuses, and the error names the parameter rather than the rate behind it.

    A plain number is left alone rather than made into an array, which is where this differs from pymc's
    ``floatX``: a rule given a float literal must build exactly the graph it built before. ``astype``
    already returns the variable itself when the dtype matches, so a well typed graph is untouched too.

    Parameters
    ----------
    value : float or TensorVariable
        A scalar a rule is about to build into its step.

    Examples
    --------
    Cast a rate to ``floatX`` so it cannot silently upcast a float32 graph to float64. A symbolic rate, such
    as one a schedule produced, passes through unchanged:

    .. code-block:: python

        from pytensor_ml.optim import to_floatx

        rate = to_floatx(1e-3)
    """
    return value.astype(pytensor.config.floatX) if isinstance(value, Variable) else value


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
    gradients : list of TensorVariable
        One gradient per parameter, in the order of ``parameters``.

    Examples
    --------
    Take gradients of a loss with respect to the parameters, or pass gradients straight through. Every rule
    calls it first, so a rule can be handed gradients you computed yourself:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import get_gradients
        from pytensor_ml.pytensorf import collect_trainable_params

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError(), ndim_out=2)

        parameters = collect_trainable_params(loss)
        gradients = get_gradients(loss, parameters)
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
    connection_pattern = io_connection_pattern(list(parameters), [loss])
    return [
        parameter.name or str(parameter)
        for parameter, to_the_loss in zip(parameters, connection_pattern)
        if not any(to_the_loss)
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
    """Return the training clock a rule counts its own steps on, reused across invocations of a rule wrapped
    in :func:`reuses_state` so the count keeps advancing.

    A :class:`~pytensor_ml.params.StepCounter` rather than a plain shared variable, so a schedule can read
    the same notion of time the rule uses, and :func:`~pytensor_ml.pytensorf.collect_clock_updates` advances
    it for a caller who does not write the advance themselves.
    """
    return _reuse_or_allocate(name, lambda: step_counter(name))


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

    Examples
    --------
    Build the scalar a rule or policy keeps between steps. Naming it makes it findable in a checkpoint and
    in a printed graph:

    .. code-block:: python

        from pytensor_ml.optim import scalar_state

        scale = scalar_state("plateau/scale", fill_value=1.0)
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


@overload
def chain(head: UpdateRule, *rest: Transform) -> UpdateRule: ...


@overload
def chain(head: Transform, *rest: Transform) -> Transform: ...


def chain(head, *rest: Transform):
    """
    Compose an update rule or transform with the transforms that follow it, left to right.

    The head decides what the result is. Given a rule, the result is a rule: it differentiates the loss and
    threads the updates through each transform, so ``adam`` then a clip then a scale is one value to pass to
    :func:`~pytensor_ml.optim.train.compile_train` or to wrap in a guard. Given a transform, the result is a
    transform, composing in step space for a rule to be pointed at later.

    .. code-block:: python

        rule = chain(adam(1e-3), clip_by_global_norm(1.0), scale(0.5))
        step = compile_train(loss, rule)

        post_process = chain(clip_by_global_norm(1.0), scale(0.5))
        step = compile_train(loss, chain(adam(1e-3), post_process))

    The composed callable owns one set of optimizer-state buffers however many times it is invoked, so two
    training functions compiled from one chain share its momentum rather than each allocating their own.

    Parameters
    ----------
    head : UpdateRule or Transform
        What runs first, and what the result is. A rule such as ``adam(1e-3)`` reads a loss; a transform
        reads an updates dict.
    *rest : Transform
        Transforms applied in order to whatever the head produces.

    Returns
    -------
    chained : UpdateRule or Transform
        A callable matching the head, applying every argument in sequence.

    Examples
    --------
    Compose a rule with the transforms that follow it. The head decides the result: given a rule, the
    whole chain is a rule, applied left to right:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, chain, clip_by_global_norm, compile_train, scale

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError(), ndim_out=2)

        step = compile_train(loss, chain(adam(1e-3), clip_by_global_norm(1.0), scale(0.5)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    @reuses_state
    def combined(loss_gradients_or_updates, parameters: Sequence[Parameter]) -> Updates:
        updates = head(loss_gradients_or_updates, parameters)
        for transform in rest:
            updates = transform(updates, parameters)
        return updates

    return combined
