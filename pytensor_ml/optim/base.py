from collections.abc import Callable, Sequence

import numpy as np
import pytensor

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.gradient import DisconnectedInputError, grad
from pytensor.graph.basic import Variable
from pytensor.graph.op import io_connection_pattern
from pytensor.tensor import TensorVariable
from pytensor.tensor.sharedvar import TensorSharedVariable

from pytensor_ml.params import StepCounter, TrainableParameter, step_counter
from pytensor_ml.pytensorf import rewrite_pregrad

type Parameter = TensorSharedVariable

# What every rule accepts first: either a scalar loss to differentiate, or gradients already computed.
type LossOrGradients = TensorVariable | Sequence[TensorVariable]


class Updates(dict[SharedVariable, TensorVariable]):
    """
    Pytensor's native ``updates`` contract, and the single currency every transform here speaks.

    Carries the next parameter values *and* the next optimizer-state values in one identity-keyed
    mapping, so a step and the momentum that produced it travel together. Every transform reads what it
    needs as ``updates[parameter] - parameter`` and writes back a new value for the parameter, which is
    what lets one be written without knowing what produced its input.

    What that difference *means* depends on where in a chain the transform sits, so the two positions are
    distinguished by :class:`Gradients` and :class:`Steps` rather than by a bare mapping. Write a result
    with :meth:`replacing` rather than ``|``, which returns a bare ``dict`` and would silently widen a
    transform's own output back to an unplaced mapping.
    """

    def replacing(self, changes: dict[SharedVariable, TensorVariable]) -> "Updates":
        """
        Return these updates with ``changes`` written over them, in the same space.

        Parameters
        ----------
        changes : dict mapping shared variable to TensorVariable
            New values to write, overriding any entry already present for the same variable.

        Returns
        -------
        updates : Updates
            A new updates dict of the same class, so a transform's output stays placed.
        """
        return type(self)({**self, **changes})

    def copy(self) -> "Updates":
        return type(self)(self)


class Gradients(Updates):
    r"""
    Updates carrying gradients: ``updates[parameter] - parameter`` is the gradient :math:`g` itself.

    What :func:`to_updates` produces from a loss, and what everything ahead of the first rule in a chain
    sees. A clip placed here bounds the gradient itself, so a spike never reaches the moment estimates.
    """


class Steps(Updates):
    """
    Updates carrying steps: ``updates[parameter] - parameter`` is the move a rule decided on.

    What every rule returns, and what everything after it in a chain sees. A clip placed here bounds the
    step an adaptive rule already normalized, which is a different and usually weaker guarantee.
    """


# What every transform accepts first. A loss or gradients seed a fresh `Gradients`; an updates dict from
# an earlier stage passes through as whatever it already is. A bare dict is admitted because a
# hand-written transform is free to build one.
type LossGradientsOrUpdates = LossOrGradients | dict[SharedVariable, TensorVariable]

Transform = Callable[[LossGradientsOrUpdates, Sequence[Parameter]], Updates]
"""
What every optimizer, clip, and schedule in this module is: a callable taking a loss, gradients, or an
updates dict, along with the parameters, and returning the updates dict that moves them.

One type covers all of them: ``adam(1e-3)`` and ``clip_by_global_norm(1.0)`` share this signature and so
compose in either order, and :func:`chain` folds them left to right. What distinguishes them is only what
each does to the difference it reads, and position decides whether that difference is a gradient or a
step.

Examples
--------
Write one as a plain function and :func:`chain` accepts it wherever a built-in transform goes. The
updates dict also carries optimizer state and training clocks, so touch only the entries for
``parameters`` -- rewriting the rest would halve a clock's advance as readily as a step.

One that keeps state of its own allocates it with :func:`state_for` and takes a ``namespace``, so that
two of them in one chain are told apart at the serialization boundary. Every invocation allocates afresh.
The updates dict it returns holds the state, so two functions that share state are compiled from one dict.

.. code-block:: python

    import numpy as np

    from pytensor_ml.layers import Input, Linear
    from pytensor_ml.loss import SquaredError, supervised_loss
    from pytensor_ml.optim import adam, chain, compile_train, to_updates


    def halve_every_step(loss_gradients_or_updates, parameters):
        updates = to_updates(loss_gradients_or_updates, parameters)
        halved = updates.copy()
        for parameter in parameters:
            halved[parameter] = parameter + 0.5 * (updates[parameter] - parameter)
        return halved


    X = Input("X", shape=(None, 4))
    loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

    step = compile_train(loss, chain(adam(1e-3), halve_every_step))
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
    loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

    step = compile_train(loss, sgd(learning_rate=rate))
    loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))

    rate.set_value(np.array(0.01, dtype=rate.dtype))
"""

type LearningRate = Rate | Schedule
"""
What any ``learning_rate`` accepts: a rate, or a schedule the rule reads off its own training clock.

Examples
--------
Every rule and alias takes either form, so a constant can be swapped for a schedule without touching
anything else:

.. code-block:: python

    from pytensor_ml.optim import adam, cosine_schedule

    fixed = adam(learning_rate=3e-4)
    scheduled = adam(learning_rate=cosine_schedule(3e-4, total_steps=10_000))
"""


def rate_on(learning_rate: LearningRate, clock: StepCounter) -> Rate:
    """
    Read a schedule off ``clock``. Any other rate passes through untouched.

    Parameters
    ----------
    learning_rate : LearningRate
        A rate, or a schedule of the step count.
    clock : StepCounter
        The training clock a schedule is evaluated at, ordinarily the one the rule counts its own steps on.

    Returns
    -------
    rate : Rate
        The rate as a number or a scalar graph.
    """
    return learning_rate(clock) if callable(learning_rate) else learning_rate


def read_rate(learning_rate: LearningRate, namespace: str) -> tuple[Rate, Updates]:
    """
    Resolve a rate at ``floatX``, reading a schedule off a clock of the caller's own.

    The clock's advance comes back as an update for the caller to write, so the updates dict carries the
    clock and a checkpoint taken from the dict resumes the schedule where it left off. A plain rate reads
    no clock, so none is allocated and nothing comes back to write.

    Parameters
    ----------
    learning_rate : LearningRate
        A rate, or a schedule of the step count.
    namespace : str
        Prefix of the clock a schedule reads, allocated as ``"{namespace}/step_count"``.

    Returns
    -------
    rate : Rate
        The rate as a number or a scalar graph.
    clock_update : Updates
        The clock's one-step advance when a schedule was read, empty otherwise.
    """
    if not callable(learning_rate):
        return to_floatx(learning_rate), Updates()
    clock = step_counter(f"{namespace}/step_count")
    return to_floatx(learning_rate(clock)), Updates({clock: clock.advance()})


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
    value : float, numpy scalar, or TensorVariable
        A scalar a rule is about to build into its step. A numpy scalar becomes a Python float, which
        pytensor folds into the graph at the graph's own dtype, where a ``np.float64`` would widen it.

    Examples
    --------
    Cast a rate to ``floatX`` so it cannot silently upcast a float32 graph to float64. A symbolic rate, such
    as one a schedule produced, passes through unchanged:

    .. code-block:: python

        from pytensor_ml.optim import to_floatx

        rate = to_floatx(1e-3)
    """
    if isinstance(value, Variable):
        return value.astype(pytensor.config.floatX)
    if isinstance(value, np.generic | np.ndarray):
        return float(value)
    return value


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
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

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


def to_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
) -> Updates:
    r"""
    Return ``loss_gradients_or_updates`` as an updates dict, differentiating a loss if that is what it is.

    The first line of every transform, which is what lets one accept a loss, gradients, or an earlier
    stage's output through a single argument. A loss or a list of gradients seeds a fresh
    :class:`Gradients` as :math:`\{p: p + g\}`, so the gradient :math:`g` is recoverable as
    ``updates[parameter] - parameter`` by exactly the arithmetic a transform already does to read a step.

    An updates dict is returned as the *same object*, not a copy, so a transform must write its result
    with :meth:`Updates.replacing` or into a :meth:`Updates.copy` rather than assigning into what this
    returns -- mutating it in place would reach back into the dict the previous stage still holds.

    The sign is positive rather than negative so that a bound written for a step means the same thing
    written for a gradient: ``clip_by_value(-0.1, 0.1)`` clips :math:`g` into that interval, not
    :math:`-g`.

    Parameters
    ----------
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        A scalar loss to differentiate, precomputed gradients one per parameter, or an updates dict an
        earlier transform produced.
    parameters : sequence of shared tensor variable
        Parameters the updates are keyed by, in the order gradients are given in.

    Returns
    -------
    updates : Updates
        A loss or gradients as a new :class:`Gradients`, a bare dict as an unplaced :class:`Updates`, and
        an updates dict as itself, keeping whichever space it already carries.

    Examples
    --------
    Open a hand-written transform with it and the transform composes in any position, reading gradients
    at the front of a chain and steps behind a rule, with no branch of its own:

    .. code-block:: python

        from pytensor_ml.optim import to_updates


        def halve(loss_gradients_or_updates, parameters):
            updates = to_updates(loss_gradients_or_updates, parameters)
            halved = updates.copy()
            for parameter in parameters:
                halved[parameter] = parameter + 0.5 * (updates[parameter] - parameter)
            return halved
    """
    if isinstance(loss_gradients_or_updates, Updates):
        return loss_gradients_or_updates
    if isinstance(loss_gradients_or_updates, dict):
        # A hand-written transform is free to build a bare dict, which says nothing about where it sits.
        # Leave it unplaced rather than guessing: the checks that read the space reject a definite mismatch
        # only, so an unplaced mapping passes every one of them rather than tripping the wrong one.
        return Updates(loss_gradients_or_updates)

    gradients = get_gradients(loss_gradients_or_updates, parameters)
    return Gradients(
        {parameter: parameter + gradient for parameter, gradient in zip(parameters, gradients)}
    )


def gradients_to_descend(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    rule_name: str,
) -> tuple[Updates, list[TensorVariable]]:
    """
    Return the updates a rule was handed and the gradients it descends along.

    The opening line of every rule. Raises when handed :class:`Steps`, which a rule cannot use: it negates
    what it reads, so descending along a step another rule already chose would move the parameters uphill.

    Parameters
    ----------
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Whatever the rule was called with.
    parameters : sequence of shared tensor variable
        Parameters to read gradients for, in the order the result is returned in.
    rule_name : str
        The rule's own name, used to say which one was misplaced.

    Returns
    -------
    incoming : Updates
        The input as an updates dict, carrying any optimizer state an earlier transform wrote.
    gradients : list of TensorVariable
        One gradient per parameter, in the order of ``parameters``.
    """
    incoming = to_updates(loss_gradients_or_updates, parameters)
    if isinstance(incoming, Steps):
        raise ValueError(
            f"{rule_name} was given the step another rule already produced, rather than gradients. A rule "
            "descends along what it reads, so it would negate that step and move the parameters uphill. "
            "Keep one rule in a chain and shape its step with `scale`, `trace`, or a clip after it."
        )
    gradients = steps_of(incoming, parameters)
    if isinstance(incoming, Gradients):
        # A gradient for a parameter this rule does not descend cannot travel on inside a Steps dict: it
        # would compile as `p + g`, an ascent. Another rule over that parameter reads it from the same
        # Gradients dict this one did, so dropping it here loses nothing.
        own_parameters = set(parameters)
        incoming = Gradients(
            {
                key: value
                for key, value in incoming.items()
                if key in own_parameters or not isinstance(key, TrainableParameter)
            }
        )
    return incoming, gradients


def steps_of(updates: Updates, parameters: Sequence[Parameter]) -> list[TensorVariable]:
    """
    Return the amount each parameter's entry moves it by.

    A gradient or a step according to which space ``updates`` carries; see :class:`Gradients` and
    :class:`Steps`.

    Parameters
    ----------
    updates : Updates
        The updates dict to read.
    parameters : sequence of shared tensor variable
        Parameters to read, in the order the result is returned in.

    Returns
    -------
    steps : list of TensorVariable
        ``updates[parameter] - parameter``, one per parameter.
    """
    return [updates[parameter] - parameter for parameter in parameters]


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


def state_for(
    parameter: Parameter, slot: str, fill_value: float = 0.0, history_size: int | None = None
) -> Parameter:
    """
    Return the optimizer-state shared variable typed like ``parameter``, or a stack of them.

    The variable is named ``"{parameter.name}/{slot}"`` and carries the parameter's layer, so a checkpoint
    numbers it where it numbers the parameter. The name is never used to *find* the variable at runtime --
    callers hold the returned object directly, and reuse within a rule is keyed on the parameter object, so
    two same-named parameters still get distinct buffers rather than silently sharing one.

    A fresh variable on every call. The updates dict a rule returns holds it, so two training functions
    that share this state are compiled from one updates dict.

    Parameters
    ----------
    parameter : shared tensor variable
        The parameter this state accompanies. Its value's shape and dtype define the state's.
    slot : str
        A short role tag for the slot, e.g. ``"adam/first_moment"`` or ``"trace/velocity"``.
    fill_value : float
        Constant to initialize the state with. Default 0.0.
    history_size : int, optional
        Number of past values to stack along a new leading axis, so the state is shaped
        ``(history_size, *parameter.shape)``. Omitted, the state has the parameter's own shape.

    Returns
    -------
    state : shared tensor variable
        A new buffer for this slot.

    Examples
    --------
    Allocate the buffer a stateful transform keeps between steps. The slot's namespace is what keeps two
    transforms of the same kind from writing to one buffer:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.optim import state_for
        from pytensor_ml.params import trainable

        weight = trainable(np.zeros(4), name="fc/W")
        velocity = state_for(weight, "trace/velocity")
    """
    if parameter.name is None:
        raise ValueError(
            f"Cannot allocate optimizer state {slot!r} for an unnamed parameter. Stateful optimizers rely on "
            "parameter names to identify their state at serialization boundaries; give the parameter a name."
        )
    if history_size is not None and history_size < 1:
        raise ValueError(f"history_size must be at least 1, got {history_size}.")

    value = parameter.get_value(borrow=True)
    shape = value.shape if history_size is None else (history_size, *value.shape)
    static_shape = (
        parameter.type.shape if history_size is None else (history_size, *parameter.type.shape)
    )
    # The declared dtype rather than the value's: after a step on mlx the value is a device array
    # whose dtype numpy cannot read.
    state = pytensor.shared(
        np.full(shape, fill_value, dtype=parameter.type.dtype),
        name=f"{parameter.name}/{slot}",
        shape=static_shape,
    )
    # Keeps `Linear_1_W` and `Linear_1_W/adam/first_moment` numbered onto the same layer.
    state.layer_name = getattr(parameter, "layer_name", None)
    return state


def scalar_state(name: str, fill_value: float = 0.0, dtype: str | None = None) -> Parameter:
    """
    Allocate a scalar shared variable, at ``floatX`` unless told otherwise.

    Parameters
    ----------
    name : str
        Name of the variable, used to match it at serialization boundaries.
    fill_value : float
        Value to initialize it with. Default 0.0.
    dtype : str, optional
        Storage dtype. Default ``floatX``. A count belongs in an integer dtype.

    Examples
    --------
    Build the scalar a rule or policy keeps between steps. Naming it makes it findable in a checkpoint and
    in a printed graph:

    .. code-block:: python

        from pytensor_ml.optim import scalar_state

        scale = scalar_state("plateau/scale", fill_value=1.0)
    """
    return pytensor.shared(np.asarray(fill_value, dtype=dtype or pytensor.config.floatX), name=name)


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
                f"Two distinct shared variables share the name {name!r}. Optimizer state is matched by "
                "name at serialization boundaries, so the two would collide there. Two transforms of the "
                "same kind in one chain are the usual cause: give one of them a `namespace` of its own. "
                "Otherwise two parameters share a name, and one of them needs a different one."
            )
        seen.add(name)


def chain(*transforms: Transform) -> Transform:
    """
    Compose transforms left to right, each reading what the one before it produced.

    Every argument has the same type, so a clip composes ahead of a rule as readily as behind it, and the
    two mean different things. Ahead of the rule the clip sees gradients, so a spike is bounded before it
    reaches the moment estimates; behind it the clip sees the step the rule already decided on, which an
    adaptive rule has normalized to roughly its learning rate whatever the gradient was.

    .. code-block:: python

        stop_the_spike = chain(clip_by_global_norm(1.0), adam(1e-3))
        bound_the_move = chain(adam(1e-3), clip_by_global_norm(1.0))

    A chain is itself a transform, so one composes into another and the result is flat.

    Parameters
    ----------
    *transforms : Transform
        Applied in order. The first reads whatever the chain is called with -- a loss, gradients, or an
        updates dict -- and each one after it reads the previous one's output.

    Returns
    -------
    chained : Transform
        A transform applying every argument in sequence.

    Examples
    --------
    Clip the gradients before the rule sees them, which is what bounds an exploding gradient rather than
    the step it produced:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, chain, clip_by_global_norm, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, chain(clip_by_global_norm(1.0), adam(1e-3)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))

    Put a transform after the rule to act on the step instead, which is where a rate or a decay belongs:

    .. code-block:: python

        from pytensor_ml.optim import adam, chain, clip_by_global_norm, scale

        rule = chain(clip_by_global_norm(1.0), adam(1.0), scale(1e-3))
    """
    if not transforms:
        raise ValueError("chain needs at least one transform.")

    def combined(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        updates = transforms[0](loss_gradients_or_updates, parameters)
        for transform in transforms[1:]:
            updates = transform(updates, parameters)
        return updates

    return combined
