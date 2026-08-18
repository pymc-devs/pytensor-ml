from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pytensor.tensor as pt

from pytensor.scan import scan
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.activations import Activation, Sigmoid, Tanh
from pytensor_ml.base import Layer
from pytensor_ml.params import trainable
from pytensor_ml.state import (
    Initializer,
    OrthogonalInitializer,
    XavierNormalInitializer,
    ZeroInitializer,
)


class RecurrentCell(ABC):
    """
    One step of a recurrence, for :class:`Recurrent` to scan over a sequence.

    A cell owns the parameters its step uses and knows the shape of the state it carries, which is all
    the loop needs from it. Subclasses implement :meth:`step` and :meth:`initial_state`.
    """

    @abstractmethod
    def step(self, x_t: TensorVariable, *state: TensorVariable) -> tuple[TensorVariable, ...]:
        """
        Advance the state by one timestep.

        Parameters
        ----------
        x_t : TensorVariable
            The sequence at this step, shape ``(..., n_in)``.
        *state : TensorVariable
            The state carried out of the previous step, as :meth:`initial_state` laid it out.

        Returns
        -------
        tuple of TensorVariable
            The new state, in the same order. Its first element is the cell's output, which is what
            :class:`Recurrent` stacks over time -- an LSTM carrying :math:`(h, c)` returns :math:`h` first.
        """

    @abstractmethod
    def initial_state(self, X: TensorVariable) -> tuple[TensorVariable, ...]:
        """
        Build the state the recurrence starts from, for a sequence ``X`` of shape ``(..., time, n_in)``.

        Carries one value per batch element, so the state's batch axes are ``X``'s and its dtype is the
        one :meth:`step` produces -- a float64 sequence through a float32 cell makes a float64 state.

        Returns
        -------
        tuple of TensorVariable
            Zero-filled state, in the order :meth:`step` takes and returns it.
        """


class Recurrent(Layer):
    """
    Scan a :class:`RecurrentCell` over the time axis of a sequence.

    Time is the second-to-last axis and everything before it is a batch axis, so the input is
    ``(..., time, n_in)`` and the output ``(..., time, n_out)``, one cell output per step. Slice the last
    step off the result -- ``out[..., -1, :]`` -- for the sequence-classification case; pytensor's
    ``scan_save_mem`` rewrite sees that the earlier steps are unused and stops storing them.

    Parameters
    ----------
    cell : RecurrentCell
        The step to run at each timestep.
    name : str or None
        Name for the layer's output and its scan. Defaults to "Recurrent" when None.
    """

    def __init__(self, cell: RecurrentCell, name: str | None = None):
        self.cell = cell
        self.name = name if name else "Recurrent"

    def __call__(
        self,
        X: pt.TensorLike,
        initial_state: pt.TensorLike | Sequence[TensorVariable] | None = None,
    ) -> TensorVariable:
        """
        Run the cell over ``X`` and return its output at every step.

        Parameters
        ----------
        X : TensorVariable
            Input sequence, shape ``(..., time, n_in)``.
        initial_state : TensorVariable or sequence of TensorVariable, optional
            The state the recurrence starts from, matching what the cell carries. The cell's zero state
            when omitted.

        Returns
        -------
        TensorVariable
            The cell's output at each step, shape ``(..., time, n_out)``.
        """
        X = pt.as_tensor(X)
        if X.ndim < 2:
            raise ValueError(
                f"{self.name} takes a sequence of shape (..., time, n_in), but got a "
                f"{X.ndim}-dimensional input, which has no time axis to recur over."
            )

        zero_state = self.cell.initial_state(X)
        if initial_state is None:
            state = zero_state
        else:
            if isinstance(initial_state, list | tuple):
                state = tuple(pt.as_tensor(tensor) for tensor in initial_state)
            else:
                state = (pt.as_tensor(initial_state),)
            self._check_state_against(state, zero_state, X)

        # Scan iterates the leading axis, so time moves to the front and back again on the way out.
        # Not strict: the step closes over the cell's parameters and scan lifts them in. A generator
        # captured that way has no update, which `collect_default_updates` refuses.
        state_sequence = scan(
            self.cell.step,
            sequences=[pt.moveaxis(X, -2, 0)],
            outputs_info=list(state),
            name=f"{self.name}_recurrence",
            return_updates=False,
        )

        # Scan hands back a bare variable for a single carried state and a list for several. The cell's
        # output is the first one either way.
        output = state_sequence[0] if isinstance(state_sequence, list) else state_sequence

        out = pt.moveaxis(output, 0, -2)
        out.name = f"{self.name}_output"
        return out

    def _check_state_against(
        self,
        given: Sequence[TensorVariable],
        expected: Sequence[TensorVariable],
        X: TensorVariable,
    ) -> None:
        """Reject a starting state the cell would not have built, before scan reports it from inside."""
        if len(given) != len(expected):
            raise ValueError(
                f"{self.name}'s cell carries {len(expected)} state tensor(s), but got {len(given)}."
            )
        for position, (state, zero) in enumerate(zip(given, expected)):
            if state.ndim != zero.ndim:
                raise ValueError(
                    f"{self.name} starts from a state carrying the same batch axes as its input, so a "
                    f"{X.ndim}-dimensional input needs a {zero.ndim}-dimensional state at position "
                    f"{position}; got a {state.ndim}-dimensional one."
                )


class ElmanCell(RecurrentCell):
    r"""
    The step of an Elman recurrence, which updates its state from the step's input and the previous state:

    .. math::

        h_t = \phi\left(x_t W_{ih} + b + h_{t-1} W_{hh}\right),

    where :math:`\phi` is the activation.

    Parameters
    ----------
    name : str or None
        Name prefix for the cell's parameters. Defaults to "ElmanCell" when None.
    n_in : int
        Size of the input feature axis.
    n_hidden : int
        Size of the hidden state.
    activation : Activation, optional
        Applied to each step's pre-activation. Default is :class:`~pytensor_ml.activations.Tanh`, which
        bounds the state and so keeps the recurrence from running away over a long sequence.
    bias : bool, optional
        Add the learned shift :math:`b`. One bias covers the step, rather than torch's separate
        ``b_ih`` and ``b_hh``, whose sum is the only thing the step can distinguish. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W_{ih}` is drawn. Xavier normal when omitted.
    recurrent_initializer : Initializer, optional
        How :math:`W_{hh}` is drawn. It meets the state once per step, so its singular values compound
        over the sequence: at one the state's norm survives any length, and spread around one the
        gradient explodes along some directions while vanishing along others. Orthogonal when omitted,
        as in keras and flax.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted.
    """

    def __init__(
        self,
        name: str | None,
        n_in: int,
        n_hidden: int,
        activation: Activation | None = None,
        bias: bool = True,
        *,
        weight_initializer: Initializer | None = None,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        self.name = name if name else "ElmanCell"
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation = activation if activation is not None else Tanh()
        self.bias = bias

        # Held directly rather than as a nested Linear: the projection runs inside the recurrence, and a
        # layer op there would bury its matmul in an inner graph where the scan rewrites cannot see it.
        self.W_ih = _trainable_parameter(
            f"{self.name}_W_ih", (n_in, n_hidden), weight_initializer, XavierNormalInitializer()
        )
        if bias:
            self.b = _trainable_parameter(
                f"{self.name}_b", (n_hidden,), bias_initializer, ZeroInitializer()
            )
        self.W_hh = _trainable_parameter(
            f"{self.name}_W_hh",
            (n_hidden, n_hidden),
            recurrent_initializer,
            OrthogonalInitializer(),
        )

    def step(self, x_t: TensorVariable, *state: TensorVariable) -> tuple[TensorVariable, ...]:
        (h_prev,) = state
        pre_activation = x_t @ self.W_ih + h_prev @ self.W_hh
        if self.bias:
            pre_activation = pre_activation + self.b
        return (self.activation(pre_activation),)

    def initial_state(self, X: TensorVariable) -> tuple[TensorVariable, ...]:
        return (_zero_state(X, self.n_hidden, self.W_ih, self.W_hh),)


class RNN(Recurrent):
    r"""
    Elman recurrent layer over a sequence: a :class:`Recurrent` scanning an :class:`ElmanCell`.

    Takes the cell's arguments directly, for the common case where a network wants a plain recurrence
    and no cell of its own. The parameters live on the cell, as ``rnn.cell.W_ih``. See
    :class:`ElmanCell` for the recurrence itself and :class:`Recurrent` for the axes.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "RNN" when None.
    n_in : int
        Size of the input feature axis.
    n_hidden : int
        Size of the hidden state.
    activation : Activation, optional
        Applied to each step's pre-activation. Default is :class:`~pytensor_ml.activations.Tanh`.
    bias : bool, optional
        Add the learned shift :math:`b`. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W_{ih}` is drawn. Xavier normal when omitted.
    recurrent_initializer : Initializer, optional
        How :math:`W_{hh}` is drawn. Orthogonal when omitted; see :class:`ElmanCell` for why that is the
        draw an RNN is most sensitive to.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted.
    """

    def __init__(
        self,
        name: str | None,
        n_in: int,
        n_hidden: int,
        activation: Activation | None = None,
        bias: bool = True,
        *,
        weight_initializer: Initializer | None = None,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        name = name if name else "RNN"
        super().__init__(
            ElmanCell(
                name,
                n_in,
                n_hidden,
                activation,
                bias,
                weight_initializer=weight_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
            ),
            name=name,
        )


class GRUCell(RecurrentCell):
    r"""
    The step of a gated recurrent unit, which interpolates between the previous state and a candidate:

    .. math::

        r_t &= \sigma\left(x_t W_{ir} + b_r + h_{t-1} W_{hr}\right) \\
        z_t &= \sigma\left(x_t W_{iz} + b_z + h_{t-1} W_{hz}\right) \\
        n_t &= \phi\left(x_t W_{in} + b_n + r_t \odot \left(h_{t-1} W_{hn} + c\right)\right) \\
        h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1},

    where :math:`\phi` is the activation and :math:`\sigma` the gate activation. The update gate
    :math:`z` decides how much of the previous state survives the step, so a unit holding :math:`z` near
    one carries its value across the whole sequence and the gradient reaches the start of it; the reset
    gate :math:`r` decides how much of that state the candidate is allowed to see.

    The three gates share one projection of the input and one of the state, so a step is two matmuls
    rather than six. :math:`r` multiplies the state's projection after that matmul rather than before
    it -- as torch, flax and keras' ``reset_after`` all do -- which is what allows the fused form.

    Parameters
    ----------
    name : str or None
        Name prefix for the cell's parameters. Defaults to "GRUCell" when None.
    n_in : int
        Size of the input feature axis.
    n_hidden : int
        Size of the hidden state.
    activation : Activation, optional
        Applied to the candidate state. Default is :class:`~pytensor_ml.activations.Tanh`.
    gate_activation : Activation, optional
        Applied to the reset and update gates. Only a squashing function makes :math:`h_t` an
        interpolation; a gate ranging outside :math:`[0, 1]` extrapolates past both the previous state
        and the candidate. Default is :class:`~pytensor_ml.activations.Sigmoid`.
    bias : bool, optional
        Add the learned shifts :math:`b` and :math:`c`. One bias per gate covers the input and state
        projections together, rather than torch's separate pair, whose sum is the only thing a gate can
        distinguish. The candidate's state projection is the exception and carries its own :math:`c`:
        :math:`r_t` scales it, so it moves independently of :math:`b_n`. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W_{ih}` is drawn. Xavier normal when omitted.
    recurrent_initializer : Initializer, optional
        How :math:`W_{hh}` is drawn, across all three gates at once, as in keras. Orthogonal when
        omitted; see :class:`ElmanCell` for why the state's own weight is the sensitive draw.
    bias_initializer : Initializer, optional
        How :math:`b` and :math:`c` are drawn. Zeros when omitted.
    """

    def __init__(
        self,
        name: str | None,
        n_in: int,
        n_hidden: int,
        activation: Activation | None = None,
        bias: bool = True,
        *,
        gate_activation: Activation | None = None,
        weight_initializer: Initializer | None = None,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        self.name = name if name else "GRUCell"
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation = activation if activation is not None else Tanh()
        self.gate_activation = gate_activation if gate_activation is not None else Sigmoid()
        self.bias = bias

        self.W_ih = _trainable_parameter(
            f"{self.name}_W_ih", (n_in, 3 * n_hidden), weight_initializer, XavierNormalInitializer()
        )
        self.W_hh = _trainable_parameter(
            f"{self.name}_W_hh",
            (n_hidden, 3 * n_hidden),
            recurrent_initializer,
            OrthogonalInitializer(),
        )
        if bias:
            self.b = _trainable_parameter(
                f"{self.name}_b", (3 * n_hidden,), bias_initializer, ZeroInitializer()
            )
            self.c = _trainable_parameter(
                f"{self.name}_c", (n_hidden,), bias_initializer, ZeroInitializer()
            )

    def step(self, x_t: TensorVariable, *state: TensorVariable) -> tuple[TensorVariable, ...]:
        (h_prev,) = state

        from_input = x_t @ self.W_ih
        if self.bias:
            from_input = from_input + self.b
        from_state = h_prev @ self.W_hh

        input_r, input_z, input_n = _split_gates(from_input, self.n_hidden)
        state_r, state_z, state_n = _split_gates(from_state, self.n_hidden)
        if self.bias:
            state_n = state_n + self.c

        reset = self.gate_activation(input_r + state_r)
        update = self.gate_activation(input_z + state_z)
        candidate = self.activation(input_n + reset * state_n)

        return ((1 - update) * candidate + update * h_prev,)

    def initial_state(self, X: TensorVariable) -> tuple[TensorVariable, ...]:
        return (_zero_state(X, self.n_hidden, self.W_ih, self.W_hh),)


class GRU(Recurrent):
    r"""
    Gated recurrent layer over a sequence: a :class:`Recurrent` scanning a :class:`GRUCell`.

    Takes the cell's arguments directly, for the common case where a network wants a plain recurrence
    and no cell of its own. The parameters live on the cell, as ``gru.cell.W_ih``. See :class:`GRUCell`
    for the recurrence itself and :class:`Recurrent` for the axes.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "GRU" when None.
    n_in : int
        Size of the input feature axis.
    n_hidden : int
        Size of the hidden state.
    activation : Activation, optional
        Applied to the candidate state. Default is :class:`~pytensor_ml.activations.Tanh`.
    gate_activation : Activation, optional
        Applied to the reset and update gates. Default is
        :class:`~pytensor_ml.activations.Sigmoid`; see :class:`GRUCell` for what a gate outside
        :math:`[0, 1]` does to the step.
    bias : bool, optional
        Add the learned shifts. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W_{ih}` is drawn. Xavier normal when omitted.
    recurrent_initializer : Initializer, optional
        How :math:`W_{hh}` is drawn. Orthogonal when omitted.
    bias_initializer : Initializer, optional
        How the biases are drawn. Zeros when omitted.
    """

    def __init__(
        self,
        name: str | None,
        n_in: int,
        n_hidden: int,
        activation: Activation | None = None,
        bias: bool = True,
        *,
        gate_activation: Activation | None = None,
        weight_initializer: Initializer | None = None,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        name = name if name else "GRU"
        super().__init__(
            GRUCell(
                name,
                n_in,
                n_hidden,
                activation,
                bias,
                gate_activation=gate_activation,
                weight_initializer=weight_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
            ),
            name=name,
        )


def _trainable_parameter(
    name: str, shape: tuple[int, ...], initializer: Initializer | None, default: Initializer
) -> TensorVariable:
    """Build a trainable parameter of ``shape``, drawn by ``initializer``, or by ``default`` if None."""
    chosen = default if initializer is None else initializer
    return trainable(chosen.initial_value(shape), name, initializer=chosen)


def _zero_state(X: TensorVariable, n_hidden: int, *parameters: TensorVariable) -> TensorVariable:
    """One zero state carrying ``X``'s batch axes, at the dtype ``X`` and ``parameters`` promote to."""
    state_dtype = np.result_type(X.dtype, *(parameter.dtype for parameter in parameters))
    return pt.zeros((*X.shape[:-2], n_hidden), dtype=state_dtype)


def _split_gates(
    projection: TensorVariable, n_hidden: int
) -> tuple[TensorVariable, TensorVariable, TensorVariable]:
    """Cut a stacked projection into its reset, update and candidate parts, in torch's gate order."""
    # Split rather than three slices: its gradient is one Join, where three slices would each
    # accumulate into a zero buffer.
    reset, update, candidate = pt.split(projection, [n_hidden] * 3, n_splits=3, axis=-1)
    return reset, update, candidate


__all__ = ["GRU", "RNN", "ElmanCell", "GRUCell", "Recurrent", "RecurrentCell"]
