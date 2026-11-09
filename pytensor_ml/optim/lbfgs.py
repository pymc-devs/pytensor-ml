from collections.abc import Sequence

import pytensor
import pytensor.tensor as pt

from pytensor.compile.builders import SymbolicOp
from pytensor.graph.basic import Variable
from pytensor.scalar import upcast
from pytensor.tensor import TensorVariable


class LBFGSDirection(SymbolicOp):
    r"""
    Multiply a gradient by the L-BFGS inverse-Hessian approximation that a ring-buffered memory defines.

    Inputs are ``count, gamma, rho, g_1..g_n, S_1..S_n, Y_1..Y_n`` and outputs are ``d_1..d_n = H g``, one
    per parameter. ``S_p`` and ``Y_p`` are ``(memory_size, *shape)`` stacks of past parameter differences
    :math:`s` and gradient differences :math:`y` for parameter ``p``, written as a ring: slot
    ``(count - 1) % memory_size`` holds the newest pair and ``count`` is the number of pairs written so
    far. ``rho`` is the ``(memory_size,)`` vector of each slot's curvature :math:`\rho_i = 1 / (y_i^\top
    s_i)`. A slot that holds nothing yet has zero stacks and a zero ``rho`` and contributes nothing to the
    recursion, and a writer retires a slot the same way. The op applies whatever pairs it is given: admitting only
    pairs with :math:`y^\top s > 0`, which keeps the approximation positive definite, is the writer's job.

    The product is the two-loop recursion, algorithm 7.4 of :cite:t:`nocedal2006numerical`. Each dot
    product sums over every parameter, so the memory of a model with several parameters is treated as one
    vector and never copied into one. Starting from :math:`\gamma I`,

    .. math::

        q &\leftarrow g \\
        \alpha_i &= \rho_i s_i^\top q, \quad q \leftarrow q - \alpha_i y_i \quad \text{newest to oldest} \\
        r &\leftarrow \gamma q \\
        \beta_i &= \rho_i y_i^\top r, \quad r \leftarrow r + (\alpha_i - \beta_i) s_i \quad \text{oldest to newest}

    and :math:`d = r`. Both loops run as a ``scan`` over the ring order, so the inner graph runs on any backend
    with a scan dispatch, and a backend without one registers its own implementation of this op.

    Parameters
    ----------
    n_parameters : int
        How many parameters the gradient and memory are split across.
    memory_size : int
        Number of slots in each memory stack.

    Examples
    --------
    Compile the direction for one vector parameter and a memory of four slots, with one pair written. The
    slot count of ``rho`` and the stacks has to be static:

    .. code-block:: python

        import pytensor
        import pytensor.tensor as pt

        from pytensor_ml.optim.lbfgs import LBFGSDirection

        g = pt.vector("g")
        rho = pt.tensor("rho", shape=(4,))
        S = pt.tensor("S", shape=(4, None))
        Y = pt.tensor("Y", shape=(4, None))
        d = LBFGSDirection(n_parameters=1, memory_size=4)(1, 1.0, rho, g, S, Y)
        direction = pytensor.function([rho, g, S, Y], d)

    References
    ----------
    The limited-memory update is from :cite:t:`liu1989lbfgs`.
    """

    __props__ = ("n_parameters", "memory_size")
    n_parameters: int
    memory_size: int

    def __init__(self, input_types=None, **kwargs):
        super().__init__(input_types, **kwargs)
        if self.n_parameters < 1:
            raise ValueError(f"n_parameters must be at least 1, got {self.n_parameters}.")
        if self.memory_size < 1:
            raise ValueError(f"memory_size must be at least 1, got {self.memory_size}.")

    @staticmethod
    def filter_inputs(*inputs: Variable | float | int) -> tuple[Variable, ...]:
        count, gamma, rho, *raw = inputs
        tensors = [pt.as_tensor_variable(tensor) for tensor in raw]
        # The curvatures and gamma come from cross-parameter dot products, so they live at the widest
        # parameter dtype.
        curvature_dtype = upcast(*(tensor.dtype for tensor in tensors))
        return (
            _scalar_at(count, "int64"),
            _scalar_at(gamma, curvature_dtype),
            pt.as_tensor_variable(rho).astype(curvature_dtype),
            *tensors,
        )

    def build_inner_graph(self, *inputs: TensorVariable) -> list[Variable]:
        n, m = self.n_parameters, self.memory_size
        count, gamma, rho, *tensors = inputs
        if len(tensors) != 3 * n:
            raise ValueError(
                f"LBFGSDirection with n_parameters={n} takes {3 * n} tensors after count, gamma and rho, "
                f"a gradient and two memory stacks per parameter, but got {len(tensors)}."
            )
        # The recursion walks exactly memory_size slots, so a slot count known only at runtime could
        # silently drop rows or index past the end.
        if rho.type.ndim != 1 or rho.type.shape[0] != m:
            raise ValueError(
                f"rho must be a vector of one curvature per slot, with a static length of "
                f"memory_size={m}, but got {rho.type}."
            )
        gradients = tensors[:n]
        S = tensors[n : 2 * n]
        Y = tensors[2 * n :]
        for index, (gradient, s, y) in enumerate(zip(gradients, S, Y)):
            for stack in (s, y):
                _require_stack_of(stack, gradient, m, index)

        order = (count + pt.arange(m)) % m

        def right_product(slot, *vector):
            s = [stack[slot] for stack in S]
            y = [stack[slot] for stack in Y]
            alpha = rho[slot] * flat_dot(s, vector)
            return [v - alpha.astype(v.dtype) * y_p for v, y_p in zip(vector, y)] + [alpha]

        *q, alphas = pytensor.scan(
            right_product,
            sequences=[order],
            outputs_info=[*gradients, None],
            go_backwards=True,
            return_updates=False,
        )
        r = [gamma.astype(v.dtype) * v[-1] for v in q]

        def left_product(slot, alpha, *vector):
            s = [stack[slot] for stack in S]
            y = [stack[slot] for stack in Y]
            beta = rho[slot] * flat_dot(y, vector)
            return [v + (alpha - beta).astype(v.dtype) * s_p for v, s_p in zip(vector, s)]

        # The backward loop reports its alphas newest first and the forward loop reads them oldest first.
        r = pytensor.scan(
            left_product,
            sequences=[order, alphas[::-1]],
            outputs_info=r,
            return_updates=False,
        )
        if n == 1:
            r = [r]
        return [v[-1] for v in r]


def _scalar_at(value: Variable | float | int, dtype: str) -> TensorVariable:
    """Return ``value`` as a scalar of ``dtype``, built at that dtype rather than cast to it when it is a
    literal, so no ``Cast`` node enters the graph for a Python number."""
    if isinstance(value, Variable):
        return pt.as_tensor_variable(value).astype(dtype)
    return pt.constant(value, dtype=dtype)


def _require_stack_of(
    stack: TensorVariable, gradient: TensorVariable, memory_size: int, index: int
) -> None:
    """Raise unless ``stack`` is a static ``memory_size`` slots of ``gradient``'s shape and dtype."""
    slots = stack.type.shape[0] if stack.type.ndim else None
    if (
        stack.type.ndim != gradient.type.ndim + 1
        or stack.type.dtype != gradient.type.dtype
        or slots != memory_size
    ):
        raise ValueError(
            f"The memory stacks of parameter {index} must be shaped (memory_size={memory_size}, "
            f"*gradient.shape) at the gradient's dtype, with the slot count static, but got "
            f"{stack.type} for a gradient of type {gradient.type}."
        )


def flat_dot(left: Sequence[TensorVariable], right: Sequence[TensorVariable]) -> TensorVariable:
    """Dot product of two lists of tensors read as one flat vector each, through BLAS under numba."""
    return pt.sum([pt.dot(a.ravel(), b.ravel()) for a, b in zip(left, right)])
