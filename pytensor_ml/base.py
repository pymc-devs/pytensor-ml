from typing import Protocol, runtime_checkable

from pytensor.compile.builders import SymbolicOp
from pytensor.tensor.variable import TensorVariable


class LayerOp(SymbolicOp):
    """Base class for the library's neural-network ops.

    A ``SymbolicOp`` is an ``OpFromGraph`` whose inner graph is rebuilt from its ``__props__`` and input
    types by :meth:`build_inner_graph`, so equal props with equal inputs yield an identical op. Basing the
    layers on it (rather than plain ``OpFromGraph``) is what lets the numba backend optimize each inner
    graph: ``SymbolicOp`` restores the fgraph-aware ``__eq__``/``__hash__`` that a props-carrying
    ``OpFromGraph`` would otherwise lose, so the ``ofg_inner_graph`` rewrite keeps its optimized inner
    graph instead of discarding it as unchanged.
    """

    __props__: tuple[str, ...] = ()


class UnaryLayerOp(LayerOp):
    """A ``LayerOp`` with exactly one output, typed as such.

    ``SymbolicOp.__call__`` is annotated ``Variable | list[Variable]`` because an op may produce many
    outputs; a unary layer op produces one, so narrow the result to ``TensorVariable``. The ``isinstance``
    guard narrows for the type checker without a cast and asserts the invariant at runtime.
    """

    def __call__(self, *inputs, **kwargs) -> TensorVariable:
        out = super().__call__(*inputs, **kwargs)
        assert isinstance(out, TensorVariable), f"{type(self).__name__} produced multiple outputs"
        return out


@runtime_checkable
class StatefulOp(Protocol):
    """An op that writes some of its outputs back to shared variables it takes as inputs, the way batch
    norm updates its running statistics. Defining :meth:`update_map` is what marks an op stateful -- the
    check is structural, so there is nothing to register."""

    def update_map(self) -> dict[int, int]:
        """Map each output index to the index of the input that output updates."""


__all__ = ["LayerOp", "StatefulOp", "UnaryLayerOp"]
