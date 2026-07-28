from pytensor.compile.builders import OpFromGraph, SymbolicOp

from pytensor_ml.serialize.base import (
    graph_from_json,
    graph_to_json,
    leaf_to_json,
    op_to_json,
    props_from_json,
    props_to_json,
    qualname,
    register_from_json,
    resolve_class,
)

# A SymbolicOp (Softmax, every pytensor_ml LayerOp, ...) is a named op implemented as an OpFromGraph but
# fully defined by its __props__ and input types -- it regenerates its own inner graph -- so serialize it as
# a leaf. A plain OpFromGraph is an opaque captured subgraph, so serialize its inner graph recursively.


@op_to_json.register(SymbolicOp)
def _symbolic_op_to_json(op: SymbolicOp) -> dict:
    # Registered so the more general OpFromGraph rule below does not claim SymbolicOp, which subclasses it.
    return leaf_to_json(op)


@op_to_json.register(OpFromGraph)
def _ofg_to_json(op: OpFromGraph) -> dict:
    return {
        "family": "inner_graph",
        "type": qualname(op),
        "inline": bool(op.is_inline),
        "name": op.name,
        "props": props_to_json(op),
        "inner": graph_to_json(op.inner_inputs, op.inner_outputs),
    }


@register_from_json("inner_graph")
def _ofg_from_json(op_dict: dict):
    cls = resolve_class(op_dict["type"])
    inputs, outputs = graph_from_json(op_dict["inner"])
    return cls(
        inputs,
        outputs,
        inline=op_dict["inline"],
        name=op_dict["name"],
        **props_from_json(op_dict["props"]),
    )
