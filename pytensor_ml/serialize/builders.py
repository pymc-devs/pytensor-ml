from pytensor.compile.builders import OpFromGraph, SymbolicOp

from pytensor_ml.json_serialize import (
    _qualname,
    _resolve_class,
    graph_from_json,
    graph_to_json,
    op_to_json,
    prop_from_json,
    prop_to_json,
    register_from_json,
)

# A SymbolicOp (Softmax, every pytensor_ml LayerOp, ...) is a named op implemented as an OpFromGraph but
# fully defined by its __props__ and input types -- it regenerates its own inner graph -- so serialize it as
# a leaf. A plain OpFromGraph is an opaque captured subgraph, so serialize its inner graph recursively.


@op_to_json.register(SymbolicOp)
def _symbolic_op_to_json(op: SymbolicOp) -> dict:
    return {
        "family": "leaf",
        "type": _qualname(op),
        "props": {name: prop_to_json(getattr(op, name)) for name in getattr(op, "__props__", ())},
    }


@op_to_json.register(OpFromGraph)
def _ofg_to_json(op: OpFromGraph) -> dict:
    return {
        "family": "inner_graph",
        "type": _qualname(op),
        "inline": bool(op.is_inline),
        "name": op.name,
        "props": {name: prop_to_json(getattr(op, name)) for name in getattr(op, "__props__", ())},
        "inner": graph_to_json(op.inner_inputs, op.inner_outputs),
    }


@register_from_json("inner_graph")
def _ofg_from_json(op_dict: dict):
    cls = _resolve_class(op_dict["type"])
    inputs, outputs = graph_from_json(op_dict["inner"])
    props = {name: prop_from_json(value) for name, value in op_dict["props"].items()}
    return cls(inputs, outputs, inline=op_dict["inline"], name=op_dict["name"], **props)
