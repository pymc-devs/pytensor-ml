from pytensor.tensor.elemwise import CAReduce, Elemwise

from pytensor_ml.serialize.base import (
    op_from_json,
    op_to_json,
    prop_from_json,
    prop_to_json,
    qualname,
    register_from_json,
    resolve_class,
)

# The math an Elemwise applies lives entirely in its scalar_op; inplace_pattern is a compilation artifact
# added by rewrites and is deliberately dropped.


@op_to_json.register(Elemwise)
def _elemwise_to_json(op: Elemwise) -> dict:
    return {"family": "elemwise", "scalar_op": op_to_json(op.scalar_op)}


@register_from_json("elemwise")
def _elemwise_from_json(op_dict: dict) -> Elemwise:
    return Elemwise(op_from_json(op_dict["scalar_op"]))


# A CAReduce subclass (Sum, Max, Prod, ...) hardwires its reduction scalar_op, so the reduction is fully
# defined by the axis. The result dtype is re-inferred from the inputs on reconstruction.
@op_to_json.register(CAReduce)
def _careduce_to_json(op: CAReduce) -> dict:
    return {"family": "careduce", "type": qualname(op), "axis": prop_to_json(op.axis)}


@register_from_json("careduce")
def _careduce_from_json(op_dict: dict):
    return resolve_class(op_dict["type"])(axis=prop_from_json(op_dict["axis"]))
