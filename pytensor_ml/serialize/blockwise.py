from pytensor.tensor.blockwise import Blockwise

from pytensor_ml.serialize.base import op_from_json, op_to_json, register_from_json

# A Blockwise just vectorizes its core_op over batch dimensions; the signature is derived from core_op and
# is not serialized.


@op_to_json.register(Blockwise)
def _blockwise_to_json(op: Blockwise) -> dict:
    return {"family": "blockwise", "core_op": op_to_json(op.core_op)}


@register_from_json("blockwise")
def _blockwise_from_json(op_dict: dict) -> Blockwise:
    return Blockwise(op_from_json(op_dict["core_op"]))
