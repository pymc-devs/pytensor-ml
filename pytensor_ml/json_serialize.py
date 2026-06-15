import importlib

from collections.abc import Callable, Sequence
from functools import singledispatch

import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.op import Op
from pytensor.graph.traversal import io_toposort
from pytensor.graph.type import Type
from pytensor.scalar.basic import ScalarType
from pytensor.scalar.basic import constant as scalar_constant
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomGeneratorType
from pytensor.tensor.type import TensorType
from pytensor.tensor.type_other import NoneConst, NoneTypeT


def type_to_json(graph_type: Type) -> dict:
    if isinstance(graph_type, TensorType):
        return {"kind": "tensor", "dtype": graph_type.dtype, "shape": list(graph_type.shape)}
    if isinstance(graph_type, ScalarType):
        return {"kind": "scalar", "dtype": graph_type.dtype}
    if isinstance(graph_type, RandomGeneratorType):
        return {"kind": "random_generator"}
    if isinstance(graph_type, NoneTypeT):
        return {"kind": "none"}
    raise TypeError(f"Unserializable type: {graph_type!r}")


def type_from_json(type_dict: dict):
    if type_dict["kind"] == "tensor":
        return TensorType(type_dict["dtype"], tuple(type_dict["shape"]))
    if type_dict["kind"] == "scalar":
        return ScalarType(type_dict["dtype"])
    if type_dict["kind"] == "random_generator":
        return RandomGeneratorType()
    if type_dict["kind"] == "none":
        return NoneTypeT()
    raise ValueError(f"Unknown type kind: {type_dict['kind']!r}")


def prop_to_json(value):
    if isinstance(value, tuple):
        return {"__tuple__": [prop_to_json(element) for element in value]}
    if isinstance(value, slice):
        return {"__slice__": [prop_to_json(part) for part in (value.start, value.stop, value.step)]}
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    raise TypeError(f"Unserializable op prop: {value!r} ({type(value).__name__})")


def prop_from_json(value):
    if isinstance(value, dict):
        if "__tuple__" in value:
            return tuple(prop_from_json(element) for element in value["__tuple__"])
        if "__slice__" in value:
            return slice(*(prop_from_json(part) for part in value["__slice__"]))
    return value


def _encode_nonfinite(value):
    """Replace inf/-inf/nan floats with sentinels. JSON has no literals for them, so a constant such as
    the causal mask's -inf would serialize to a non-standard ``-Infinity`` token that strict, portable
    JSON parsers reject."""
    if isinstance(value, list):
        return [_encode_nonfinite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"__float__": "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")}
    return value


def _decode_nonfinite(value):
    if isinstance(value, list):
        return [_decode_nonfinite(item) for item in value]
    if isinstance(value, dict) and "__float__" in value:
        return float(value["__float__"])
    return value


def const_to_json(constant: Constant) -> dict:
    if isinstance(constant.type, NoneTypeT):
        return {"type": {"kind": "none"}}
    value = _encode_nonfinite(np.asarray(constant.data).tolist())
    return {"type": type_to_json(constant.type), "value": value}


def const_from_json(const_dict: dict):
    graph_type = type_from_json(const_dict["type"])
    if isinstance(graph_type, NoneTypeT):
        return NoneConst
    value = np.asarray(_decode_nonfinite(const_dict["value"]), dtype=graph_type.dtype)
    # Use the type-specific constant wrappers: a raw Constant holds an unhashable ndarray and breaks the
    # FrozenApply interning that reconstruction relies on.
    if isinstance(graph_type, ScalarType):
        return scalar_constant(value.item(), dtype=graph_type.dtype)
    return pt.constant(value, dtype=graph_type.dtype)


@singledispatch
def op_to_json(op: Op) -> dict:
    """Serialize an op to a JSON dict, dispatching on op type. The default handles a leaf op from its
    JSON-native ``__props__``; structural ops register their own rules in ``pytensor_ml.serialize``."""
    return {
        "family": "leaf",
        "type": _qualname(op),
        "props": {name: prop_to_json(getattr(op, name)) for name in getattr(op, "__props__", ())},
    }


# The reverse of op_to_json: a registry keyed by the "family" tag each forward handler emits.
_FROM_JSON: dict[str, Callable[[dict], Op]] = {}


def register_from_json(family: str) -> Callable[[Callable[[dict], Op]], Callable[[dict], Op]]:
    """Register the handler that rebuilds an op from a JSON dict tagged with ``family``."""

    def register(handler: Callable[[dict], Op]) -> Callable[[dict], Op]:
        _FROM_JSON[family] = handler
        return handler

    return register


def op_from_json(op_dict: dict) -> Op:
    """Rebuild an op from its JSON dict, dispatching on the ``family`` tag."""
    return _FROM_JSON[op_dict["family"]](op_dict)


@register_from_json("leaf")
def _leaf_from_json(op_dict: dict):
    cls = _resolve_class(op_dict["type"])
    return cls(**{name: prop_from_json(value) for name, value in op_dict["props"].items()})


def _qualname(op: Op) -> str:
    return f"{type(op).__module__}.{type(op).__name__}"


def _resolve_class(path: str):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def graph_to_json(inputs: Sequence[Variable], outputs: Sequence[Variable]) -> dict:
    """Serialize a graph to a dict of input types, op nodes, and output references."""
    nodes = io_toposort(inputs, outputs)
    var_ref: dict[int, dict] = {id(inp): {"input": index} for index, inp in enumerate(inputs)}
    for node_index, node in enumerate(nodes):
        for output_index, out in enumerate(node.outputs):
            var_ref[id(out)] = {"node": node_index, "out": output_index}

    def make_ref(variable: Variable) -> dict:
        existing = var_ref.get(id(variable))
        if existing is not None:
            return existing
        if isinstance(variable, Constant):
            return {"const": const_to_json(variable)}
        raise ValueError(f"Unresolved variable while serializing graph: {variable!r}")

    return {
        "inputs": [type_to_json(inp.type) for inp in inputs],
        "nodes": [
            {
                "op": op_to_json(node.op),
                "inputs": [make_ref(inp) for inp in node.inputs],
                "outputs": [type_to_json(out.type) for out in node.outputs],
            }
            for node in nodes
        ],
        "outputs": [make_ref(out) for out in outputs],
    }


def graph_from_json(
    graph_dict: dict, inputs: Sequence[Variable] | None = None
) -> tuple[list[Variable], list[Variable]]:
    """Rebuild a graph from :func:`graph_to_json` output, onto ``inputs`` if given, else fresh leaves."""
    if inputs is None:
        inputs = [type_from_json(type_dict)() for type_dict in graph_dict["inputs"]]
    else:
        inputs = list(inputs)
    built: dict[tuple[int, int], Variable] = {}

    def resolve_ref(reference: dict):
        if "input" in reference:
            return inputs[reference["input"]]
        if "node" in reference:
            return built[(reference["node"], reference["out"])]
        if "const" in reference:
            return const_from_json(reference["const"])
        raise ValueError(f"Bad reference: {reference!r}")

    for node_index, node in enumerate(graph_dict["nodes"]):
        op = op_from_json(node["op"])
        node_inputs = [resolve_ref(reference) for reference in node["inputs"]]
        # A RandomVariable's __call__ takes distribution params in a different order than its node inputs, so
        # reconstruction goes through make_node. Other ops use __call__, which (for OpFromGraph/SymbolicOp)
        # also builds the inner fgraph that make_node would assume already exists.
        if isinstance(op, RandomVariable):
            node_outputs = op.make_node(*node_inputs).outputs
        else:
            result = op(*node_inputs)
            node_outputs = list(result) if isinstance(result, list | tuple) else [result]
        for output_index, out in enumerate(node_outputs):
            built[(node_index, output_index)] = out

    return list(inputs), [resolve_ref(reference) for reference in graph_dict["outputs"]]


def serialize_graph(inputs: Sequence[Variable], outputs: Sequence[Variable]) -> dict:
    """
    Serialize a pytensor graph to a JSON-native dict.

    Parameters
    ----------
    inputs : sequence of Variable
        The graph's inputs, in order. Include every shared variable the graph reads, since the serialized
        form addresses inputs positionally and carries no values.
    outputs : sequence of Variable
        The graph's outputs.

    Returns
    -------
    dict
        A JSON-native description of the graph's structure (no parameter values).
    """
    return graph_to_json(list(inputs), list(outputs))


def deserialize_graph(
    graph_dict: dict, inputs: Sequence[Variable] | None = None
) -> tuple[list[Variable], list[Variable]]:
    """
    Rebuild a pytensor graph from :func:`serialize_graph` output.

    Parameters
    ----------
    graph_dict : dict
        A graph description produced by :func:`serialize_graph`.
    inputs : sequence of Variable, optional
        Input leaves to build the graph on, in the original order. Supply these to control input identity —
        for example to reattach named shared variables. Fresh, plain symbolic variables are created when
        omitted.

    Returns
    -------
    inputs : list of Variable
        The input variables, in the original order.
    outputs : list of Variable
        The rebuilt graph outputs.
    """
    return graph_from_json(graph_dict, inputs)


# Importing this module populates the op-family dispatch registry. Kept last so the family modules can
# import the names defined above.
from pytensor_ml import serialize as _serialize  # noqa: E402, F401
