import importlib

from pytensor.scalar.basic import Cast, Composite, ScalarOp, get_scalar_type

from pytensor_ml.serialize.base import op_to_json, register_from_json

# Most scalar ops are singletons identified by class (one Tanh, one Add) and carry an unpicklable
# output_types_preference function, so they are rebuilt from their canonical module-level instances rather
# than by class-call. Cast is the exception: it is parameterized by its target dtype, with one instance per
# dtype, so it gets its own rule below.
_SCALAR_INSTANCES: dict[str, ScalarOp] = {}
for _module_name in ("pytensor.scalar.basic", "pytensor.scalar.math"):
    _module = importlib.import_module(_module_name)
    for _name in dir(_module):
        _obj = getattr(_module, _name)
        if isinstance(_obj, ScalarOp) and not isinstance(_obj, Composite | Cast):
            _SCALAR_INSTANCES.setdefault(type(_obj).__name__, _obj)


@op_to_json.register(ScalarOp)
def _scalar_to_json(op: ScalarOp) -> dict:
    return {"family": "scalar", "type": type(op).__name__}


@register_from_json("scalar")
def _scalar_from_json(op_dict: dict) -> ScalarOp:
    try:
        return _SCALAR_INSTANCES[op_dict["type"]]
    except KeyError:
        raise NotImplementedError(f"Unregistered scalar op: {op_dict['type']!r}")


@op_to_json.register(Cast)
def _cast_to_json(op: Cast) -> dict:
    return {"family": "scalar_cast", "dtype": op.o_type.dtype}


@register_from_json("scalar_cast")
def _cast_from_json(op_dict: dict) -> Cast:
    return Cast(get_scalar_type(op_dict["dtype"]))


@op_to_json.register(Composite)
def _composite_to_json(op: Composite) -> dict:
    # Composite is both a ScalarOp and a HasInnerGraph; register it explicitly so dispatch is unambiguous.
    # Fused Composites only appear after compilation rewrites, never in the graphs we serialize, so defer.
    raise NotImplementedError(
        "Composite serialization is deferred (only appears in compiled graphs)."
    )
