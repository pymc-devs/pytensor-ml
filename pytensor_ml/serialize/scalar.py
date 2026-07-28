import importlib

from pytensor.scalar.basic import Cast, Composite, ScalarOp, get_scalar_type

from pytensor_ml.serialize.base import op_to_json, register_from_json


def _canonical_scalar_instances() -> dict[str, ScalarOp]:
    """Index pytensor's module-level ScalarOp singletons by class name.

    They are class-identified and carry an unpicklable ``output_types_preference``, so they are rebuilt from
    pytensor's own instances rather than by calling the class. Cast is per-dtype, hence its own rule below.
    """
    instances: dict[str, ScalarOp] = {}
    for module_name in ("pytensor.scalar.basic", "pytensor.scalar.math"):
        module = importlib.import_module(module_name)
        for name in dir(module):
            candidate = getattr(module, name)
            if isinstance(candidate, ScalarOp) and not isinstance(candidate, Composite | Cast):
                instances.setdefault(type(candidate).__name__, candidate)
    return instances


_SCALAR_INSTANCES = _canonical_scalar_instances()


@op_to_json.register(ScalarOp)
def _scalar_to_json(op: ScalarOp) -> dict:
    return {"family": "scalar", "type": type(op).__name__}


@register_from_json("scalar")
def _scalar_from_json(op_dict: dict) -> ScalarOp:
    try:
        return _SCALAR_INSTANCES[op_dict["type"]]
    except KeyError:
        raise NotImplementedError(f"Unregistered scalar op: {op_dict['type']!r}") from None


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
