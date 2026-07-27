from pytensor.scan.op import Scan, ScanInfo

from pytensor_ml.json_serialize import (
    graph_from_json,
    graph_to_json,
    op_to_json,
    prop_from_json,
    prop_to_json,
    register_from_json,
)

# Scan is just an inner step graph plus ScanInfo (integer tap bookkeeping) and a few flags. The inner graph
# reuses the graph codec; ScanInfo is entirely JSON-native.
_SCAN_INFO_FIELDS = tuple(ScanInfo.__annotations__)


@op_to_json.register(Scan)
def _scan_to_json(op: Scan) -> dict:
    return {
        "family": "scan",
        "inner": graph_to_json(op.inner_inputs, op.inner_outputs),
        "info": {field: prop_to_json(getattr(op.info, field)) for field in _SCAN_INFO_FIELDS},
        "truncate_gradient": op.truncate_gradient,
        "name": op.name,
        "allow_gc": op.allow_gc,
        "strict": op.strict,
    }


@register_from_json("scan")
def _scan_from_json(scan_dict: dict) -> Scan:
    inner_inputs, inner_outputs = graph_from_json(scan_dict["inner"])
    info = ScanInfo(**{field: prop_from_json(value) for field, value in scan_dict["info"].items()})
    return Scan(
        inner_inputs,
        inner_outputs,
        info,
        truncate_gradient=scan_dict["truncate_gradient"],
        name=scan_dict["name"],
        allow_gc=scan_dict["allow_gc"],
        strict=scan_dict["strict"],
    )
