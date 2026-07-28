import json

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import pytensor

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Variable
from pytensor.tensor.random.type import RandomGeneratorType

from pytensor_ml.checkpoint import load_state, save_state
from pytensor_ml.json_serialize import deserialize_graph, serialize_graph, type_from_json
from pytensor_ml.params import NonTrainableParameter, TrainableParameter, non_trainable, trainable
from pytensor_ml.pytensorf import (
    as_output_list,
    collect_data_inputs,
    collect_shared_variables,
    find_rng_nodes,
)

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"

# Marks a config as a pytensor_ml graph (vs a HuggingFace config, which shares the config.json filename but
# is a hyperparameter sheet, not a serialized graph). The version guards future schema changes.
GRAPH_FORMAT = "pytensor_ml.graph"
GRAPH_FORMAT_VERSION = 3

Format = Literal["auto", "pytensor", "huggingface"]


class InputKind(StrEnum):
    """How a serialized graph input is rebuilt: as a data placeholder or a kind of shared variable."""

    DATA = "data"
    TRAINABLE = "trainable"
    NON_TRAINABLE = "non_trainable"
    SHARED = "shared"
    RNG = "rng"


def _looks_like_huggingface(config: dict) -> bool:
    return "model_type" in config or "architectures" in config


def _detect_format(config: dict) -> Format:
    if config.get("format") == GRAPH_FORMAT:
        return "pytensor"
    if _looks_like_huggingface(config):
        return "huggingface"
    raise ValueError("Unrecognized config: not a pytensor_ml graph or a HuggingFace model.")


def _weight_variables(outputs: Variable | Sequence[Variable]) -> list[SharedVariable]:
    # Random generators are excluded: their state rides in the config as JSON, not as a tensor.
    output_list = as_output_list(outputs)
    random_generators = set(find_rng_nodes(output_list))
    return [
        variable
        for variable in collect_shared_variables(output_list)
        if variable not in random_generators
    ]


def _input_kind(variable: Variable) -> InputKind:
    if isinstance(variable.type, RandomGeneratorType):
        return InputKind.RNG
    if isinstance(variable, TrainableParameter):
        return InputKind.TRAINABLE
    if isinstance(variable, NonTrainableParameter):
        return InputKind.NON_TRAINABLE
    if isinstance(variable, SharedVariable):
        return InputKind.SHARED
    return InputKind.DATA


def _input_meta(variable: Variable) -> dict:
    meta = {"name": variable.name, "kind": _input_kind(variable)}
    if isinstance(variable, SharedVariable) and meta["kind"] == InputKind.RNG:
        # Captured for exact reproducibility even though load does not restore it by default.
        meta["rng_state"] = variable.get_value(borrow=True).bit_generator.state
    return meta


def _rebuild_input(type_json: dict, meta: dict, restore_rng: bool):
    kind, name = meta["kind"], meta["name"]
    if kind == InputKind.DATA:
        return type_from_json(type_json)(name=name)
    if kind == InputKind.RNG:
        generator = np.random.default_rng()
        if restore_rng:
            generator.bit_generator.state = meta["rng_state"]
        return pytensor.shared(generator, name=name)

    graph_type = type_from_json(type_json)
    placeholder = np.zeros(graph_type.shape, dtype=graph_type.dtype)
    if kind == InputKind.TRAINABLE:
        return trainable(placeholder, name)
    if kind == InputKind.NON_TRAINABLE:
        return non_trainable(placeholder, name)
    return pytensor.shared(placeholder, name=name)


def save_network(
    outputs: Variable | Sequence[Variable],
    path: str | Path,
    *,
    inputs: Sequence[Variable] | None = None,
) -> None:
    """
    Serialize a network's architecture (its graph) to a JSON config file, without parameter values.

    Records each input's name and kind (data, trainable, non-trainable, or plain shared) so
    :func:`load_network` can rebuild the graph with the right variable identities. The data inputs are
    collected from ``outputs`` unless given explicitly.

    Parameters
    ----------
    outputs : Variable or sequence of Variable
        The network's output(s).
    path : str or pathlib.Path
        Destination JSON file.
    inputs : sequence of Variable, optional
        The network's data inputs, in call order. Collected from ``outputs`` when omitted; pass explicitly
        when call order matters.
    """
    output_list = as_output_list(outputs)
    data_inputs = list(inputs) if inputs is not None else collect_data_inputs(output_list)
    leaves = [*data_inputs, *collect_shared_variables(output_list)]

    config = {"format": GRAPH_FORMAT, "format_version": GRAPH_FORMAT_VERSION}
    config.update(serialize_graph(leaves, output_list))
    config["input_meta"] = [_input_meta(leaf) for leaf in leaves]
    config["n_outputs"] = len(output_list)
    Path(path).write_text(json.dumps(config))


def load_network(
    path: str | Path, *, restore_rng: bool = False
) -> tuple[list[Variable], Variable | list[Variable]]:
    """
    Rebuild a network's graph from a :func:`save_network` config file.

    Shared variables (parameters) come back zero-initialized and keep their original names and kinds, so a
    subsequent :func:`load_state` (or :func:`from_pretrained`) can fill them by name. This restores the
    architecture only.

    Parameters
    ----------
    path : str or pathlib.Path
        A config file written by :func:`save_network`.
    restore_rng : bool
        If True, restore each random generator to its saved state for exact reproducibility. By default a
        fresh generator is created, so stochastic layers draw new randomness. Default False.

    Returns
    -------
    inputs : list of Variable
        The network's data inputs, in call order.
    outputs : Variable or list of Variable
        The rebuilt output(s) -- a single variable when the network has one output, otherwise a list.
    """
    config = json.loads(Path(path).read_text())
    if config.get("format") != GRAPH_FORMAT:
        hint = " (this looks like a HuggingFace config)" if _looks_like_huggingface(config) else ""
        raise ValueError(f"{path} is not a pytensor_ml network config{hint}.")

    # Before rebuilding: op classes are recorded by import path, so a stale layout would otherwise fail
    # deep inside class resolution.
    written_version = config.get("format_version")
    if written_version != GRAPH_FORMAT_VERSION:
        raise ValueError(
            f"{path} is graph format version {written_version}, but this pytensor_ml reads version "
            f"{GRAPH_FORMAT_VERSION}. Rebuild the network and call save_network again."
        )

    leaves = [
        _rebuild_input(type_json, meta, restore_rng)
        for type_json, meta in zip(config["inputs"], config["input_meta"])
    ]
    _, outputs = deserialize_graph(config, inputs=leaves)

    data_inputs = [
        leaf for leaf, meta in zip(leaves, config["input_meta"]) if meta["kind"] == InputKind.DATA
    ]
    return data_inputs, (outputs[0] if config["n_outputs"] == 1 else outputs)


def save_pretrained(
    outputs: Variable | Sequence[Variable],
    directory: str | Path,
    *,
    inputs: Sequence[Variable] | None = None,
) -> None:
    """
    Save a complete network -- architecture and weights -- to a directory.

    Writes ``config.json`` (the graph) and ``model.safetensors`` (the parameter values), the layout
    :func:`from_pretrained` expects.

    Parameters
    ----------
    outputs : Variable or sequence of Variable
        The network's output(s).
    directory : str or pathlib.Path
        Destination directory, created if needed.
    inputs : sequence of Variable, optional
        The network's data inputs, in call order. Collected from ``outputs`` when omitted.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_network(outputs, directory / CONFIG_FILENAME, inputs=inputs)
    save_state(_weight_variables(outputs), directory / WEIGHTS_FILENAME)


def from_pretrained(
    directory: str | Path, source_format: Format = "auto", *, restore_rng: bool = False
) -> tuple[list[Variable], Variable | list[Variable]]:
    """
    Load a complete network -- architecture and weights -- from a directory.

    For a pytensor_ml directory, rebuilds the graph from ``config.json`` and fills its parameters from
    ``model.safetensors``. The format is detected from the config by default, since a pytensor_ml graph and
    a HuggingFace model share the same filenames but not the same schema.

    Parameters
    ----------
    directory : str or pathlib.Path
        A directory holding ``config.json`` and ``model.safetensors``.
    source_format : {'auto', 'pytensor', 'huggingface'}
        Which loader to use. ``'auto'`` detects the format from the config's marker. Default 'auto'.
    restore_rng : bool
        If True, restore each random generator to its saved state for exact reproducibility. Default False.

    Returns
    -------
    inputs : list of Variable
        The network's data inputs, in call order.
    outputs : Variable or list of Variable
        The rebuilt, weight-filled output(s).
    """
    directory = Path(directory)
    if source_format == "auto":
        source_format = _detect_format(json.loads((directory / CONFIG_FILENAME).read_text()))
    if source_format == "huggingface":
        raise NotImplementedError(
            "Loading HuggingFace models is not yet supported; pass a pytensor_ml directory."
        )

    data_inputs, outputs = load_network(directory / CONFIG_FILENAME, restore_rng=restore_rng)
    load_state(_weight_variables(outputs), directory / WEIGHTS_FILENAME)
    return data_inputs, outputs
