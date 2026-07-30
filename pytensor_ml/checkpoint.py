from collections.abc import Mapping, Sequence
from os import fspath
from pathlib import Path
from typing import Any

import numpy as np

from pytensor.compile.sharedvalue import SharedVariable
from safetensors.numpy import load_file, save_file


def _index_by_name(shared_variables: Sequence[SharedVariable]) -> dict[str, SharedVariable]:
    """Index shared variables by name, rejecting unnamed or duplicate names."""
    indexed: dict[str, SharedVariable] = {}
    for variable in shared_variables:
        name = variable.name
        if name is None:
            raise ValueError(
                f"Cannot checkpoint the unnamed shared variable {variable!r}: the name is the only "
                f"handle at the serialization boundary, so every variable must be named."
            )
        if name in indexed:
            raise ValueError(f"Duplicate shared-variable name {name!r}; names must be unique.")
        indexed[name] = variable
    return indexed


def save_state(shared_variables: Sequence[SharedVariable], path: str | Path) -> None:
    """
    Save the current values of shared variables to a name-keyed ``.safetensors`` archive.

    The variables' names become the archive keys, so the caller is responsible for naming them
    distinctly. Pass the parameters together with the optimizer state to capture a complete training
    checkpoint; both are ordinary shared variables and carry self-describing names (e.g. ``"fc1/weight"``,
    ``"fc1/weight/adam/first_moment"``).

    Parameters
    ----------
    shared_variables : sequence of SharedVariable
        Variables whose values to save. Every variable must have a unique, non-``None`` name.
    path : str or pathlib.Path
        Destination archive, written verbatim.
    """
    indexed = _index_by_name(shared_variables)
    tensors = {name: _as_saveable_array(variable.get_value()) for name, variable in indexed.items()}
    save_file(tensors, fspath(path))


def _as_saveable_array(value: Any) -> np.ndarray:
    """
    Return a value as a C-contiguous numpy array safetensors will accept, preserving its rank.

    A JIT backend stores its own array type in the shared variables a compiled function updates, so a
    trained model's values are ``jax.Array`` or ``mlx.core.array``; both convert through the array
    protocol.
    """
    array = np.asarray(value)
    # Not np.ascontiguousarray: it forces ndim >= 1, reshaping a rank-0 array such as the step count.
    return array if array.flags["C_CONTIGUOUS"] else np.array(array, order="C")


def load_state(
    shared_variables: Sequence[SharedVariable],
    path: str | Path,
    name_map: Mapping[str, str] | None = None,
) -> None:
    """
    Load values from a name-keyed ``.safetensors`` archive into shared variables, in place.

    Match is by name: each variable's name (optionally remapped through ``name_map``) selects the archive
    entry to load into it. The archive's keys and the target names must correspond exactly — a missing or
    extra key, or any shape or dtype mismatch, raises rather than loading partial state. Every target is
    validated before any value is written, so a failed load leaves all variables untouched.

    Parameters
    ----------
    shared_variables : sequence of SharedVariable
        Variables to restore. Every variable must have a unique, non-``None`` name.
    path : str or pathlib.Path
        A ``.safetensors`` archive, e.g. one written by :func:`save_state` or a HuggingFace checkpoint.
    name_map : mapping of str to str, optional
        Maps a variable's name to the archive key to read it from, for loading a checkpoint saved under
        different names (such as HuggingFace's). The mapping must be injective. Names absent from the map
        are matched directly.
    """
    indexed = _index_by_name(shared_variables)
    name_map = name_map or {}

    target_by_key: dict[str, SharedVariable] = {}
    for name, variable in indexed.items():
        key = name_map.get(name, name)
        collision = target_by_key.get(key)
        if collision is not None:
            raise ValueError(
                f"name_map is not injective: both {collision.name!r} and {name!r} map to {key!r}."
            )
        target_by_key[key] = variable

    values = load_file(fspath(path))
    missing = set(target_by_key) - set(values)
    unexpected = set(values) - set(target_by_key)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing from archive: {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected in archive: {sorted(unexpected)}")
        raise ValueError(f"Archive keys do not match the targets ({'; '.join(details)}).")

    mismatches = []
    for key, variable in target_by_key.items():
        value = values[key]
        # Through numpy: a backend array's own dtype object never compares equal to a numpy dtype.
        current = np.asarray(variable.get_value(borrow=True))
        if value.shape != current.shape or value.dtype != current.dtype:
            mismatches.append(
                f"  {variable.name!r}: target is {current.dtype} of shape {current.shape}, "
                f"archive has {value.dtype} of shape {value.shape}"
            )
    if mismatches:
        raise ValueError("Archive values do not match their targets:\n" + "\n".join(mismatches))

    for key, variable in target_by_key.items():
        variable.set_value(values[key])
