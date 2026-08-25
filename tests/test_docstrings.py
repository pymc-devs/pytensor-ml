import importlib
import pkgutil

import pytest

import pytensor_ml

# A LaTeX macro in a docstring that forgot its `r` prefix silently becomes a control character:
# `\text` is a tab, `\rceil` a carriage return, `\b` a backspace. The docstring still renders, just
# wrong, so nothing but a scan like this catches it.
CONTROL_CHARACTERS = {"\t": r"\t", "\r": r"\r", "\x08": r"\b", "\x0c": r"\f", "\x0b": r"\v"}


# Importing a backend dispatch module pulls in the backend itself, and the core test jobs deliberately
# install none of them.
BACKEND_DISPATCH = "pytensor_ml.dispatch."


def _public_objects() -> list[tuple[str, object]]:
    objects: list[tuple[str, object]] = []
    for module_info in pkgutil.walk_packages(pytensor_ml.__path__, f"{pytensor_ml.__name__}."):
        if module_info.name.startswith(BACKEND_DISPATCH):
            continue
        module = importlib.import_module(module_info.name)
        objects.append((module_info.name, module))
        for name, obj in vars(module).items():
            # Skip re-exports so each object is checked once, under the module that defines it.
            if not name.startswith("_") and getattr(obj, "__module__", None) == module_info.name:
                objects.append((f"{module_info.name}.{name}", obj))
    return objects


PUBLIC_OBJECTS = _public_objects()


@pytest.mark.parametrize(
    "qualified_name, obj", PUBLIC_OBJECTS, ids=[name for name, _ in PUBLIC_OBJECTS]
)
def test_docstring_has_no_control_characters(qualified_name, obj):
    docstring = getattr(obj, "__doc__", None)
    if not docstring:
        return

    found = {escape for character, escape in CONTROL_CHARACTERS.items() if character in docstring}
    assert not found, (
        f"{qualified_name} has {sorted(found)} in its docstring, which means a LaTeX macro was "
        f'interpreted as an escape sequence. Mark the docstring raw: r"""..."""'
    )
