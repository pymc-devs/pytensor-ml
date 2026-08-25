import pytest

from tests.public_api import public_objects

# A LaTeX macro in a docstring that forgot its `r` prefix silently becomes a control character:
# `\text` is a tab, `\rceil` a carriage return, `\b` a backspace. The docstring still renders, just
# wrong, so nothing but a scan like this catches it.
CONTROL_CHARACTERS = {"\t": r"\t", "\r": r"\r", "\x08": r"\b", "\x0c": r"\f", "\x0b": r"\v"}

PUBLIC_OBJECTS = public_objects()


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
