import pytest

from tests.public_api import all_docstrings

# A LaTeX macro in a docstring that forgot its `r` prefix silently becomes a control character:
# `\text` is a tab, `\rceil` a carriage return, `\frac` a formfeed. The docstring still renders, just
# wrong, so nothing but a scan like this catches it.
CONTROL_CHARACTERS = {"\t": r"\t", "\r": r"\r", "\x08": r"\b", "\x0c": r"\f", "\x0b": r"\v"}

DOCSTRINGS = all_docstrings()


@pytest.mark.parametrize(
    "qualified_name, docstring", DOCSTRINGS, ids=[name for name, _ in DOCSTRINGS]
)
def test_docstring_has_no_control_characters(qualified_name, docstring):
    found = {escape for character, escape in CONTROL_CHARACTERS.items() if character in docstring}
    assert not found, (
        f"{qualified_name} has {sorted(found)} in its docstring, which means a LaTeX macro was "
        f'interpreted as an escape sequence. Mark the docstring raw: r"""..."""'
    )
