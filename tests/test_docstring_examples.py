import textwrap
import types

import pytest

from tests.public_api import attribute_docstrings, public_objects

CODE_BLOCK_DIRECTIVE = ".. code-block:: python"


def _example_blocks(docstring: str) -> list[str]:
    lines = docstring.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "Examples"), None)
    if start is None:
        return []

    blocks = []
    for index, line in enumerate(lines[start:], start=start):
        if line.strip() != CODE_BLOCK_DIRECTIVE:
            continue
        directive_indent = len(line) - len(line.lstrip())
        body = []
        for candidate in lines[index + 1 :]:
            indent = len(candidate) - len(candidate.lstrip())
            if candidate.strip() and indent <= directive_indent:
                break
            body.append(candidate)
        code = textwrap.dedent("\n".join(body)).strip()
        # An empty block means the directive is there but its body is not indented under it, which used to
        # pass as a trivially successful exec.
        assert code, f"code-block at line {index} has no indented body"
        blocks.append(code)
    return blocks


def _collect_examples() -> list[tuple[str, str]]:
    examples = []
    for qualified_name, obj in public_objects():
        docstrings = [(qualified_name, obj.__doc__ or "")]
        if isinstance(obj, types.ModuleType):
            docstrings.extend(attribute_docstrings(obj))
        for name, docstring in docstrings:
            for position, block in enumerate(_example_blocks(docstring)):
                examples.append((f"{name}[{position}]", block))
    return examples


EXAMPLES = _collect_examples()


@pytest.mark.parametrize("qualified_name, source", EXAMPLES, ids=[name for name, _ in EXAMPLES])
def test_docstring_example_runs(qualified_name, source, tmp_path, monkeypatch):
    # A fresh namespace per block: an example that leans on a name another example imported is not the
    # self-contained snippet a reader is invited to paste into a script. Each runs in its own directory
    # so an example that writes a checkpoint can use the plain relative path a reader would.
    monkeypatch.chdir(tmp_path)
    exec(compile(source, f"<{qualified_name}>", "exec"), {"__name__": "__main__"})
