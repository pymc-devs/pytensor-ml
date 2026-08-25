import ast
import importlib
import inspect
import itertools
import pkgutil
import types

import pytensor_ml

# Importing a backend dispatch module pulls in the backend itself, and the core test jobs deliberately
# install none of them.
BACKEND_DISPATCH = "pytensor_ml.dispatch."


def public_objects() -> list[tuple[str, object]]:
    """
    Collect every public object the package defines, for tests that sweep the whole API.

    Returns
    -------
    objects : list of tuple of str and object
        Each object paired with its qualified name, listed under the module that defines it rather than
        every module that re-exports it.
    """
    objects: list[tuple[str, object]] = []
    for module_info in pkgutil.walk_packages(pytensor_ml.__path__, f"{pytensor_ml.__name__}."):
        if module_info.name.startswith(BACKEND_DISPATCH):
            continue
        module = importlib.import_module(module_info.name)
        objects.append((module_info.name, module))
        for name, obj in vars(module).items():
            if not name.startswith("_") and getattr(obj, "__module__", None) == module_info.name:
                objects.append((f"{module_info.name}.{name}", obj))
    return objects


def _assigned_name(node: ast.stmt) -> str | None:
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
        return node.name.id
    return None


def attribute_docstrings(module: types.ModuleType) -> list[tuple[str, str]]:
    """
    Collect the docstrings written under a module's public assignments.

    A type alias cannot carry ``__doc__``, so the string literal below its assignment is the only
    documentation it has, and the only place Sphinx looks.

    Parameters
    ----------
    module : module
        The module to read.

    Returns
    -------
    documented : list of tuple of str and str
        Each qualified attribute name paired with the docstring written beneath it.
    """
    body = ast.parse(inspect.getsource(module)).body
    documented = []
    for assignment, following in itertools.pairwise(body):
        name = _assigned_name(assignment)
        is_docstring = (
            isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        )
        if name and not name.startswith("_") and is_docstring:
            documented.append((f"{module.__name__}.{name}", following.value.value))
    return documented


def all_docstrings() -> list[tuple[str, str]]:
    """
    Collect every public docstring in the package, wherever it is written.

    Returns
    -------
    documented : list of tuple of str and str
        Each qualified name paired with its docstring, covering objects that carry ``__doc__`` and
        module attributes documented by a string literal beneath them.
    """
    documented = []
    for qualified_name, obj in public_objects():
        docstring = getattr(obj, "__doc__", None)
        if docstring:
            documented.append((qualified_name, docstring))
        if isinstance(obj, types.ModuleType):
            documented.extend(attribute_docstrings(obj))
    return documented
