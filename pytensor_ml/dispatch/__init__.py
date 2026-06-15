import importlib
import importlib.abc
import importlib.util
import sys

# pytensor auto-loads its own backend dispatches because its linker imports them at compile time -- a
# third-party op can't hook into that, and pytensor exposes no plugin/entry-point system. To get the same
# "activate when the backend is actually used, never on the main import path" behavior, watch for pytensor
# loading a backend's dispatch package and register ours right after. Installing the hook imports nothing
# heavy; jax/mlx are pulled in only when (and if) that backend compiles a graph.
_REGISTRATIONS = {
    "pytensor.link.jax.dispatch": "pytensor_ml.dispatch.jax",
    "pytensor.link.mlx.dispatch": "pytensor_ml.dispatch.mlx",
}


class _RegisterAfterImport(importlib.abc.MetaPathFinder):
    """Import a registration module right after its target backend-dispatch package finishes loading."""

    def find_spec(self, fullname, path=None, target=None):
        registration = _REGISTRATIONS.get(fullname)
        if registration is None:
            return None

        # Resolve the real spec with ourselves out of the way to avoid infinite recursion.
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None

        load = spec.loader.exec_module

        def exec_module(module):
            load(module)
            _REGISTRATIONS.pop(fullname, None)
            if not _REGISTRATIONS and self in sys.meta_path:
                sys.meta_path.remove(self)
            importlib.import_module(registration)

        spec.loader.exec_module = exec_module
        return spec


# A backend dispatch already loaded before us (e.g. a prior compile in the same process) won't trip the
# finder, so register against it immediately.
for _dispatch_module, _registration in list(_REGISTRATIONS.items()):
    if _dispatch_module in sys.modules:
        _REGISTRATIONS.pop(_dispatch_module)
        importlib.import_module(_registration)

if _REGISTRATIONS:
    sys.meta_path.insert(0, _RegisterAfterImport())
