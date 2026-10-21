"""The services package.

This package is imported under two names, ``services`` (with ``src`` on the path) and ``src.services``.
Python treats those as two separate modules, each with its own globals. For most modules that is harmless.
For a module that holds process-wide state it is not: ``ollama_client`` counts foreground generations (which
background work yields to), queues requests, and remembers GPU layout refusals, so with two copies the work
done through one is invisible through the other.

``_SHARED_STATE_MODULES`` lists the modules that must exist once. When the package is loaded as ``services``,
a small import hook answers ``services.<name>`` with the one real ``src.services.<name>`` instead of loading
a second copy. Callers change nothing, and every other module loads as it always did. (These modules already
import their own neighbours as ``src.services...``, so they have always needed ``src`` on the path.)

Add a module here only after checking what state it holds and what depends on it: the effect is that two
copies become one. ``tests/test_import_unification.py`` checks the list.
"""
import importlib
import importlib.abc
import importlib.util
import sys

_SHARED_STATE_MODULES = ("ollama_client",)
_CANONICAL = "src.services."


class _OneCopy(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Answers ``services.<shared module>`` with the canonical ``src.services.<shared module>``."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("services."):
            return None
        if fullname.split(".", 1)[1] not in _SHARED_STATE_MODULES:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return importlib.import_module(_CANONICAL + spec.name.split(".", 1)[1])

    def exec_module(self, module):  # the canonical module is already executed
        return None


if __name__ == "services" and not any(isinstance(f, _OneCopy) for f in sys.meta_path):
    sys.meta_path.insert(0, _OneCopy())
