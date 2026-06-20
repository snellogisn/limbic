"""The primitive registry: auto-discovers every primitive in ``library/``.

This is the catalog the LLM brain browses. It imports every module under
``primitives/library/``, collects each :class:`Primitive` subclass, and indexes
them by ``name``. Because discovery is automatic, the LLM (or a human) can add a
new primitive simply by dropping a new file in ``library/`` — no central list to
edit. Likewise an old primitive can be revised in place, or removed by deleting
its file.

Typical use:
    from limbic.primitives import registry
    registry.catalog()           # list of {name, summary, parameters} for the LLM
    prim = registry.get("pick")  # an instantiated primitive
    prim(arm, x_mm=160, y_mm=0, object_height_mm=25)
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

from .base import Primitive

_REGISTRY: dict[str, Primitive] = {}
_LOADED = False


def _discover() -> None:
    """Import every module in ``library/`` and register the primitives it defines."""
    global _LOADED
    if _LOADED:
        return

    from . import library  # local import to avoid a cycle at module load

    for module_info in pkgutil.iter_modules(library.__path__):
        module = importlib.import_module(f"{library.__name__}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            # Register concrete Primitive subclasses defined in this module only.
            if (
                issubclass(obj, Primitive)
                and obj is not Primitive
                and obj.__module__ == module.__name__
                and getattr(obj, "name", "")
            ):
                _REGISTRY[obj.name] = obj()
    _LOADED = True


def reload() -> None:
    """Forget all discovered primitives and re-scan ``library/`` on next access.

    Useful after the LLM writes or edits a primitive file at runtime and wants the
    new/changed primitive picked up without restarting the process.
    """
    global _LOADED
    _REGISTRY.clear()
    _LOADED = False


def all_primitives() -> dict[str, Primitive]:
    """Return ``{name: primitive}`` for every discovered primitive."""
    _discover()
    return dict(_REGISTRY)


def get(name: str) -> Primitive:
    """Return the instantiated primitive named ``name`` (raises ``KeyError`` if absent)."""
    _discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"no primitive named '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[name]


def catalog() -> list[dict[str, Any]]:
    """Return the browsable catalog: a ``describe()`` dict per primitive.

    This is what gets shown to the LLM so it can choose which primitives to use.
    """
    _discover()
    return [p.describe() for p in sorted(_REGISTRY.values(), key=lambda p: p.name)]
