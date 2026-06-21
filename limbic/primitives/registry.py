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

from typing import Any

from .._core import Registry
from . import library
from .base import Primitive

# One shared auto-discovery registry, scoped to the Primitive subclasses in library/.
_registry = Registry(Primitive, library)


def reload() -> None:
    """Forget all discovered primitives and re-scan ``library/`` on next access.

    Useful after a primitive file is written or edited at runtime and the new or
    changed primitive should be picked up without restarting the process.
    """
    _registry.reload()


def all_primitives() -> dict[str, Primitive]:
    """Return ``{name: primitive}`` for every discovered primitive."""
    return _registry.all()  # type: ignore[return-value]


def get(name: str) -> Primitive:
    """Return the instantiated primitive named ``name`` (raises ``KeyError`` if absent)."""
    return _registry.get(name)  # type: ignore[return-value]


def catalog() -> list[dict[str, Any]]:
    """Return the browsable catalog: a ``describe()`` dict per primitive.

    This is what gets shown to the LLM so it can choose which primitives to use.
    """
    return _registry.catalog()
