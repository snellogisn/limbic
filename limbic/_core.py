"""Shared internals: the capability contract + the auto-discovery registry.

Motion primitives and sensory inputs are the same shape — a named, parameterised,
self-describing unit — and both are auto-discovered from a ``library/`` folder.
This module factors out that shared machinery so ``primitives/`` and ``inputs/``
don't each carry a near-identical copy:

    Capability  the common base: ``name`` / ``summary`` / ``parameters`` +
                ``describe()`` + required-argument validation. ``Primitive`` and
                ``Input`` subclass it and add ``run`` / ``read`` respectively.
    Registry    discovers every ``Capability`` subclass under a package and
                indexes them by ``name`` — the catalog the LLM browses.

Kept to the standard library only, so importing it never pulls in the control
stack (or anything heavy).
"""

from __future__ import annotations

import abc
import importlib
import inspect
import pkgutil
from typing import Any


def required_args(parameters: dict[str, dict[str, Any]] | None) -> list[str]:
    """Names of the arguments a caller MUST supply.

    Our convention: a parameter is required exactly when its spec has no
    ``"default"`` key. Shared by the validation here, the tool-schema builder
    (``brain/tools.py``), and the catalog renderer (``brain/system_prompt.py``).
    """
    return [arg for arg, spec in (parameters or {}).items() if "default" not in (spec or {})]


class Capability(abc.ABC):
    """Base for anything the brain browses and invokes: a primitive or an input."""

    #: Unique identifier the planner uses to refer to this capability.
    name: str = ""
    #: One-line description for the browsable catalog.
    summary: str = ""
    #: Parameter spec: ``{arg: {"type", "description", "default"?}}``.
    parameters: dict[str, dict[str, Any]] = {}
    #: Word used in error messages ("primitive" / "input"); overridden by subclasses.
    _kind: str = "capability"

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Serialisable description (name, summary, parameter schema) for the LLM."""
        return {"name": cls.name, "summary": cls.summary, "parameters": cls.parameters}

    def _check_required(self, kwargs: dict[str, Any]) -> None:
        """Raise a clear ``TypeError`` if any required argument is missing."""
        missing = [arg for arg in required_args(self.parameters) if arg not in kwargs]
        if missing:
            raise TypeError(
                f"{self._kind} '{self.name}' missing required argument(s): "
                f"{', '.join(missing)}"
            )


class Registry:
    """Auto-discovers ``Capability`` subclasses under a package and indexes by name.

    One instance backs ``primitives.registry`` and another backs
    ``inputs.registry``. Discovery is lazy (on first access) and cached; call
    :meth:`reload` to re-scan after a library file is added or changed.
    """

    def __init__(self, base_class: type, package: Any):
        self._base = base_class
        self._package = package
        self._items: dict[str, Capability] = {}
        self._loaded = False

    def _discover(self) -> None:
        if self._loaded:
            return
        for info in pkgutil.iter_modules(self._package.__path__):
            module = importlib.import_module(f"{self._package.__name__}.{info.name}")
            for _, obj in inspect.getmembers(module, inspect.isclass):
                # Register concrete subclasses DEFINED in this module (not the base
                # class itself, nor a base imported into the module).
                if (
                    issubclass(obj, self._base)
                    and obj is not self._base
                    and obj.__module__ == module.__name__
                    and getattr(obj, "name", "")
                ):
                    self._items[obj.name] = obj()
        self._loaded = True

    def reload(self) -> None:
        """Forget everything and re-scan the library on next access."""
        self._items.clear()
        self._loaded = False

    def all(self) -> dict[str, Capability]:
        """Return ``{name: instance}`` for every discovered capability."""
        self._discover()
        return dict(self._items)

    def get(self, name: str) -> Capability:
        """Return the instance named ``name`` (raises ``KeyError`` if absent)."""
        self._discover()
        if name not in self._items:
            kind = self._base.__name__.lower()
            raise KeyError(
                f"no {kind} named '{name}'. Available: {', '.join(sorted(self._items))}"
            )
        return self._items[name]

    def catalog(self) -> list[dict[str, Any]]:
        """Return a ``describe()`` dict per capability, name-sorted, for the LLM."""
        self._discover()
        return [item.describe() for item in sorted(self._items.values(), key=lambda i: i.name)]
