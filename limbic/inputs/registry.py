"""The input registry: auto-discovers every sense in ``library/``.

Mirrors the primitive registry. It imports every module under
``inputs/library/``, collects each :class:`Input` subclass, and indexes them by
``name``. The LLM browses ``catalog()`` to learn what perceptions exist, then
calls ``read(name, ...)`` to query one.

Runtime context injection
-------------------------
Some senses need live objects the LLM should never have to (or be able to) pass
— most importantly the connected ``RobotArm`` for joint/motor readings. The
registry holds a ``context`` dict (e.g. ``{"arm": arm}``) and injects any of its
keys that an input's ``read()`` actually accepts. The camera, which needs no
context, simply doesn't declare those parameters. This keeps the LLM-facing
parameter schema clean while still wiring senses to live hardware.
"""

from __future__ import annotations

import inspect
from typing import Any

from .. import runlog
from .._core import Registry
from . import library
from .base import Input

# One shared auto-discovery registry, scoped to the Input subclasses in library/.
_registry = Registry(Input, library)

# Runtime objects (e.g. the connected arm) injected into input reads — see read().
_CONTEXT: dict[str, Any] = {}


def reload() -> None:
    """Re-scan ``library/`` on next access (after adding/editing a sense)."""
    _registry.reload()


def set_context(**context: Any) -> None:
    """Provide runtime objects (e.g. ``arm=...``) injectable into input reads."""
    _CONTEXT.update(context)


def all_inputs() -> dict[str, Input]:
    """Return ``{name: input}`` for every discovered sense."""
    return _registry.all()  # type: ignore[return-value]


def get(name: str) -> Input:
    """Return the instantiated input named ``name`` (raises ``KeyError`` if absent)."""
    return _registry.get(name)  # type: ignore[return-value]


def catalog() -> list[dict[str, Any]]:
    """Return the browsable catalog: a ``describe()`` dict per input, for the LLM."""
    return _registry.catalog()


def read(name: str, **kwargs: Any) -> Any:
    """Query input ``name``, injecting any runtime context its ``read`` accepts.

    The LLM passes only the schema parameters; the registry adds context keys
    (like ``arm``) automatically if the input's ``read`` signature declares them.
    """
    sense = get(name)  # triggers discovery on first use
    # The LLM-facing query args, before runtime context (e.g. the arm) is mixed in.
    query_args = dict(kwargs)
    sig = inspect.signature(sense.read)
    for key, value in _CONTEXT.items():
        if key in sig.parameters and key not in kwargs:
            kwargs[key] = value
    reading = sense(**kwargs)
    # Record what the arm saw into the active run's data stream (no-op if no run).
    runlog.current().data(source=name, reading=reading, args=query_args)
    return reading
