"""The Input contract: one named, queryable source of sensory information.

The arm will grow many senses — joint/motor feedback and cameras now, more
later (force, depth, tags). An *input* wraps one such source behind a uniform,
self-describing interface so the LLM brain can browse what perceptions are
available and call the ones it needs to solve an instruction (e.g. "where is the
red block?" -> query the camera; "is the gripper holding something?" -> query
joint state).

Each input lives as its own file in ``inputs/library/`` and subclasses
:class:`Input`, declaring:
    * ``name``       — unique identifier the planner refers to
    * ``summary``    — one line: what this sense reports
    * ``parameters`` — JSON-schema-style spec of any query arguments
    * ``read(**kwargs)`` — return the reading (a JSON-serialisable value)

Inputs are read-only: they observe, they never command the arm.
"""

from __future__ import annotations

import abc
from typing import Any


class Input(abc.ABC):
    """Base class for every sensory input (motors, cameras, ...)."""

    #: Unique identifier the planner uses to refer to this input.
    name: str = ""
    #: One-line description for the browsable catalog.
    summary: str = ""
    #: JSON-schema-style argument spec: {arg: {"type", "description", "default"?}}.
    parameters: dict[str, dict[str, Any]] = {}

    @abc.abstractmethod
    def read(self, **kwargs: Any) -> Any:
        """Return the current reading from this sense (must be JSON-serialisable)."""

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Serialisable description (name, summary, parameter schema) for the LLM."""
        return {
            "name": cls.name,
            "summary": cls.summary,
            "parameters": cls.parameters,
        }

    def __call__(self, **kwargs: Any) -> Any:
        """Validate required args, then read. Missing required args raise clearly."""
        missing = [
            arg
            for arg, spec in self.parameters.items()
            if "default" not in spec and arg not in kwargs
        ]
        if missing:
            raise TypeError(
                f"input '{self.name}' missing required argument(s): {', '.join(missing)}"
            )
        return self.read(**kwargs)
