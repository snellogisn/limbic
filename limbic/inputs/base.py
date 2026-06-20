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

from .._core import Capability


class Input(Capability):
    """Base class for every sensory input (motors, cameras, ...).

    Shares ``name`` / ``summary`` / ``parameters`` / ``describe()`` and the
    required-argument check with :class:`~limbic._core.Capability`; an input just
    adds :meth:`read`. Inputs are read-only — they observe, never command the arm.
    """

    _kind = "input"

    @abc.abstractmethod
    def read(self, **kwargs: Any) -> Any:
        """Return the current reading from this sense (must be JSON-serialisable)."""

    def __call__(self, **kwargs: Any) -> Any:
        """Validate required args (clear ``TypeError`` if missing), then read."""
        self._check_required(kwargs)
        return self.read(**kwargs)
