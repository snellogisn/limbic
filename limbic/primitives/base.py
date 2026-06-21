"""The Primitive contract: one self-contained, named, parameterised arm skill.

A *motion primitive* is a small, reusable building block — "pick", "place",
"home", "push" — that the LLM brain selects and chains into a plan. Each
primitive is a subclass of :class:`Primitive` living as its own file in
``primitives/library/``. Keeping them one-per-file and uniformly shaped is what
lets the LLM (a) browse the catalog, (b) pick the right ones, and (c) author new
ones or revise stale ones by writing a single file in the same mould.

A primitive must declare:
    * ``name``        — unique, kebab/snake identifier the planner refers to
    * ``summary``     — one line: what it does, shown in the browsable catalog
    * ``parameters``  — JSON-schema-style dict describing its arguments (this is
                        what becomes the LLM tool schema, so be descriptive)
    * ``run(arm, **kwargs)`` — the actual motion, calling only the ``RobotArm``
                        tool surface so it inherits all safety + smoothing

Primitives never bypass safety: they only call ``RobotArm`` methods, which always
clamp to the workspace and per-joint soft limits.
"""

from __future__ import annotations

import abc
from typing import Any

from .._core import Capability
from ..control import RobotArm


class Primitive(Capability):
    """Base class for every motion primitive.

    Shares ``name`` / ``summary`` / ``parameters`` / ``describe()`` and the
    required-argument check with :class:`~limbic._core.Capability`; a primitive
    just adds :meth:`run`. See ``primitives/library/`` for worked examples.
    """

    _kind = "primitive"

    @abc.abstractmethod
    def run(self, arm: RobotArm, **kwargs: Any) -> Any:
        """Execute the primitive against ``arm`` with validated keyword arguments."""

    def __call__(self, arm: RobotArm, **kwargs: Any) -> Any:
        """Validate required args (clear ``TypeError`` if missing), then run."""
        self._check_required(kwargs)
        return self.run(arm, **kwargs)
