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

from ..control import RobotArm


class Primitive(abc.ABC):
    """Base class for every motion primitive.

    Subclasses set the three class attributes and implement :meth:`run`. See
    ``primitives/library/`` for worked examples and ``Primitive.describe`` for
    the shape the planner/LLM consumes.
    """

    #: Unique identifier the planner uses to refer to this primitive.
    name: str = ""
    #: One-line description for the browsable catalog.
    summary: str = ""
    #: JSON-schema-style parameter spec: {arg: {"type", "description", "default"?}}.
    parameters: dict[str, dict[str, Any]] = {}

    @abc.abstractmethod
    def run(self, arm: RobotArm, **kwargs: Any) -> Any:
        """Execute the primitive against ``arm`` with validated keyword arguments."""

    # ------------------------------------------------------------------ #
    # Introspection used by the registry, the catalog, and the LLM tools
    # ------------------------------------------------------------------ #
    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Return a serialisable description (name, summary, parameter schema).

        This is the unit of information the LLM sees when choosing primitives and
        the basis for the generated tool schema in ``brain/tools.py``.
        """
        return {
            "name": cls.name,
            "summary": cls.summary,
            "parameters": cls.parameters,
        }

    def __call__(self, arm: RobotArm, **kwargs: Any) -> Any:
        """Validate required args against the schema, then run.

        Required = any parameter whose schema has no ``"default"`` key. Missing
        required args raise a clear ``TypeError`` before any motion happens.
        """
        missing = [
            arg
            for arg, spec in self.parameters.items()
            if "default" not in spec and arg not in kwargs
        ]
        if missing:
            raise TypeError(
                f"primitive '{self.name}' missing required argument(s): "
                f"{', '.join(missing)}"
            )
        return self.run(arm, **kwargs)
