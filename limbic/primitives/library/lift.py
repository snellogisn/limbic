"""``lift`` — raise (or lower) the tip vertically from where it currently is.

A purely relative vertical move: it changes height while holding ``(x, y)``,
which is exactly what's wanted immediately after a grasp (lift the object clear
of the table before transiting) or to set down gently (a negative ``dz``). Being
relative to the *current* pose makes it composable — the planner doesn't need to
know the absolute height, just "go up 80mm from here".
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class Lift(Primitive):
    """Raise (positive) or lower (negative) the current tip position by ``dz_mm``."""

    name = "lift"
    summary = "Raise or lower the current tip position vertically by dz_mm."
    parameters: dict[str, dict[str, Any]] = {
        "dz_mm": {
            "type": "number",
            "description": "Vertical change in mm; positive raises, negative lowers.",
        },
    }

    def run(
        self,
        arm: RobotArm,
        dz_mm: float,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        return arm.lift_by(dz_mm)
