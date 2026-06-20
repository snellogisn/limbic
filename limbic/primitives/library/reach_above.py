"""``reach_above`` — hover above a table point, ready to descend.

Hovering first, then descending straight down, is the safe two-phase approach to
any point on the table: the transit move happens at a clearance height where the
tip can't clip the object or its neighbours, and only the final vertical descent
happens near the surface. This primitive is the "get into position" half of that
pattern, used on its own when the planner wants to stage above a target before
deciding what to do next.
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class ReachAbove(Primitive):
    """Position the tip directly above ``(x, y)`` at a clearance height."""

    name = "reach_above"
    summary = "Hover the tip above a table point (x, y) at a clearance height."
    parameters: dict[str, dict[str, Any]] = {
        "x_mm": {
            "type": "number",
            "description": "Target x (forward, mm).",
        },
        "y_mm": {
            "type": "number",
            "description": "Target y (left, mm).",
        },
        "height_mm": {
            "type": "number",
            "description": "Clearance height above the table to hover at (mm). Default 70.",
            "default": 70.0,
        },
    }

    def run(
        self,
        arm: RobotArm,
        x_mm: float,
        y_mm: float,
        height_mm: float = 70.0,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        return arm.reach_above(x_mm, y_mm, height_mm=height_mm)
