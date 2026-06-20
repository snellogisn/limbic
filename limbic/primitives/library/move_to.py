"""``move_to`` — move the tool tip to an explicit table-frame point.

The lowest-level Cartesian primitive: it just exposes ``RobotArm.move_to_xyz``
as a named, browsable skill. The planner reaches for this when it needs a raw
positioning move that the higher-level composites (pick/place/push) don't cover
— for example, parking the tip at a staging point or tracing a path of
waypoints. The approach pitch is exposed so the planner can choose a top-down
grasp (``-90``) or an angled approach when geometry demands it.
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class MoveTo(Primitive):
    """Move the tool tip to a table-frame ``(x, y, z)`` at a chosen approach pitch."""

    name = "move_to"
    summary = "Move the tool tip to a table-frame (x, y, z) point in millimetres."
    parameters: dict[str, dict[str, Any]] = {
        "x_mm": {
            "type": "number",
            "description": "Target x (forward, mm; origin under the shoulder-pan axis).",
        },
        "y_mm": {
            "type": "number",
            "description": "Target y (left, mm).",
        },
        "z_mm": {
            "type": "number",
            "description": "Target z (up from the table surface, mm).",
        },
        "approach_pitch_deg": {
            "type": "number",
            "description": "Tool pitch; -90 = straight down (top-down). Default -90.",
            "default": -90.0,
        },
    }

    def run(
        self,
        arm: RobotArm,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        approach_pitch_deg: float = -90.0,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        return arm.move_to_xyz(x_mm, y_mm, z_mm, approach_pitch_deg=approach_pitch_deg)
