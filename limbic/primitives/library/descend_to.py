"""``descend_to`` — lower the tip to a grasp/contact height at a table point.

The second half of the hover-then-descend pattern (see ``reach_above``). It uses
the control layer's slow precision profile because near-surface motion is where
contact happens — slower, finer steps keep the descent controlled so the tip
seats on or into the object cleanly rather than crashing or bouncing. Pair it
with a preceding ``reach_above`` over the same ``(x, y)`` for a safe approach.
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class DescendTo(Primitive):
    """Lower the tip to height ``z`` at ``(x, y)`` using the slow precision profile."""

    name = "descend_to"
    summary = "Lower the tip to a grasp/contact height z at table point (x, y)."
    parameters: dict[str, dict[str, Any]] = {
        "x_mm": {
            "type": "number",
            "description": "Target x (forward, mm).",
        },
        "y_mm": {
            "type": "number",
            "description": "Target y (left, mm).",
        },
        "z_mm": {
            "type": "number",
            "description": "Target height to descend to above the table (mm).",
        },
    }

    def run(
        self,
        arm: RobotArm,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        return arm.descend_to(x_mm, y_mm, z_mm)
