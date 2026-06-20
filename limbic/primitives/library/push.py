"""``push`` — slide an object across the table without grasping it.

A non-grasp manipulation primitive, and a nice demonstration that primitives
aren't all about the gripper. Instead of lifting the object, it uses the closed
claw as a flat pusher and shoves the object along the surface from one point to
another:

  1. Close the claw so it presents a solid pushing face (an open claw would let
     the object slip between the fingers).
  2. Hover just behind/above the start point, then descend to ``push_z_mm`` — a
     *low* height so the contact point is near the object's base. Pushing low
     keeps the object sliding flat rather than tipping it over.
  3. Drive horizontally at that same low height from ``(x, y)`` to ``(x2, y2)``,
     carrying the object with it.
  4. Lift clear so the claw doesn't drag back over the object on the next move.

Useful for nudging things into place, clearing a lane, or repositioning objects
too large or awkward to grasp.
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class Push(Primitive):
    """Slide an object along the table from ``(x, y)`` to ``(x2, y2)`` at a low height."""

    name = "push"
    summary = "Push an object along the table from (x, y) to (x2, y2) at a low height."
    parameters: dict[str, dict[str, Any]] = {
        "x_mm": {
            "type": "number",
            "description": "Start x of the push (forward, mm).",
        },
        "y_mm": {
            "type": "number",
            "description": "Start y of the push (left, mm).",
        },
        "x2_mm": {
            "type": "number",
            "description": "End x of the push (forward, mm).",
        },
        "y2_mm": {
            "type": "number",
            "description": "End y of the push (left, mm).",
        },
        "push_z_mm": {
            "type": "number",
            "description": "Low contact height to push at, near the object base (mm). Default 15.",
            "default": 15.0,
        },
        "hover_z_mm": {
            "type": "number",
            "description": "Clearance height to approach/retreat at (mm). Default 60.",
            "default": 60.0,
        },
    }

    def run(
        self,
        arm: RobotArm,
        x_mm: float,
        y_mm: float,
        x2_mm: float,
        y2_mm: float,
        push_z_mm: float = 15.0,
        hover_z_mm: float = 60.0,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        # 1. Close the claw so it presents a solid pushing face.
        arm.close_gripper()

        # 2. Approach above the start point, then descend to the low push height.
        arm.reach_above(x_mm, y_mm, height_mm=hover_z_mm)
        arm.descend_to(x_mm, y_mm, push_z_mm)

        # 3. Drive horizontally at the low height, sliding the object along.
        arm.move_to_xyz(x2_mm, y2_mm, push_z_mm, slow=True)

        # 4. Lift clear so the claw doesn't drag back over the object next move.
        return arm.lift_by(hover_z_mm - push_z_mm)
