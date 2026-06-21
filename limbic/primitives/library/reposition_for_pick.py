"""``reposition_for_pick`` — push a poorly-placed object into the grasp sweet spot.

A clean TOP-DOWN grasp is accurate only in an inner reach band near the centerline
(§A.6/§A.7): too far out and the wrist tilts (degrading the grasp); too far off the
centerline and a not-quite-level base rides the tip high. When a detected object
sits OUTSIDE that band, the smart move is to PUSH it in first, then grasp it where
the arm is strongest — rather than attempting a degraded tilted grasp where it lies.

This primitive does exactly that, deterministically:

  1. Decide if the object at ``(x, y)`` is already well placed
     (``calibration.in_pick_zone``). If so, it's a NO-OP — no push, nothing moved.
  2. Otherwise compute the NEAREST point inside the good zone
     (``calibration.pick_staging_target``) — the minimal nudge, not a haul across
     the table — and push the object there: close the claw to a flat face, approach
     from just BEHIND the object (on the side away from the target), descend to a
     low contact height near the base, slide it to the staging point, and lift clear.

It does NOT grasp. After it runs, the object has moved, so the brain must
RE-DETECT (sense_object_detections) to read the new position, then pick — the
standard act-then-see pattern. The push geometry mirrors the ``push`` primitive
(low contact so the object slides flat instead of tipping; lift-clear at the end).
"""

from __future__ import annotations

import math
from typing import Any

from ..base import Primitive
from ...control import RobotArm
from ...control import calibration


class RepositionForPick(Primitive):
    """Push an object at ``(x, y)`` into the workspace sweet spot for an optimal grasp."""

    name = "reposition_for_pick"
    summary = (
        "If an object at (x, y) is in a poor spot for a clean top-down grasp (too "
        "far out, or too far off the centerline), push it into the workspace sweet "
        "spot so it can be picked optimally; NO-OP if it's already well placed. Does "
        "NOT grasp — re-detect the object afterward (its position changed) and then "
        "pick."
    )
    parameters: dict[str, dict[str, Any]] = {
        "x_mm": {
            "type": "number",
            "description": "Object centre x (forward, mm) — its CURRENT detected position.",
        },
        "y_mm": {
            "type": "number",
            "description": "Object centre y (left, mm) — its CURRENT detected position.",
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
        push_z_mm: float = 15.0,
        hover_z_mm: float = 60.0,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        # 1. Already well placed? Do nothing — don't disturb a good grasp.
        if calibration.in_pick_zone(x_mm, y_mm):
            return arm.current_xyz()

        # 2. Minimal push: the nearest point inside the good pick zone.
        target_x, target_y = calibration.pick_staging_target(x_mm, y_mm)

        # Push direction (object -> staging target) and a start point just BEHIND the
        # object, so the closed claw contacts it and drives it toward the target.
        dx, dy = target_x - x_mm, target_y - y_mm
        dist = math.hypot(dx, dy)
        if dist < 1.0:  # nothing meaningful to move
            return arm.current_xyz()
        ux, uy = dx / dist, dy / dist
        behind = calibration.PUSH_BEHIND_MM
        start_x, start_y = x_mm - behind * ux, y_mm - behind * uy

        # 3. Push: closed claw, approach behind, descend low, slide to the staging
        #    point, lift clear (low contact keeps the object sliding flat, not tipping).
        arm.close_gripper()
        arm.reach_above(start_x, start_y, height_mm=hover_z_mm)
        arm.descend_to(start_x, start_y, push_z_mm)
        arm.move_to_xyz(target_x, target_y, push_z_mm, slow=True)
        return arm.lift_by(hover_z_mm - push_z_mm)
