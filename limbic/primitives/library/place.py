"""``place`` — the composite release: carry, lower, open, retreat.

The mirror image of ``pick``. It assumes the claw is already holding an object
(typically straight after a ``pick``) and sets it down at a new ``(x, y)``:

  1. Carry the object to the destination at ``carry_z_mm`` — a clearance height
     high enough that the held object transits over the table and any obstacles
     without dragging or knocking them.
  2. Lower slowly to ``release_z_mm``, the height at which the object rests on the
     surface (or its drop target), using the precision profile for a gentle set-down.
  3. Open the claw to release.
  4. Retreat back up to the carry height before the next move — opening *then*
     lifting (rather than lifting while still gripping) leaves the object settled
     in place instead of snatching it back up or tipping it.
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class Place(Primitive):
    """Carry a held object to ``(x, y)``, lower, release, and retreat clear."""

    name = "place"
    summary = "Set a held object down at (x, y): carry, lower, open, and retreat."
    parameters: dict[str, dict[str, Any]] = {
        "x_mm": {
            "type": "number",
            "description": "Destination x (forward, mm).",
        },
        "y_mm": {
            "type": "number",
            "description": "Destination y (left, mm).",
        },
        "release_z_mm": {
            "type": "number",
            "description": "Height at which to release the object (mm). Default 10.",
            "default": 10.0,
        },
        "carry_z_mm": {
            "type": "number",
            "description": "Clearance height to carry/retreat at (mm). Default 70.",
            "default": 70.0,
        },
    }

    def run(
        self,
        arm: RobotArm,
        x_mm: float,
        y_mm: float,
        release_z_mm: float = 10.0,
        carry_z_mm: float = 70.0,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        # 1. Carry the held object over to the destination at clearance height.
        arm.reach_above(x_mm, y_mm, height_mm=carry_z_mm)

        # 2. Lower slowly to the release height (precision profile).
        arm.descend_to(x_mm, y_mm, release_z_mm)

        # 3. Release.
        arm.open_gripper()

        # 4. Retreat up before the next move, leaving the object settled in place.
        return arm.lift_by(carry_z_mm - release_z_mm)
