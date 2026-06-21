"""``place`` — the composite release: carry, lower, open, retreat.

The mirror image of ``pick``. It assumes the claw is already holding an object
(typically straight after a ``pick``) and sets it down at a new ``(x, y)``:

  1. Carry the object to the destination at ``carry_z_mm`` — a clearance height
     high enough that the held object transits over the table and any obstacles
     without dragging or knocking them.
  2. Lower slowly to the release height, using the precision profile for a gentle
     set-down. The release height is computed to DROP FROM ABOVE, not to push the
     held object down into whatever it lands on (§0.6): the held object's bottom
     sits ~1 cm BELOW the claw tip, so to set it on a surface of height H the tip
     releases at about ``H + drop_offset_mm`` (~1.5 cm of object-below-claw plus a
     small clearance). E.g. stacking a 2.5 cm cube on another → support_height_mm
     25, tip releases at ~40 mm; on the bare table → ~15 mm. Better to drop from a
     touch too high than to ram the lower object.
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
    summary = (
        "Set a held object down at (x, y): carry, lower, open, and retreat. Set "
        "support_height_mm to the height of the surface you are placing ONTO (0 for "
        "the table, ~25 for stacking on a 2.5 cm cube) and it drops from just above "
        "that — it will NOT push the object down into the surface."
    )
    parameters: dict[str, dict[str, Any]] = {
        "x_mm": {
            "type": "number",
            "description": "Destination x (forward, mm).",
        },
        "y_mm": {
            "type": "number",
            "description": "Destination y (left, mm).",
        },
        "support_height_mm": {
            "type": "number",
            "description": "Height of the surface you are setting the object ONTO (mm): "
            "0 for the table, ~25 to stack on a 2.5 cm cube. The release height is "
            "computed from this so the object drops from just above the surface "
            "instead of being pushed into it. Default 0 (the table).",
            "default": 0.0,
        },
        "drop_offset_mm": {
            "type": "number",
            "description": "How far ABOVE the support the claw tip releases (mm): the "
            "held object's bottom sits ~1 cm below the tip, plus ~0.5 cm clearance so "
            "it drops rather than pushes down. Default 15.",
            "default": 15.0,
        },
        "release_z_mm": {
            "type": "number",
            "description": "Explicit tip height to release at (mm). Leave unset to "
            "compute it safely as support_height_mm + drop_offset_mm; only set this to "
            "override the computed drop height.",
            "default": None,
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
        support_height_mm: float = 0.0,
        drop_offset_mm: float = 15.0,
        release_z_mm: float | None = None,
        carry_z_mm: float = 70.0,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        # Drop from ABOVE the support (don't push the held object down into it):
        # release the claw at support height + the object-below-claw + clearance,
        # unless an explicit release_z_mm overrides it.
        if release_z_mm is None:
            release_z_mm = support_height_mm + drop_offset_mm

        # 1. Carry the held object over to the destination at clearance height.
        arm.reach_above(x_mm, y_mm, height_mm=carry_z_mm)

        # 2. Lower slowly to the release height (precision profile).
        arm.descend_to(x_mm, y_mm, release_z_mm)

        # 3. Release.
        arm.open_gripper()

        # 4. Retreat up before the next move, leaving the object settled in place.
        return arm.lift_by(carry_z_mm - release_z_mm)
