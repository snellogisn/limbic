"""``pick`` — the composite grasp: hover, descend into the object, close, lift.

This is the canonical four-phase grasp, with the learned grasp rules baked in as
documented defaults so the planner only has to supply *where* the object is:

  1. Open the claw and hover at ``hover_z_mm`` above the target, at a clearance
     height so the transit can't clip the object or its neighbours.
  2. Descend slowly to the grasp height. Two learned rules shape this height:
       * **Descend INTO the object, not onto it.** The claw fingers need to
         straddle the object's sides, so the tip is driven ``grasp_depth_mm``
         *below* the object's top (top = ``object_height_mm``). Stopping at the
         top would only pinch the lid. The result is clamped to never go below
         ``min_grasp_z_mm`` so the fingers can't be jammed into the table.
       * **Account for the claw's lateral offset.** On this arm the closing claw
         pulls slightly to one side, so aiming the *tip* a few mm off the object
         centre lands the *grip* on centre. ``claw_y_offset_mm`` (default -10,
         i.e. 10mm toward -y) encodes that bias; set it to 0 for a symmetric claw.
  3. Close the claw onto the object.
  4. Lift back to the hover height so the object clears the table before any
     transit move — lift-before-retract is what stops a grasped object from
     being dragged across the surface.
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class Pick(Primitive):
    """Hover above an object, descend into it, close the claw, and lift it clear."""

    name = "pick"
    summary = "Grasp an object at (x, y): hover, descend into it, close, and lift."
    parameters: dict[str, dict[str, Any]] = {
        "x_mm": {
            "type": "number",
            "description": "Object centre x (forward, mm).",
        },
        "y_mm": {
            "type": "number",
            "description": "Object centre y (left, mm).",
        },
        "object_height_mm": {
            "type": "number",
            "description": "Height of the object's top above the table (mm). Default 25.",
            "default": 25.0,
        },
        "hover_z_mm": {
            "type": "number",
            "description": "Clearance height to hover at before/after the grasp (mm). Default 60.",
            "default": 60.0,
        },
        "grasp_depth_mm": {
            "type": "number",
            "description": "How far BELOW the object top to drive the tip so the "
            "fingers straddle it (mm). Default 20.",
            "default": 20.0,
        },
        "claw_y_offset_mm": {
            "type": "number",
            "description": "Lateral tip offset to compensate for the claw pulling to "
            "one side as it closes (mm in y). Default -10.",
            "default": -10.0,
        },
        "min_grasp_z_mm": {
            "type": "number",
            "description": "Table guard: never descend below this height (mm). Default 3.",
            "default": 3.0,
        },
    }

    def run(
        self,
        arm: RobotArm,
        x_mm: float,
        y_mm: float,
        object_height_mm: float = 25.0,
        hover_z_mm: float = 60.0,
        grasp_depth_mm: float = 20.0,
        claw_y_offset_mm: float = -10.0,
        min_grasp_z_mm: float = 3.0,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        # Aim the tip slightly off-centre so the offset claw grips on centre.
        grasp_y = y_mm + claw_y_offset_mm

        # Descend INTO the object (below its top), but never into the table.
        grasp_z = max(min_grasp_z_mm, object_height_mm - grasp_depth_mm)

        # 1. Open and hover clear above the target.
        arm.open_gripper()
        arm.reach_above(x_mm, grasp_y, height_mm=hover_z_mm)

        # 2. Descend slowly to the grasp height (precision profile).
        arm.descend_to(x_mm, grasp_y, grasp_z)

        # 3. Close the claw onto the object.
        arm.close_gripper()

        # 4. Lift clear before any transit so the object doesn't drag on the table.
        return arm.lift_by(hover_z_mm - grasp_z)
