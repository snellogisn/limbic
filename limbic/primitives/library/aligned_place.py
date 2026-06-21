"""``aligned_place`` — a place that VISUALLY corrects its aim before releasing.

The drop-side mirror of ``aligned_pick``. Setting an object down is just as much a
precision step as picking one up: if the arm one-shots the place open-loop and the
wrist is tilted, it releases IN FRONT OF the intended spot (or off the cube it was
meant to stack on) and the object lands wrong. So this primitive wedges a
look-before-you-drop step between the carry and the lower:

  1. Carry the held object to the destination at clearance height.
  2. While hovering there, take an overhead screenshot and ask the vision model how
     far off the destination it is, then nudge — a few damped looks until it is
     reasonably centred (``limbic.vision.visual_align.align_to_object``; the same
     convergent loop aligned_pick uses, so it self-corrects without over-adjusting).
  3. Lower STRAIGHT DOWN to the computed release height and open — dropping from
     above the support, never pushing the object down into it (§0.6 / like ``place``).
  4. Retreat up, leaving the object settled.

Align to ``target_label`` — for a STACK that's the object you're placing ONTO (e.g.
"the bottom cube"); for a marked spot, that landmark. Graceful: with no camera, no
calibration, or no API key the visual step applies no correction (logged) and this
degrades to an ordinary straight-down ``place`` — always safe to use.
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm
from ... import runlog


class AlignedPlace(Primitive):
    """Place a held object at ``(x, y)`` with a camera-in-the-loop correction first."""

    name = "aligned_place"
    summary = (
        "Like place, but FIRST stops over the destination, takes a camera "
        "screenshot, and asks the vision model how to nudge so it's centred on the "
        "drop spot — then lowers straight down and releases. Use for precision drops "
        "and stacking (pass target_label = what you're placing onto, e.g. 'the "
        "bottom cube'). Set support_height_mm to the surface height you place onto "
        "(0 table, ~25 on one cube). Falls back to an ordinary place if no camera/API."
    )
    parameters: dict[str, dict[str, Any]] = {
        "x_mm": {"type": "number", "description": "Destination x (forward, mm)."},
        "y_mm": {"type": "number", "description": "Destination y (left, mm)."},
        "target_label": {
            "type": "string",
            "description": "What to centre on at the destination (e.g. 'the bottom cube' for a "
            "stack, or a landmark) — given to the vision model so it aligns to the right thing.",
            "default": "the placement spot",
        },
        "support_height_mm": {
            "type": "number",
            "description": "Height of the surface you set the object ONTO (mm). ACCOUNT FOR CUBE "
            "HEIGHT: each cube is 2.5 cm, so 0 for the bare table, ~25 to stack on one cube, ~50 "
            "on a stack of two. The release height is computed from this so the object drops from "
            "above the surface instead of being pushed into it. Default 0 (the table).",
            "default": 0.0,
        },
        "drop_offset_mm": {
            "type": "number",
            "description": "How far ABOVE the support the claw tip releases (mm): object bottom "
            "sits ~1 cm below the tip, plus ~1.5 cm clearance. Default 25.",
            "default": 25.0,
        },
        "release_z_mm": {
            "type": "number",
            "description": "Explicit tip height to release at (mm). Leave unset to compute it as "
            "support_height_mm + drop_offset_mm; only set to override.",
            "default": None,
        },
        "carry_z_mm": {
            "type": "number",
            "description": "Clearance height to carry/align/retreat at (mm). Default 70.",
            "default": 70.0,
        },
        "max_align_iters": {
            "type": "integer",
            "description": "Max visual-correction looks (camera+API+nudge) before releasing. Default 3.",
            "default": 3,
        },
        "align_tolerance_mm": {
            "type": "number",
            "description": "Consider the aim good enough once within this many mm of centred. Default 6.",
            "default": 6.0,
        },
    }

    def run(
        self,
        arm: RobotArm,
        x_mm: float,
        y_mm: float,
        target_label: str = "the placement spot",
        support_height_mm: float = 0.0,
        drop_offset_mm: float = 25.0,
        release_z_mm: float | None = None,
        carry_z_mm: float = 70.0,
        max_align_iters: int = 3,
        align_tolerance_mm: float = 6.0,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        log = runlog.current()
        save_dir = getattr(log, "run_dir", None)

        # Drop from ABOVE the support (don't push the held object into it).
        if release_z_mm is None:
            release_z_mm = support_height_mm + drop_offset_mm

        # 1. Carry the held object over to the destination at clearance height.
        arm.reach_above(x_mm, y_mm, height_mm=carry_z_mm)

        # 2. Visual closed-loop correction (look before you drop). Never raises;
        #    returns the original (x, y) unchanged if vision isn't available.
        from ...vision.visual_align import align_to_object

        result = align_to_object(
            arm, x_mm, y_mm,
            hover_z_mm=carry_z_mm,
            target_label=target_label,
            max_iters=max_align_iters,
            tolerance_mm=align_tolerance_mm,
            save_dir=save_dir,
        )
        log.data("visual_align", result)
        log.thought(
            "visual_align",
            message=(
                f"drop aligned at ({result['x_mm']:.0f},{result['y_mm']:.0f}) "
                f"after {result['iters']} look(s)"
                if result["adjusted"] or result["aligned"]
                else f"no correction applied ({result.get('note') or 'aligned on first look'})"
            ),
            **{k: result[k] for k in ("aligned", "adjusted", "iters", "note", "calibrated")},
        )

        cx, cy = result["x_mm"], result["y_mm"]

        # 3. Lower STRAIGHT DOWN to the release height (precision profile) and release.
        arm.descend_to(cx, cy, release_z_mm)
        arm.open_gripper()

        # 4. Retreat up, leaving the object settled in place.
        return arm.lift_by(carry_z_mm - release_z_mm)
