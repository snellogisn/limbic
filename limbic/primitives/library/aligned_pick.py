"""``aligned_pick`` — a grasp that VISUALLY corrects its aim before closing.

Same four-phase grasp as ``pick`` (hover, descend-into, close, lift), but with a
look-before-you-grab step wedged between the hover and the descend: while hovering
over the detector's reported coordinate, it takes a literal screenshot from the
overhead camera and asks Claude (vision) how far the gripper is off the object,
then nudges the arm by that millimetre correction and re-checks — up to a couple
of iterations — until it is centred. This fixes the "the detector is close but not
precise enough to grasp a small object" problem.

It is SELF-CONTAINED on purpose. The brain submits one static plan, so a separate
"align" step couldn't hand a corrected coordinate to a following "pick" — the
correction and the grasp must live in one primitive. The visual loop itself lives
in :mod:`limbic.vision.visual_align`; this primitive just sequences it with the
grasp motion.

Graceful by design: if there is no camera, no calibration, or no API key, the
visual step applies no correction (with a logged note) and this degrades to an
ordinary ``pick`` — so it is always safe to use.
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm
from ... import runlog


class AlignedPick(Primitive):
    """Grasp at (x, y) with a camera-in-the-loop correction before the descent."""

    name = "aligned_pick"
    summary = (
        "Like pick, but FIRST stops over the target, takes a camera screenshot, "
        "and asks the vision model how to nudge the arm so it's precisely centred "
        "on the object — then descends and grasps. Use this for precision grasps "
        "of small objects when the detected position may be slightly off. Falls "
        "back to an ordinary grasp if no camera/API is available."
    )
    parameters: dict[str, dict[str, Any]] = {
        "x_mm": {"type": "number", "description": "Object centre x (forward, mm)."},
        "y_mm": {"type": "number", "description": "Object centre y (left, mm)."},
        "target_label": {
            "type": "string",
            "description": "What the object is (e.g. 'red cube') — given to the vision model so it aligns to the right thing.",
            "default": "the object",
        },
        "object_height_mm": {
            "type": "number",
            "description": "Height of the object's top above the table (mm). Default 25.",
            "default": 25.0,
        },
        "hover_z_mm": {
            "type": "number",
            "description": "Clearance height to hover/align at before the grasp (mm). Default 60.",
            "default": 60.0,
        },
        "grasp_depth_mm": {
            "type": "number",
            "description": "How far BELOW the object top to drive the tip so the fingers straddle it (mm). Reach ~7.5 mm (0.75 cm) INTO the object; try not to go deeper. Default 7.5.",
            "default": 7.5,
        },
        "claw_y_offset_mm": {
            "type": "number",
            "description": "Lateral tip offset for the claw pulling to one side as it closes (mm in y). Default 0 (no offset).",
            "default": 0.0,
        },
        "min_grasp_z_mm": {
            "type": "number",
            "description": "Table guard: never descend below this height (mm). Default 3.",
            "default": 3.0,
        },
        "max_align_iters": {
            "type": "integer",
            "description": "Max visual-correction iterations (camera+API+nudge) before grasping. Default 3.",
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
        target_label: str = "the object",
        object_height_mm: float = 25.0,
        hover_z_mm: float = 60.0,
        grasp_depth_mm: float = 7.5,
        claw_y_offset_mm: float = 0.0,
        min_grasp_z_mm: float = 3.0,
        max_align_iters: int = 3,
        align_tolerance_mm: float = 6.0,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        log = runlog.current()
        save_dir = getattr(log, "run_dir", None)

        # 1. Open and hover with the TIP aimed at the object centre (no claw offset
        #    yet) so the camera can judge tip-vs-object cleanly.
        arm.open_gripper()
        arm.reach_above(x_mm, y_mm, height_mm=hover_z_mm)

        # 2. Visual closed-loop correction (look before you grab). Never raises;
        #    returns the original (x, y) unchanged if vision isn't available.
        from ...vision.visual_align import align_to_object

        result = align_to_object(
            arm, x_mm, y_mm,
            hover_z_mm=hover_z_mm,
            target_label=target_label,
            max_iters=max_align_iters,
            tolerance_mm=align_tolerance_mm,
            save_dir=save_dir,
        )
        log.data("visual_align", result)
        log.thought(
            "visual_align",
            message=(
                f"aligned at ({result['x_mm']:.0f},{result['y_mm']:.0f}) "
                f"after {result['iters']} look(s)"
                if result["adjusted"] or result["aligned"]
                else f"no correction applied ({result.get('note') or 'aligned on first look'})"
            ),
            **{k: result[k] for k in ("aligned", "adjusted", "iters", "note", "calibrated")},
        )

        cx, cy = result["x_mm"], result["y_mm"]

        # 3. Apply the claw's lateral bias for the descent and grasp into the object.
        grasp_y = cy + claw_y_offset_mm
        grasp_z = max(min_grasp_z_mm, object_height_mm - grasp_depth_mm)
        arm.descend_to(cx, grasp_y, grasp_z)
        arm.close_gripper()

        # 4. Lift clear before any transit.
        return arm.lift_by(hover_z_mm - grasp_z)
