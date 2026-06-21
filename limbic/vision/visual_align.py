"""Visual closed-loop grasp correction (visual servoing).

The detector + localization gives a good-but-imperfect table coordinate for an
object. When the arm hovers over that coordinate the gripper is often a centimetre
or two off — enough to miss a small object. This module CLOSES THE LOOP with the
eye the brain already has: at the point of interest it

    1. STOPS and captures a literal screenshot from the overhead camera,
    2. annotates it with the gripper's current aim point and the table-frame
       +x / +y axis directions (so image directions map to table mm),
    3. sends that image to Claude (vision) and asks for a millimetre correction
       (``dx_mm, dy_mm``) that would centre the gripper on the object,
    4. nudges the arm by that correction and repeats until Claude says it is
       aligned (or a small iteration budget runs out).

It is deliberately STANDALONE and INJECTABLE: pass a ``client`` (any object with
``messages.create``) to test it offline, or let it build a real
``anthropic.Anthropic()``. Every external prerequisite is optional — no camera,
no calibration, or no API key each degrade to "no correction applied" with a
note, never an exception, so a caller (the ``aligned_pick`` primitive) can always
fall back to an ordinary grasp.

This is the §0.6 "descend-to-grasp is a precision move" rule taken one step
further: look before you grab.
"""

from __future__ import annotations

import base64
import os
from typing import Any

# A vision-capable model (multimodal). Matches the brain's capable model so the
# whole stack speaks one model id; override per-call if needed.
DEFAULT_ALIGN_MODEL = "claude-opus-4-8"

# Defaults for the correction loop — small, because each iteration is a real
# camera capture + an API round-trip + an arm move.
DEFAULT_MAX_ITERS = 3
DEFAULT_TOLERANCE_MM = 4.0     # |dx|,|dy| below this => already aligned, stop
DEFAULT_MAX_STEP_MM = 40.0     # clamp any single suggested nudge (safety)


# --------------------------------------------------------------------------- #
# Frame capture + annotation
# --------------------------------------------------------------------------- #
def grab_frame(camera_spec):
    """Capture one BGR frame from ``camera_spec``. Returns ``(frame, error)``.

    Exactly one of the two is None. ``camera_spec`` is an index or a name
    substring, resolved by :func:`limbic.platform_support.open_camera`.
    """
    try:
        import cv2  # noqa: F401  (presence check; used by open_camera)
    except ImportError:
        return None, "opencv-python is not installed (pip install opencv-python)"

    from limbic.platform_support import open_camera

    cap = None
    try:
        try:
            cap = open_camera(camera_spec)
        except (ImportError, RuntimeError) as exc:
            return None, str(exc)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None, (
                f"camera {camera_spec!r} opened but returned no frame "
                "(in use, or — macOS — Camera permission not granted)."
            )
        return frame, None
    finally:
        if cap is not None:
            cap.release()


def annotate_aim(frame, aim_xy, intr, extr, *, axis_len_mm: float = 40.0):
    """Draw the aim point + table-frame +x/+y axes on a copy of ``frame``.

    Projects the table point ``aim_xy`` (and short +x/+y vectors from it) into the
    image with :func:`limbic.control.localization.table_to_pixel`, so the vision
    model can see WHERE the gripper is aimed and WHICH way ``dx_mm``/``dy_mm``
    point. Returns ``(annotated_frame, ok)``; ``ok`` is False (frame returned
    unchanged) if projection fails — the caller then sends the raw frame.
    """
    import cv2 as cv

    from limbic.control.localization import table_to_pixel

    try:
        ax, ay = aim_xy
        u0, v0 = table_to_pixel(ax, ay, intr, extr)
        ux, vx = table_to_pixel(ax + axis_len_mm, ay, intr, extr)  # +x (forward)
        uy, vy = table_to_pixel(ax, ay + axis_len_mm, intr, extr)  # +y (left)
    except Exception:
        return frame, False

    out = frame.copy()
    p0 = (int(round(u0)), int(round(v0)))
    # Aim reticle (magenta crosshair + ring).
    cv.drawMarker(out, p0, (255, 0, 255), cv.MARKER_CROSS, 26, 2)
    cv.circle(out, p0, 16, (255, 0, 255), 2)
    # +x axis arrow (red) and +y axis arrow (green), labelled.
    cv.arrowedLine(out, p0, (int(round(ux)), int(round(vx))), (0, 0, 255), 2, tipLength=0.3)
    cv.arrowedLine(out, p0, (int(round(uy)), int(round(vy))), (0, 200, 0), 2, tipLength=0.3)
    cv.putText(out, "+x", (int(round(ux)) + 4, int(round(vx))), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.putText(out, "+y", (int(round(uy)) + 4, int(round(vy))), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    cv.putText(out, "AIM", (p0[0] + 18, p0[1] - 14), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return out, True


def _encode_png(frame) -> str | None:
    """BGR frame -> base64 PNG string for the Anthropic image block (or None)."""
    try:
        import cv2 as cv

        ok, buf = cv.imencode(".png", frame)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# The single vision call: image -> suggested correction
# --------------------------------------------------------------------------- #
_ADJUST_TOOL = {
    "name": "report_adjustment",
    "description": (
        "Report how to move the gripper so it is centred over the target object "
        "for a top-down grasp, judging from the overhead image."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "dx_mm": {
                "type": "number",
                "description": "Move the gripper this many mm along +x (the red arrow / forward). Negative = backward.",
            },
            "dy_mm": {
                "type": "number",
                "description": "Move the gripper this many mm along +y (the green arrow / left). Negative = right.",
            },
            "aligned": {
                "type": "boolean",
                "description": "True if the aim point (magenta reticle) is already centred on the object within a few mm — no move needed.",
            },
            "found": {
                "type": "boolean",
                "description": "True if you can actually see the target object in the frame.",
            },
            "reason": {"type": "string", "description": "Brief: where the object is relative to the reticle."},
        },
        "required": ["dx_mm", "dy_mm", "aligned", "found"],
    },
}


def _build_client():
    """Build a real Anthropic client, or return ``(None, reason)`` if unavailable."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "ANTHROPIC_API_KEY not set — skipping visual correction"
    try:
        import anthropic
    except ImportError:
        return None, "anthropic SDK not installed — skipping visual correction"
    return anthropic.Anthropic(), None


def request_adjustment(
    frame,
    *,
    target_label: str,
    aim_xy: tuple[float, float],
    calibrated: bool,
    client: Any = None,
    model: str = DEFAULT_ALIGN_MODEL,
) -> dict[str, Any]:
    """Ask Claude (vision) for a mm correction to centre the gripper on the object.

    Returns ``{"dx_mm", "dy_mm", "aligned", "found", "reason", "ok"}``. On any
    failure (no key, no client, encode error, model returned nothing) ``ok`` is
    False and ``dx_mm == dy_mm == 0`` so the caller applies no move.
    """
    fail = {"dx_mm": 0.0, "dy_mm": 0.0, "aligned": False, "found": False, "ok": False}

    b64 = _encode_png(frame)
    if b64 is None:
        return {**fail, "reason": "could not encode frame"}

    used_client = client
    if used_client is None:
        used_client, reason = _build_client()
        if used_client is None:
            return {**fail, "reason": reason}

    axis_help = (
        "The magenta reticle (AIM) marks where the gripper is currently aimed. The "
        "red arrow is table +x, the green arrow is table +y; both are 40 mm long, so "
        "use them to gauge distances in millimetres."
        if calibrated
        else "This frame is NOT calibrated, so no axis arrows are shown — give your "
        "best estimate of the correction and lower your confidence."
    )
    text = (
        f"Overhead view of a robot gripper about to grasp '{target_label}' on a table. "
        f"The gripper is aimed at table coordinate (x={aim_xy[0]:.0f} mm, y={aim_xy[1]:.0f} mm). "
        f"{axis_help} "
        "Find the target object and report dx_mm, dy_mm (table-frame) to move the "
        "gripper so it is centred on the object for a clean top-down grasp. If the "
        "reticle is already on the object, set aligned=true. Call report_adjustment."
    )

    try:
        resp = used_client.messages.create(
            model=model,
            max_tokens=1024,
            tools=[_ADJUST_TOOL],
            tool_choice={"type": "tool", "name": "report_adjustment"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": text},
            ]}],
        )
    except Exception as exc:
        return {**fail, "reason": f"vision call failed: {exc}"}

    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "report_adjustment":
            data = dict(block.input or {})
            return {
                "dx_mm": float(data.get("dx_mm", 0.0)),
                "dy_mm": float(data.get("dy_mm", 0.0)),
                "aligned": bool(data.get("aligned", False)),
                "found": bool(data.get("found", False)),
                "reason": data.get("reason", ""),
                "ok": True,
            }
    return {**fail, "reason": "model returned no adjustment"}


# --------------------------------------------------------------------------- #
# The loop: hover -> capture -> correct -> repeat
# --------------------------------------------------------------------------- #
def _clamp_step(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def align_to_object(
    arm,
    x_mm: float,
    y_mm: float,
    *,
    hover_z_mm: float,
    target_label: str = "the object",
    camera_spec=None,
    role: str | None = None,
    calib_dir=None,
    client: Any = None,
    model: str = DEFAULT_ALIGN_MODEL,
    max_iters: int = DEFAULT_MAX_ITERS,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    max_step_mm: float = DEFAULT_MAX_STEP_MM,
    save_dir=None,
) -> dict[str, Any]:
    """Visually centre the gripper over the object near ``(x_mm, y_mm)``.

    Assumes the arm is already hovering at ``hover_z_mm`` above ``(x_mm, y_mm)``.
    Captures the camera, asks Claude for a correction, nudges the arm, and repeats
    up to ``max_iters`` times. Returns the corrected aim::

        {"x_mm", "y_mm", "aligned": bool, "iters": int, "adjusted": bool,
         "history": [...], "note": str|None}

    Never raises. If the camera/calibration/API are unavailable it returns the
    input ``(x_mm, y_mm)`` unchanged with ``adjusted=False`` and a ``note``.
    """
    # Kill-switch: when disabled, do NOT re-look with the camera — use the aim
    # the detector already gave and grasp open-loop. This is what makes the whole
    # run single-shot: the camera is used once (initial detection), then the plan
    # executes start-to-finish with no mid-grasp visual correction.
    if os.environ.get("LIMBIC_VISUAL_ALIGN", "1").strip().lower() in ("0", "false", "off", "no"):
        return {
            "x_mm": float(x_mm), "y_mm": float(y_mm),
            "aligned": False, "iters": 0, "adjusted": False, "history": [],
            "note": "visual alignment disabled (LIMBIC_VISUAL_ALIGN=0) — open-loop grasp",
            "camera": None, "calibrated": False,
        }

    from limbic.control import calibration

    # Resolve which camera to use (the one on the object's side, §A.5).
    if camera_spec is None:
        try:
            camera_spec = calibration.camera_for_y(y_mm)
            if role is None:
                role = calibration.camera_role_for_y(y_mm)
        except Exception:
            camera_spec = 0
    if calib_dir is None:
        calib_dir = os.environ.get("LIMBIC_CALIB_DIR", None)
        if calib_dir is None:
            import pathlib

            calib_dir = pathlib.Path(__file__).resolve().parents[2] / "calib"

    # Load calibration for annotation (optional — falls back to a raw frame).
    intr = extr = None
    if role is not None:
        try:
            from limbic.control.localization import load_camera

            intr, extr = load_camera(role, calib_dir)
        except Exception:
            intr = extr = None
    calibrated = intr is not None and extr is not None

    history: list[dict[str, Any]] = []
    cx, cy = float(x_mm), float(y_mm)
    aligned = False
    note: str | None = None

    for i in range(1, max_iters + 1):
        frame, cam_err = grab_frame(camera_spec)
        if frame is None:
            note = cam_err
            break

        if calibrated:
            shot, _ok = annotate_aim(frame, (cx, cy), intr, extr)
        else:
            shot = frame

        # Persist the literal screenshot the model is reasoning over (for the demo
        # UI / the run log), if a directory was given.
        saved_to = None
        if save_dir is not None:
            try:
                import cv2 as cv

                saved_to = str(save_dir / f"align_{i}.png")
                cv.imwrite(saved_to, shot)
            except Exception:
                saved_to = None

        adj = request_adjustment(
            shot, target_label=target_label, aim_xy=(cx, cy),
            calibrated=calibrated, client=client, model=model,
        )
        step = {"iter": i, "aim": [round(cx, 1), round(cy, 1)], "saved_to": saved_to, **adj}
        history.append(step)

        if not adj["ok"]:
            note = adj.get("reason")
            break

        dx = _clamp_step(adj["dx_mm"], max_step_mm)
        dy = _clamp_step(adj["dy_mm"], max_step_mm)

        if adj["aligned"] or (abs(dx) <= tolerance_mm and abs(dy) <= tolerance_mm):
            aligned = True
            break

        # Apply the correction and re-hover at the new aim point.
        cx += dx
        cy += dy
        arm.reach_above(cx, cy, height_mm=hover_z_mm)

    return {
        "x_mm": cx,
        "y_mm": cy,
        "aligned": aligned,
        "iters": len(history),
        "adjusted": (abs(cx - x_mm) > 1e-6 or abs(cy - y_mm) > 1e-6),
        "history": history,
        "note": note,
        "camera": str(camera_spec),
        "calibrated": calibrated,
    }
