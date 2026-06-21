"""Sensory input: open-vocabulary object detection in TABLE-FRAME coordinates.

This is the **Part C composition** the architecture calls for (§0.3 #4 /
§0.4 #4 / §C.3): the Brain never sees pixels, it gets object positions in the
arm's own frame. Part B and Part A each own one half; this input *chains* them:

    camera frame  --(Part B: limbic.vision.detect)-->  [(label, (u, v)), ...]
                  --(Part A: control.localization)-->   [(label, (x, y)), ...]

So a single `sense_object_detections(prompt="red cup")` call gives the planner
the millimetre table coordinate it can hand straight to `move_to_xyz` / `pick`.

Why this lives in ``inputs/library/`` (and not a bespoke brain tool)
-------------------------------------------------------------------
Dropping it here means the registry auto-discovers it, so:
  * the planner automatically gains a ``sense_object_detections`` tool
    (``brain/tools.py`` generates one tool per input), and
  * it is automatically included in the perception ``snapshot()`` the verifier
    reads, and
  * it is exactly the ``object_detections`` sense the orchestrator's
    ``_vision_available()`` gate looks for — so registering this file is what
    FLIPS ON verification + retries (the brain only re-reasons when it has a
    real world-state feed; see ``orchestrator._VISION_SENSES``).

Multiple cameras (§A.5)
-----------------------
One OR several cameras are supported via ``$LIMBIC_CAMERAS`` (e.g. two USB-C
cameras as ``"0:A,1:B"`` — ``spec:role`` pairs, where ``role`` selects that
camera's calibration files). Each camera detects and localizes independently;
when more than one is configured, an object seen by several cameras is reported
ONCE, taking the reading from the camera physically CLOSEST to it (computed from
each camera's calibrated position — so the rule is mounting-agnostic and needs no
hardcoded left/right). We never average across cameras, per §A.5. An object both
cameras confirm is additionally STEREO-FUSED: its two box-centre rays are
triangulated to a parallax-free ``(x, y)`` and a top-of-object ``height_mm``
(§B.3), and its footprint is re-measured on that height plane — so coordinates
stay accurate for objects taller than the mat, which a single z=0 read inflates.

Graceful degradation + self-diagnosis (nothing here ever raises)
----------------------------------------------------------------
The full pipeline has three external prerequisites, any of which may be missing
on a fresh machine. This input reports each one instead of crashing:

  * ``ultralytics`` (the optional 'vision' extra, §0.4) — needed to run YOLO.
  * camera access — a camera must open and return a frame (on macOS the terminal
    needs Camera permission: System Settings → Privacy → Camera).
  * camera calibration — ``pixel_to_table`` needs intrinsics + extrinsics .npz
    per camera role (``$LIMBIC_CALIB_DIR``); without them we still return PIXELS,
    but cannot produce table coordinates.

A read with NO ``prompt`` is a HEALTH CHECK: it reports each prerequisite per
camera and sets ``ok`` True only when at least one camera can produce table-frame
detections. That is precisely what the verification gate needs: it keeps retries
OFF until the rig can genuinely perceive the workspace.
"""

from __future__ import annotations

import math
import os
import pathlib
from typing import Any

from limbic.inputs.base import Input

# Repo root (…/limbic) — three parents up from this file
# (inputs/library/object_detections.py).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _camera_offset_mm() -> tuple[float, float]:
    """Constant table-frame correction added to EVERY camera-derived (x, y) (mm).

    Measured on the rig: the localized camera coordinate reads ~20 mm too far
    FORWARD of where the arm actually reaches it (a camera coordinate of 250 mm is
    the arm's 230 mm), so we subtract 20 mm in +x by default. This only shifts
    coordinates that come from the camera — typed coordinates and the object-size
    geometry are untouched. Override per axis with ``$LIMBIC_CAM_OFFSET_X_MM`` /
    ``$LIMBIC_CAM_OFFSET_Y_MM`` (re-measure if the rig moves).
    """
    dx = float(os.environ.get("LIMBIC_CAM_OFFSET_X_MM", "-20.0"))
    dy = float(os.environ.get("LIMBIC_CAM_OFFSET_Y_MM", "0.0"))
    return dx, dy


def _calib_dir() -> pathlib.Path:
    """Directory holding the camera calibration .npz files.

    ``$LIMBIC_CALIB_DIR`` overrides; default is ``<repo>/calib`` — the same
    directory the calibration scripts write to (and that ``.gitignore`` excludes).
    The extrinsics here are produced by the (Part A) calibration scripts; this
    input only consumes them.
    """
    return pathlib.Path(os.environ.get("LIMBIC_CALIB_DIR", str(_REPO_ROOT / "calib")))


def _camera_configs() -> list[tuple["int | str", str]]:
    """Return the ``[(spec, role), ...]`` cameras to use, in priority order.

    Configured by environment so the same code serves one or many cameras:

      * ``$LIMBIC_CAMERAS = "0:A,1:B"`` — explicit ``spec:role`` pairs. ``spec``
        is a camera index or a name substring (resolved by ``open_camera``);
        ``role`` selects that camera's ``*_CAM_<role>.npz`` calibration. This is
        the two-USB-C-camera setup.
      * else single camera: ``$LIMBIC_CAMERA`` (index or name) with role
        ``$LIMBIC_CAM_ROLE`` (default "A").
      * else (no env at all): the calibrated DUAL RIG from
        ``limbic.control.calibration.CAMERAS`` — each role opened BY NAME (so a
        device-index shuffle doesn't break it), exactly like the dual-camera
        detector script. This is the default the moving-arm pipeline perceives
        with, so it cross-verifies across both cameras out of the box.
    """
    spec_list = os.environ.get("LIMBIC_CAMERAS")
    if spec_list:
        configs: list[tuple[int | str, str]] = []
        for pair in spec_list.split(","):
            pair = pair.strip()
            if not pair:
                continue
            spec, _, role = pair.partition(":")
            spec = spec.strip()
            role = role.strip() or "A"
            configs.append((int(spec) if spec.isdigit() else spec, role))
        if configs:
            return configs

    single = os.environ.get("LIMBIC_CAMERA")
    if single is not None:
        role = os.environ.get("LIMBIC_CAM_ROLE", "A")
        return [(int(single) if single.isdigit() else single, role)]

    # Default: the calibrated dual rig, addressed by camera NAME per role.
    try:
        from limbic.control.calibration import CAMERAS

        return [(cam["name"], role) for role, cam in CAMERAS.items()]
    except Exception:
        # If calibration config can't be imported, degrade to a single camera 0.
        return [(0, "A")]


def _get_detector():
    """Return the shared detection backend (Grounding DINO by default).

    Delegates to ``limbic.vision.get_detector``, which builds the model once per
    process and picks the backend from ``$LIMBIC_VISION_BACKEND`` (else DINO when
    ``transformers`` is installed, else YOLO-World). The import is lazy (the
    'vision' extra is optional, §0.4) so merely importing this module never
    requires torch — only actually detecting does.
    """
    from limbic.vision import get_detector  # lazy: optional 'vision' extra

    return get_detector()


def _load_calibration(role: str):
    """Load ``(intrinsics, extrinsics)`` for a camera role, or ``None``.

    Returns ``None`` (never raises) if the files are absent or unreadable — the
    caller then degrades to returning pixels without table coordinates.
    """
    from limbic.control.localization import load_camera

    try:
        return load_camera(role, _calib_dir())
    except Exception:
        return None


def _vision_installed() -> bool:
    """True if the active detection backend's deps are importable (no model build).

    Both backends need ``torch``; Grounding DINO additionally needs
    ``transformers`` and YOLO-World needs ``ultralytics``. We check the deps for
    whichever backend ``limbic.vision`` would actually select, so the health
    check matches what detection will really try to run.
    """
    import importlib.util

    def have(*mods: str) -> bool:
        return all(importlib.util.find_spec(m) is not None for m in mods)

    try:
        from limbic.vision import _default_backend

        backend = _default_backend()
    except Exception:
        backend = "dino"
    if backend == "yolo":
        return have("torch", "ultralytics")
    return have("torch", "transformers")


def _install_hint() -> str:
    """Backend-appropriate install instruction for the missing vision deps."""
    try:
        from limbic.vision import _default_backend

        backend = _default_backend()
    except Exception:
        backend = "dino"
    if backend == "yolo":
        return "the vision extra is not installed (torch + ultralytics). Install with: pip install ultralytics"
    return "the vision extra is not installed (torch + transformers). Install with: pip install transformers torch"


def _calib_image_size(role: str) -> tuple[int, int]:
    """``(width, height)`` the intrinsics for ``role`` were calibrated at.

    The whole pixel→table chain assumes the frame is the SAME size as the
    calibration image — otherwise ``pixel_to_table`` applies a camera matrix for
    one resolution to pixels from another and mislocates everything (this was the
    bug: a 640×480 default capture localized against 1280×720 intrinsics gave
    ~100 mm position errors). Read it from the intrinsics ``.npz`` (key
    ``image_size``); default to the rig's 1280×720 if unavailable.
    """
    try:
        import numpy as np

        d = np.load(str(_calib_dir() / f"intrinsics_CAM_{role}.npz"))
        if "image_size" in d.files:
            w, h = (int(x) for x in d["image_size"])
            return w, h
    except Exception:
        pass
    return 1280, 720


def _grab_frame(camera_spec, width: int | None = None, height: int | None = None):
    """Capture one BGR frame. Returns ``(frame, error)`` — exactly one is None."""
    try:
        import cv2  # noqa: F401  (presence check; used by open_camera)
    except ImportError:
        return None, "opencv-python is not installed. Install it with: pip install opencv-python"

    from limbic.platform_support import open_camera

    cap = None
    try:
        try:
            cap = open_camera(camera_spec, width=width, height=height)
        except (ImportError, RuntimeError) as exc:
            return None, str(exc)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None, (
                f"camera {camera_spec!r} opened but returned no frame "
                "(in use by another app, or — on macOS — Camera permission not "
                "granted to this terminal: System Settings → Privacy → Camera)."
            )
        return frame, None
    finally:
        if cap is not None:
            cap.release()


def _detect_one_camera(
    spec, role: str, prompt: str, conf: float | None, restrict_to_mat: bool
) -> tuple[list[dict[str, Any]], str | None, Any, Any]:
    """Detect + localize on a single camera. Returns ``(objects, error, frame, calib)``.

    Each object dict carries ``camera`` (the role), ``table_mm`` (the z=0 read, or
    None if that camera isn't calibrated), ``size_mm`` (real footprint when
    calibrated), ``height_mm`` (None here — only stereo fusion can fill it) and two
    INTERNAL fields, ``_cam_xy`` (the camera's own table-frame position, for the
    closest-camera merge) and ``_box`` (the pixel box, for re-measuring the
    footprint on a stereo height plane). Both ``_``-fields are stripped before the
    reading is returned to the model. ``frame`` (resized to the calibrated space)
    and ``calib`` ``(intr, extr)`` are returned so the caller can triangulate this
    camera's box-centre rays against another camera's (parallax-free fusion).
    """
    # Capture at the SAME resolution the intrinsics were calibrated at, so the
    # detected pixels live in the calibration's pixel space and pixel_to_table is
    # correct. (Opening without a size gives the driver default — usually 640×480 —
    # which is what broke localization in the web run.)
    calib_w, calib_h = _calib_image_size(role)
    frame, cam_err = _grab_frame(spec, width=calib_w, height=calib_h)
    if frame is None:
        return [], cam_err, None, None

    # Safety net: if the driver refused the requested size, resize to the calibrated
    # size BEFORE detecting so boxes/pixels/segmentation all share that space.
    fh, fw = frame.shape[:2]
    if (fw, fh) != (calib_w, calib_h):
        import cv2

        frame = cv2.resize(frame, (calib_w, calib_h))

    try:
        detector = _get_detector()
        # conf=None -> let the backend apply its own rig-tuned threshold (DINO and
        # YOLO-World score on different scales, so there is no single good default).
        kw = {} if conf is None else {"conf": float(conf)}
        detections = detector.detect_boxes(frame, prompt, **kw)
    except Exception as exc:
        return [], f"detection failed on camera {spec!r}: {exc}", frame, None

    # Optional workspace (gray-mat) filter — drops anything off the reachable mat.
    mask = None
    if restrict_to_mat:
        try:
            from limbic.vision.workspace import gray_mat_mask

            mask, _contour = gray_mat_mask(frame)
        except Exception:
            mask = None  # any failure -> no filter, never blocks detection

    calib = _load_calibration(role)
    cam_xy = None
    if calib is not None:
        _intr, extr = calib
        try:
            cam_xy = (float(extr.t_cam2base[0]), float(extr.t_cam2base[1]))
        except Exception:
            cam_xy = None

    objects: list[dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        if mask is not None:
            from limbic.vision.workspace import box_in_workspace

            if not box_in_workspace(mask, int(x1), int(y1), int(x2), int(y2)):
                continue
        u, v = det.center
        table_mm = None
        size_mm = None
        if calib is not None:
            intr, extr = calib
            try:
                from limbic.control.localization import pixel_to_table

                x_mm, y_mm = pixel_to_table(u, v, intr, extr)
                # NOTE: the measured camera->arm offset is NOT applied here. It is
                # applied ONCE at the very end of read(), so it lands on the FINAL
                # coordinate whether that's this z=0 read or a stereo-triangulated
                # (x, y) that overwrites it — otherwise the offset would be silently
                # dropped for cross-confirmed objects.
                table_mm = [round(x_mm, 1), round(y_mm, 1)]
            except Exception:
                table_mm = None  # ray/extrinsics issue — keep the pixel, drop coord
            try:
                # Real footprint (long, short) in mm — lets the brain check the
                # object fits the gripper and pick an approach. Best-effort.
                from limbic.vision.sizing import object_size_mm

                long_mm, short_mm = object_size_mm(frame, det.box, intr, extr)
                size_mm = [round(long_mm, 1), round(short_mm, 1)]
            except Exception:
                size_mm = None
        objects.append(
            {
                "label": det.label,
                "confidence": round(float(det.confidence), 3),
                "pixel": [round(u, 1), round(v, 1)],
                "table_mm": table_mm,
                "size_mm": size_mm,
                "height_mm": None,  # filled only by stereo fusion (see _merge_closest_camera)
                "camera": role,
                "_cam_xy": cam_xy,
                "_box": [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return objects, None, frame, calib


def _merge_closest_camera(
    objects: list[dict[str, Any]],
    merge_mm: float,
    calib_by_role: dict[str, Any] | None = None,
    frame_by_role: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Collapse multi-camera duplicates, keeping the CLOSEST camera's reading (§A.5).

    Same-label detections whose table positions fall within ``merge_mm`` are
    treated as one object; from that group we keep the single reading whose
    camera is physically nearest the object (by calibrated camera position) — we
    never average. Detections without a table coordinate can't be spatially
    matched, so they pass through untouched.

    For any group spanning MORE THAN ONE camera (a cross-verified object), the
    kept reading is then UPGRADED by stereo triangulation (:func:`_stereo_fix`):
    its z=0 ``table_mm`` is replaced with a parallax-free ``(x, y)``, ``height_mm``
    is recovered, and ``size_mm`` is re-measured on the object's own height plane.
    z=0 over-reads tall objects (the silhouette is the elevated top seen
    obliquely, §A.7/§B.3), so this is what makes the Brain's coordinate accurate
    for things taller than the mat. ``calib_by_role``/``frame_by_role`` map each
    camera role to its ``(intr, extr)`` and captured frame; without them the merge
    still works, just without the stereo upgrade.
    """
    calib_by_role = calib_by_role or {}
    frame_by_role = frame_by_role or {}

    located = [o for o in objects if o.get("table_mm") is not None]
    rest = [o for o in objects if o.get("table_mm") is None]

    used = [False] * len(located)
    merged: list[dict[str, Any]] = []
    for i, anchor in enumerate(located):
        if used[i]:
            continue
        group = [anchor]
        used[i] = True
        ax, ay = anchor["table_mm"]
        for j in range(i + 1, len(located)):
            if used[j] or located[j]["label"] != anchor["label"]:
                continue
            bx, by = located[j]["table_mm"]
            if math.hypot(ax - bx, ay - by) <= merge_mm:
                group.append(located[j])
                used[j] = True
        winner = _pick_closest(group)
        # Cross-verification (§dual-camera): an object seen by MORE THAN ONE
        # camera at the same (x, y) is "confirmed" — the brain can trust it and
        # treat a single-camera sighting (occlusion / glare / false positive)
        # with suspicion.
        winner["confirmed"] = len({o.get("camera") for o in group}) > 1
        if winner["confirmed"]:
            _stereo_fix(group, winner, calib_by_role, frame_by_role)
        merged.append(winner)

    for o in rest:
        o["confirmed"] = False  # no table coord -> can't cross-match

    # Drop the internal-only fields before returning to the model.
    for o in merged + rest:
        o.pop("_cam_xy", None)
        o.pop("_box", None)
    return merged + rest


def _stereo_fix(
    group: list[dict[str, Any]],
    winner: dict[str, Any],
    calib_by_role: dict[str, Any],
    frame_by_role: dict[str, Any],
) -> bool:
    """Upgrade a cross-camera group to a parallax-free ``(x, y)`` + height via stereo.

    Triangulates the two box-centre rays from two DIFFERENT cameras (§B.3): the
    closest-approach midpoint is a position free of the z=0 projection error that
    shifts tall objects outward, and its z is the object's top height. With the
    height known, the footprint is re-measured on the object's own plane (top ≈
    base for vertical sides) instead of z=0. Mutates ``winner`` in place and
    returns True on success; on any problem (missing calibration, parallel rays)
    it leaves the winner's z=0 reading untouched and returns False.
    """
    # One detection per camera, each needing calibration + a pixel to ray-cast.
    by_cam: dict[str, dict[str, Any]] = {}
    for d in group:
        role = d.get("camera")
        if role in calib_by_role and d.get("pixel") is not None and role not in by_cam:
            by_cam[role] = d
    if len(by_cam) < 2:
        return False

    (role_a, det_a), (role_b, det_b) = list(by_cam.items())[:2]
    intr_a, extr_a = calib_by_role[role_a]
    intr_b, extr_b = calib_by_role[role_b]
    try:
        from limbic.vision.sizing import pixel_ray, triangulate

        o_a, dir_a = pixel_ray(det_a["pixel"][0], det_a["pixel"][1], intr_a, extr_a)
        o_b, dir_b = pixel_ray(det_b["pixel"][0], det_b["pixel"][1], intr_b, extr_b)
        pt, resid = triangulate(o_a, dir_a, o_b, dir_b)
    except Exception:
        return False
    if pt is None:
        return False

    height = max(0.0, float(pt[2]))
    winner["table_mm"] = [round(float(pt[0]), 1), round(float(pt[1]), 1)]
    winner["height_mm"] = round(height, 1)
    winner["resid_mm"] = round(float(resid), 1)  # ray gap; large => suspect match/calib

    # Re-measure the footprint on the object's own height plane (drops the z=0
    # over-read), using the winner camera's frame + calibration when available.
    wrole = winner.get("camera")
    frame = frame_by_role.get(wrole)
    calib = calib_by_role.get(wrole)
    box = winner.get("_box")
    if frame is not None and calib is not None and box is not None:
        try:
            from limbic.vision.sizing import object_size_mm

            intr, extr = calib
            long_mm, short_mm = object_size_mm(frame, box, intr, extr, plane_z=height)
            winner["size_mm"] = [round(long_mm, 1), round(short_mm, 1)]
        except Exception:
            pass  # keep the z=0 footprint
    return True


def _pick_closest(group: list[dict[str, Any]]) -> dict[str, Any]:
    """From duplicate readings of one object, return the closest camera's (else best conf)."""
    if len(group) == 1:
        return group[0]

    def distance(o: dict[str, Any]) -> float:
        cam_xy = o.get("_cam_xy")
        if cam_xy is None or o.get("table_mm") is None:
            return math.inf
        ox, oy = o["table_mm"]
        return math.hypot(ox - cam_xy[0], oy - cam_xy[1])

    # Prefer the camera nearest the object; if none has a known position, the
    # most confident reading wins.
    if all(math.isinf(distance(o)) for o in group):
        return max(group, key=lambda o: o["confidence"])
    return min(group, key=distance)


class ObjectDetections(Input):
    """Detect named objects and report each one's TABLE-FRAME (x, y) in mm.

    With a ``prompt`` it returns detected objects; with no prompt it returns a
    health check of the detection pipeline (camera / model / calibration).
    Supports one or several cameras (``$LIMBIC_CAMERAS``).
    """

    name = "object_detections"
    summary = (
        "Find objects on the table by name and get their TABLE-FRAME position in "
        "millimetres, ready to pass to move_to_xyz/pick. Give `prompt` as the "
        "object name(s) to look for (comma-separated, e.g. 'red cup, blue block'); "
        "open-vocabulary, so any plain object name works. Returns a list of "
        "{label, confidence, pixel:[u,v], table_mm:[x,y], size_mm:[long,short], "
        "height_mm, camera, confirmed} — use table_mm to plan and size_mm to check "
        "the object fits the gripper; `confirmed` is True when more than one camera "
        "agrees on the object (trust those; treat single-camera sightings with more "
        "caution). For `confirmed` objects table_mm is parallax-corrected and "
        "height_mm (top-of-object height above the table, mm) is provided; "
        "single-camera objects report height_mm null and a table_mm that can read "
        "slightly outward for tall objects. "
        "Call this FIRST to locate anything you must grasp or move. With no prompt "
        "it returns a health check of the camera(s)/detector/calibration."
    )
    parameters: dict[str, dict[str, Any]] = {
        "prompt": {
            "type": "string",
            "description": (
                "Object name(s) to detect, comma-separated (e.g. 'red cup' or "
                "'cup, block, marker'). Open-vocabulary — no retraining needed. "
                "Omit to get a pipeline health check instead of detections."
            ),
            "default": None,
        },
        "conf": {
            "type": "number",
            "description": (
                "Minimum detection confidence 0..1. Omit to use the detector's "
                "rig-tuned default (Grounding DINO and YOLO-World score on "
                "different scales); raise it if you get false positives."
            ),
            "default": None,
        },
        "restrict_to_mat": {
            "type": "boolean",
            "description": (
                "Keep only objects fully inside the gray workspace mat (the arm's "
                "reachable area). True by default; set False to also see objects "
                "off the mat. Has no effect if the mat isn't found in the frame."
            ),
            "default": True,
        },
    }

    def read(
        self,
        *,
        prompt: str | None = None,
        conf: float | None = None,
        restrict_to_mat: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Detect objects across all configured cameras (or health-check) — never raises.

        With a prompt::

            {"ok": True, "prompt": <str>, "cameras": [<roles>], "calibrated": <bool>,
             "count": <int>,
             "objects": [{"label", "confidence", "pixel": [u, v],
                          "table_mm": [x, y] | None, "size_mm": [long, short] | None,
                          "height_mm": <float> | None, "camera": <role>,
                          "confirmed": <bool>, "resid_mm": <float, when stereo>}, ...],
             "note": <str, optional>}

        ``size_mm`` is the object's real footprint and ``confirmed`` is True when
        more than one camera agrees on it (cross-verification, §dual-camera). For a
        ``confirmed`` object the two cameras' box-centre rays are triangulated, so
        ``table_mm`` is parallax-free and ``height_mm`` (top height above the table)
        is filled (with ``resid_mm``, the ray-gap quality flag); a single-camera
        object keeps its z=0 ``table_mm`` and reports ``height_mm`` None.

        With NO prompt, a health check whose ``ok`` is True only when at least one
        camera can produce TABLE-FRAME detections (vision installed + that camera
        opens + that camera calibrated) — which is what the verification gate keys
        on.
        """
        configs = _camera_configs()

        # ---- Health check (no prompt): probe each camera, don't run detection. ----
        if prompt is None or (isinstance(prompt, str) and not prompt.strip()):
            vision_ok = _vision_installed()
            cams: list[dict[str, Any]] = []
            any_ready = False
            for spec, role in configs:
                frame, cam_err = _grab_frame(spec)
                camera_ok = frame is not None
                calibrated = _load_calibration(role) is not None
                cams.append(
                    {
                        "camera": spec,
                        "role": role,
                        "camera_ok": camera_ok,
                        "calibrated": calibrated,
                        "error": None if camera_ok else cam_err,
                    }
                )
                if vision_ok and camera_ok and calibrated:
                    any_ready = True
            missing = []
            if not vision_ok:
                missing.append(_install_hint())
            if not any_ready and vision_ok:
                missing.append(
                    "no camera is both reachable and calibrated "
                    f"(calibration dir {_calib_dir()})"
                )
            return {
                "ok": any_ready,
                "ready": any_ready,
                "vision_installed": vision_ok,
                "cameras": cams,
                "note": (
                    "detection pipeline ready"
                    if any_ready
                    else "not ready: " + "; ".join(missing or ["see per-camera status"])
                ),
            }

        # ---- Detection path. ----
        if not _vision_installed():
            return {"ok": False, "error": _install_hint()}

        all_objects: list[dict[str, Any]] = []
        errors: list[str] = []
        roles_used: list[str] = []
        # Kept per camera so cross-confirmed objects can be triangulated below.
        frame_by_role: dict[str, Any] = {}
        calib_by_role: dict[str, Any] = {}
        for spec, role in configs:
            objs, err, frame, calib = _detect_one_camera(spec, role, prompt, conf, restrict_to_mat)
            roles_used.append(role)
            if frame is not None:
                frame_by_role[role] = frame
            if calib is not None:
                calib_by_role[role] = calib
            if err:
                errors.append(err)
            all_objects.extend(objs)

        # Every camera failed -> surface the errors as a single failure.
        if not all_objects and errors:
            return {"ok": False, "error": "; ".join(errors)}

        # With multiple cameras, collapse duplicates to the closest camera (§A.5)
        # and stereo-triangulate the cross-confirmed ones (parallax-free x,y + height).
        if len(configs) > 1:
            merge_mm = float(os.environ.get("LIMBIC_CAM_MERGE_MM", "50"))
            objects = _merge_closest_camera(all_objects, merge_mm, calib_by_role, frame_by_role)
        else:
            for o in all_objects:
                o.pop("_cam_xy", None)
                o.pop("_box", None)
                o["confirmed"] = False  # single camera -> no cross-verification
            objects = all_objects

        # Apply the measured camera->arm correction ONCE, on the FINAL coordinate of
        # every object — after closest-camera merge and stereo triangulation — so it
        # lands the same whether table_mm is a single-camera z=0 read or a parallax-
        # free triangulated (x, y). (Matching/triangulation above run on the raw
        # coords; shifting both cameras equally afterwards doesn't change them.)
        off_x, off_y = _camera_offset_mm()
        if off_x or off_y:
            for o in objects:
                tm = o.get("table_mm")
                if tm is not None:
                    o["table_mm"] = [round(tm[0] + off_x, 1), round(tm[1] + off_y, 1)]

        calibrated = any(o.get("table_mm") is not None for o in objects)
        result: dict[str, Any] = {
            "ok": True,
            "prompt": prompt,
            "cameras": roles_used,
            "calibrated": calibrated,
            "count": len(objects),
            "objects": objects,
        }
        if not calibrated:
            result["note"] = (
                "NOT calibrated: pixel locations only, no table_mm. Add camera "
                f"calibration to {_calib_dir()} to get table coordinates."
            )
        if errors:
            result["camera_warnings"] = errors
        return result
