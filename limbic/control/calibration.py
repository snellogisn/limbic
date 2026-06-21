"""Measured bridge between the ikpy/URDF model (Stage 1, ``ik_chain``) and THIS
physical arm: the ikpy<->arm joint convention (sign + offset, §A.3), the table
frame<->ikpy frame offset (base height / pan axis, §A.6), and an empirical
top-down accuracy correction (§A.6) fit from a measured command/real sweep.

Every number below was handed over directly by the user as a trusted
measurement of this rig (the "Measured Constants" sheet) -- nothing here is
invented. Per CLAUDE.md §A.6: if a live measurement on the rig disagrees with
one of these, the measurement wins; update the constant, don't patch around it.

§2 (joint soft limits) and §3/§4 (workspace envelope, gripper scale, motion
profile timing) of that sheet already match ``safety.py`` / ``config.py``
exactly -- nothing to change there. This module covers what wasn't wired yet:
the ikpy<->arm conversion, the frame offset, and the accuracy correction.
Sections 6-8 (grasp/claw, push/stack, cameras) are recorded here for Stage 3/4
but not yet consumed by any primitive.
"""

from __future__ import annotations

import math
import os
import pathlib
from bisect import bisect_left

from .ik_chain import ACTIVE_JOINTS, geometry

# --------------------------------------------------------------------------- #
# §1 -- table frame (mm, origin under the pan axis at the table surface)
#       <-> ikpy frame (m, URDF base_link origin)
#
# The table frame is a 180-degree-yawed view of the ikpy base frame: ikpy
# "forward" at pan=0 is the base -X axis (shoulder_pan offset=0 is defined
# exactly on this), so table x/y are the pan axis position MINUS the ikpy
# x/y, not plus. z stays a simple additive offset (z is shared, up is up):
# the table surface sits BASE_HEIGHT_ABOVE_TABLE_MM below the base origin.
# --------------------------------------------------------------------------- #
BASE_HEIGHT_ABOVE_TABLE_MM = 103.4  # base/model origin sits this high above the table
PAN_AXIS_X_MM = 38.8                # cross-checked at runtime against ik_chain.geometry()


def table_to_ikpy_m(x_mm: float, y_mm: float, z_mm: float) -> tuple[float, float, float]:
    """Table-frame mm -> ikpy-frame metres."""
    px, py = geometry().pan_axis_xy
    return (
        px - x_mm / 1000.0,
        py - y_mm / 1000.0,
        (z_mm - BASE_HEIGHT_ABOVE_TABLE_MM) / 1000.0,
    )


def ikpy_to_table_mm(x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
    """ikpy-frame metres -> table-frame mm. Inverse of :func:`table_to_ikpy_m`."""
    px, py = geometry().pan_axis_xy
    return (
        (px - x_m) * 1000.0,
        (py - y_m) * 1000.0,
        z_m * 1000.0 + BASE_HEIGHT_ABOVE_TABLE_MM,
    )


# --------------------------------------------------------------------------- #
# §1 -- ikpy joint convention <-> arm joint convention
#       arm_deg = sign * ikpy_deg + offset   (order = ACTIVE_JOINTS)
# --------------------------------------------------------------------------- #
_SIGN = dict(zip(ACTIVE_JOINTS, (+1.0, -1.0, -1.0, -1.0, +1.0)))
_OFFSET_DEG = dict(zip(ACTIVE_JOINTS, (0.0, -90.5, -81.5, -7.6, 0.0)))


def ikpy_to_arm_deg(ikpy_rad: dict[str, float]) -> dict[str, float]:
    """ikpy/URDF joint angles (radians) -> this arm's joint convention (degrees)."""
    return {j: _SIGN[j] * math.degrees(ikpy_rad[j]) + _OFFSET_DEG[j] for j in ACTIVE_JOINTS}


def arm_to_ikpy_rad(arm_deg: dict[str, float]) -> dict[str, float]:
    """Inverse of :func:`ikpy_to_arm_deg`."""
    return {j: math.radians(_SIGN[j] * (arm_deg[j] - _OFFSET_DEG[j])) for j in ACTIVE_JOINTS}


# --------------------------------------------------------------------------- #
# §5 -- empirical accuracy correction. The real tip lands SHORT and LOW of the
# point the IK aims at (slack + gravity droop), so the correction aims FARTHER
# and HIGHER. Re-fit 2026-06-20 from a move-to-point / ruler sweep on THIS rig,
# then refined by a hardware ruler check at (160,0,60) [landed (165,0,84)] which
# pinned down the z slope. The old reach-dropoff table over-corrected z at near
# reach and under-extended reach -- it's gone, folded into the linear terms.
#
# Two functions, opposite directions -- be precise about which one is called:
#   * command_for_real(real_fwd, real_z, pitch_deg) -- the one to CALL to
#     move. Pass the desired REAL table position; get back the command (the
#     "aim") to feed the solver.
#   * real_for_command(aim_fwd, aim_z) -- forward fit, prediction/diagnostic
#     ONLY (where a given aim lands). Do NOT use this to drive the arm.
#
# Model (table-frame mm, measured at pitch -90, planar reach + z; the planar
# IK has ~0 model error, so "aim" = where the IK puts the model tip):
#   aim_fwd = 1.1582*real_fwd + 9.908
#   aim_z   = 0.7222*real_z   + 28.333
# Anchored on two careful on-centerline ruler checks -- (150,0,30)->(155,0,30)
# and (160,0,60)->(165,0,84) -- which pinned the z slope (the noisy first sweep
# had it ~2x too steep, over-aiming z high); reach also uses two far points for
# how droop grows with distance (resid <2mm). Reach and z are decoupled here.
#
# Pitch blend: the droop is full with the gripper vertical and fades to NONE
# (aim = real) as it tilts toward horizontal -- we only have vertical data, so
# identity is the safe default off-vertical. Full at pitch <= -88, none >= -82.
#
# Trust region (measured here -- OUTSIDE it, esp. far/low reach and off-
# centerline |y|>~100mm, the fit EXTRAPOLATES and is approximate): real fwd
# ~150-235mm, real z ~30-150mm, near the centerline.
#
# CONFIRM ON THE FIRST REAL MOVE: command a known centerline target and ruler-
# check the tip lands AT it. If it lands short/low, the sign is wrong -- stop
# and fix before further moves.
# --------------------------------------------------------------------------- #
_AIM_FWD_COEF = (1.1582, 0.0, 9.908)         # aim_fwd = a*real_fwd + b*real_z + c
_AIM_Z_COEF = (0.0, 0.7222, 28.333)          # aim_z   = a*real_fwd + b*real_z + c

_PITCH_FULL_DEG = -88.0  # correction is fully on at/beyond this (vertical)
_PITCH_NONE_DEG = -82.0  # correction is fully off at/beyond this


def _interp(xs: tuple[float, ...], ys: tuple[float, ...], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect_left(xs, x)
    if xs[i] == x:
        return ys[i]
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _z_blend(pitch_deg: float) -> float:
    """0 (no droop correction) .. 1 (full), based on how vertical the gripper is."""
    if pitch_deg <= _PITCH_FULL_DEG:
        return 1.0
    if pitch_deg >= _PITCH_NONE_DEG:
        return 0.0
    return (pitch_deg - _PITCH_NONE_DEG) / (_PITCH_FULL_DEG - _PITCH_NONE_DEG)


# --------------------------------------------------------------------------- #
# Fitted accuracy model (data-driven; see accuracy_model.py + calibrate_accuracy).
# If a fitted model file exists it SUPERSEDES the affine constants below; if not
# (or it fails to load) we fall back to them. Re-fit with calibrate_accuracy.py,
# then call reload_accuracy_model() (or restart) to pick it up.
# --------------------------------------------------------------------------- #
_ACCURACY_MODEL = None
_ACCURACY_MODEL_LOADED = False


def _accuracy_model_path() -> pathlib.Path:
    override = os.environ.get("LIMBIC_ACCURACY_MODEL")
    if override:
        return pathlib.Path(override)
    return pathlib.Path(__file__).resolve().parents[2] / "calibration" / "accuracy_model.json"


def _get_accuracy_model():
    global _ACCURACY_MODEL, _ACCURACY_MODEL_LOADED
    if not _ACCURACY_MODEL_LOADED:
        _ACCURACY_MODEL_LOADED = True
        try:
            from .accuracy_model import AccuracyModel

            _ACCURACY_MODEL = AccuracyModel.load(_accuracy_model_path())
        except Exception:
            _ACCURACY_MODEL = None
    return _ACCURACY_MODEL


def reload_accuracy_model():
    """Force a re-load of the fitted model (call after re-fitting)."""
    global _ACCURACY_MODEL_LOADED
    _ACCURACY_MODEL_LOADED = False
    return _get_accuracy_model()


def command_for_real(real_fwd_mm: float, real_z_mm: float, pitch_deg: float = -90.0) -> tuple[float, float]:
    """(aim_fwd, aim_z) to feed the IK so the tip lands at REAL (real_fwd, real_z).

    This is the one to call when driving the arm. Uses the fitted accuracy model
    if one is present, else the affine constants below. The droop correction is
    full with the gripper vertical and fades to identity (aim = real) as the
    approach tilts toward horizontal (no off-vertical data, so identity default).
    """
    model = _get_accuracy_model()
    if model is not None:
        return model.command_for_real(real_fwd_mm, real_z_mm, pitch_deg)

    f = _z_blend(pitch_deg)
    af0, af1, af2 = _AIM_FWD_COEF
    az0, az1, az2 = _AIM_Z_COEF
    aim_fwd_full = af0 * real_fwd_mm + af1 * real_z_mm + af2
    aim_z_full = az0 * real_fwd_mm + az1 * real_z_mm + az2
    aim_fwd = real_fwd_mm + f * (aim_fwd_full - real_fwd_mm)
    aim_z = real_z_mm + f * (aim_z_full - real_z_mm)
    return aim_fwd, aim_z


def real_for_command(aim_fwd_mm: float, aim_z_mm: float, pitch_deg: float = -90.0) -> tuple[float, float]:
    """Forward fit: where a given IK aim actually lands (inverse of the §5 map).

    DIAGNOSTIC / PREDICTION ONLY -- never use this to drive the arm. Inverts the
    full-correction 2x2 affine (the pitch blend is ignored here; diagnostic).
    """
    model = _get_accuracy_model()
    if model is not None:
        return model.real_for_command(aim_fwd_mm, aim_z_mm, pitch_deg)

    af0, af1, af2 = _AIM_FWD_COEF
    az0, az1, az2 = _AIM_Z_COEF
    det = af0 * az1 - af1 * az0
    df, dz = aim_fwd_mm - af2, aim_z_mm - az2
    real_fwd = (az1 * df - af1 * dz) / det
    real_z = (-az0 * df + af0 * dz) / det
    return real_fwd, real_z


# --------------------------------------------------------------------------- #
# §6 -- grasp / claw (grab & drop only -- NOT push). Stage 4 territory: not
# yet consumed by any primitive, recorded here so the numbers aren't lost.
# --------------------------------------------------------------------------- #
CLAW_Y_OFFSET_MM = -10.0     # aim 1 cm to the gripper's RIGHT of the object center
GRASP_DEPTH_MM = 20.0        # descend ~2 cm INTO the object
MIN_GRASP_Z_MM = 3.0         # never command the tip below this (table guard)
HOVER_Z_MM = 60.0            # approach/retreat height above the table
LIFT_MM = 100.0              # lift height after a pick
DEFAULT_OBJ_HEIGHT_MM = 25.0
PHASE_PAUSE_S = 0.4
GRASP_REACH_MIN_MM = 50.0    # refuse grasps inside this radius (steep tilt)

_CLAW_BACKOFF_REACH_MM = (150.0, 250.0)
_CLAW_BACKOFF_MM = (-20.0, -10.0)  # pull back along the reach direction


def claw_back_off_mm(reach_mm: float) -> float:
    return _interp(_CLAW_BACKOFF_REACH_MM, _CLAW_BACKOFF_MM, reach_mm)


# --------------------------------------------------------------------------- #
# §7 -- push / stack / knock-off / throw. Stage 4 territory: not yet wired.
# --------------------------------------------------------------------------- #
PUSH_BEHIND_MM = 35.0
PUSH_DIST_MM = 100.0
PUSH_Z_MM = 12.0
STACK_CARRY_CLEAR_MM = 50.0
KNOCKOFF_DIST_MM = 60.0
THROW_WINDUP_XYZ_MM = (130.0, 0.0, 55.0)
THROW_RELEASE_XYZ_MM = (275.0, 0.0, 195.0)
THROW_RELEASE_LEAD_S = 0.10  # open the gripper this early to cover its opening latency


# --------------------------------------------------------------------------- #
# §8 -- cameras / localization. Stage 3 territory: not yet wired.
# --------------------------------------------------------------------------- #
CAM_RIGHT_NAME = "Logitech Webcam C930e"
CAM_LEFT_NAME = "HD Pro Webcam C920"
APRILTAG_FAMILY = "36h11"
APRILTAG_SIZE_MM = 57.5
APRILTAG_RIGHT_XYZ_MM = (60.0, -145.0, 5.0)  # id 0
APRILTAG_LEFT_XYZ_MM = (60.0, 145.0, 5.0)    # id 1

# One registry both the localization loader and the extrinsics script read, so
# role <-> camera name <-> tag never drift. Roles A/B match the .npz filename
# suffix (intrinsics_CAM_A.npz, ...). side is the §8 camera-selection side.
CAMERAS: dict[str, dict] = {
    "A": {"name": CAM_RIGHT_NAME, "side": "RIGHT", "tag_id": 0, "tag_xyz_mm": APRILTAG_RIGHT_XYZ_MM},
    "B": {"name": CAM_LEFT_NAME,  "side": "LEFT",  "tag_id": 1, "tag_xyz_mm": APRILTAG_LEFT_XYZ_MM},
}


def camera_for_y(y_mm: float) -> str:
    """Pick the camera on the side the target is closest to (§8 rule)."""
    return CAM_LEFT_NAME if y_mm > 0 else CAM_RIGHT_NAME


def camera_role_for_y(y_mm: float) -> str:
    """Role ('A'/'B') of the camera on the side the target is closest to (§8)."""
    return "B" if y_mm > 0 else "A"
