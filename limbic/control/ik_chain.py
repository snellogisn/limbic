"""SO-101 forward kinematics + geometry, built on the ikpy chain (Part A, §A.2).

``build_chain()`` / ``fk()`` — forward kinematics via ikpy (pure Python, runs on
any arch). This is the trusted FK reference: the rest of the bridge (the
table<->ikpy frame conversion and FK read-back in ``kinematics``) is built on it.

``geometry()`` extracts the planar link lengths / rest angles / pan axis ONCE
from this FK chain — nothing is hardcoded — so anything derived from it stays
consistent with FK by construction.

Reaching IK is NOT here: it lives in ``_prep_planar_ik.PlanarSO101IK``, the
hardware-validated deterministic closed-form planar solver, wired up in
``kinematics.solve_ik``. (ikpy's own numerical ``inverse_kinematics`` is avoided
for reaching because it branch-jumps non-deterministically.)

FRAMES & UNITS in THIS module:
  * Everything here is in the ikpy/URDF frame and the URDF joint convention
    (RADIANS, base_link origin). The mapping to the real table frame and to the
    arm's degree convention (per-joint sign + offset, base height) is MEASURED on
    the physical arm in Stage 2 — it does not belong here.
  * The 5 active joints, in chain order:
        shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
  * Approach pitch (radians): elevation of the gripper approach axis. -pi/2 =
    straight down (top-down grasp); 0 = horizontal forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ACTIVE_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
# ikpy chain links: [base, *ACTIVE_JOINTS, gripper_frame(fixed tip)] = 7 links.
_ACTIVE_MASK = [False, True, True, True, True, True, False]

URDF_PATH = Path(__file__).resolve().parents[2] / "assets" / "so101" / "so101_new_calib.urdf"


@lru_cache(maxsize=1)
def build_chain():
    """Build (and cache) the ikpy chain for the 5 active joints, tip = gripper frame."""
    import warnings

    from ikpy.chain import Chain

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Chain.from_urdf_file(
            str(URDF_PATH),
            base_elements=["base_link"],
            active_links_mask=_ACTIVE_MASK,
            name="so101",
        )


def _full_q(angles: dict[str, float] | list[float]) -> list[float]:
    """Expand the 5 active-joint angles (rad) to ikpy's 7-element vector."""
    if isinstance(angles, dict):
        active = [angles.get(j, 0.0) for j in ACTIVE_JOINTS]
    else:
        active = list(angles)
    return [0.0, *active, 0.0]


def fk(angles: dict[str, float] | list[float]) -> np.ndarray:
    """Forward kinematics: tip position (x, y, z) in metres, ikpy base frame."""
    frame = build_chain().forward_kinematics(_full_q(angles))
    return frame[:3, 3]


def fk_pitch(angles: dict[str, float] | list[float]) -> float:
    """In-plane elevation (rad) of the gripper APPROACH axis.

    The approach axis is the wrist_flex->tip (fingertip) link — the direction the
    gripper descends for a grasp. Measured as its elevation projected onto the
    vertical plane that contains the reach (pan) direction, so it matches the
    ``pitch`` argument of :func:`solve_reach`. ``-pi/2`` = straight down.
    """
    frames = build_chain().forward_kinematics(_full_q(angles), full_kinematics=True)
    wflex = frames[4][:3, 3]
    tip = frames[6][:3, 3]
    v = tip - wflex
    px, py = geometry().pan_axis_xy
    az = math.atan2(tip[1] - py, tip[0] - px)  # reach-plane azimuth
    horiz = v[0] * math.cos(az) + v[1] * math.sin(az)
    return math.atan2(v[2], horiz)


# --------------------------------------------------------------------------- #
# Planar geometry — extracted ONCE from the ikpy FK chain (no hardcoded numbers)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlanarGeometry:
    """The pan-frame planar reduction of the SO-101, read off the ikpy chain."""

    pan_axis_xy: tuple[float, float]  # base-frame (x, y) of the shoulder-pan axis
    pan_gain: float                   # d(tip azimuth)/d(pan joint), rad/rad (≈ -1)
    x0: float                         # pan-frame x of the shoulder_lift axis
    z0: float                         # base z of the shoulder_lift axis
    L1: float                         # lift axis -> elbow axis, in-plane length
    L2: float                         # elbow axis -> wrist_flex axis, in-plane
    L3: float                         # wrist_flex axis -> tip, in-plane length
    a1: float                         # rest plane-angle of link1 (lift->elbow)
    a2: float                         # rest plane-angle of link2 (elbow->wrist_flex)
    a3: float                         # rest plane-angle of link3 (wrist_flex->tip)

    @property
    def reach_max(self) -> float:
        return self.L1 + self.L2

    @property
    def reach_min(self) -> float:
        return abs(self.L1 - self.L2)


@lru_cache(maxsize=1)
def geometry() -> PlanarGeometry:
    """Extract the planar geometry from the ikpy chain via FK probes (cached)."""
    chain = build_chain()
    frames = chain.forward_kinematics([0.0] * 7, full_kinematics=True)
    # frames: [base, pan, lift, elbow, wrist_flex, wrist_roll, tip]
    pan = frames[1][:3, 3]
    lift = frames[2][:3, 3]
    elbow = frames[3][:3, 3]
    wflex = frames[4][:3, 3]
    tip = frames[6][:3, 3]

    pan_axis_xy = (float(pan[0]), float(pan[1]))
    # pan-frame x of an axis = its horizontal distance from the pan axis (tip lies
    # at y≈0 in the pan frame, so x_base differences are the planar x coordinate).
    x0 = float(lift[0] - pan[0])

    def plane_angle(p_from, p_to) -> float:
        return math.atan2(p_to[2] - p_from[2], p_to[0] - p_from[0])

    def plane_len(p_from, p_to) -> float:
        return math.hypot(p_to[0] - p_from[0], p_to[2] - p_from[2])

    # pan gain: how tip azimuth (about the pan axis) responds to the pan joint.
    eps = 0.1
    tip_p = chain.forward_kinematics([0.0, eps, 0, 0, 0, 0, 0.0])[:3, 3]
    az = math.atan2(tip_p[1] - pan[1], tip_p[0] - pan[0])
    pan_gain = az / eps

    return PlanarGeometry(
        pan_axis_xy=pan_axis_xy,
        pan_gain=pan_gain,
        x0=x0,
        z0=float(lift[2]),
        L1=plane_len(lift, elbow),
        L2=plane_len(elbow, wflex),
        L3=plane_len(wflex, tip),
        a1=plane_angle(lift, elbow),
        a2=plane_angle(elbow, wflex),
        a3=plane_angle(wflex, tip),
    )


# NOTE: reaching IK lives in ``_prep_planar_ik.PlanarSO101IK`` (the
# hardware-validated closed-form solver), wired up in ``kinematics.solve_ik``.
# This module now only provides FK + extracted geometry (the trusted FK
# reference the rest of the bridge is built on).
