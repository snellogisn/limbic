"""SO-101 kinematics built on the ikpy chain (Part A, §A.2).

Two cooperating pieces, both derived from the SAME source of truth — the ikpy
chain parsed from the SO-101 URDF — so they can never drift apart:

  * ``build_chain()`` / ``fk()`` — forward kinematics + geometry via ikpy. ikpy
    is pure Python and runs on any arch. This is the trusted FK reference.

  * ``solve_reach()`` — a DETERMINISTIC closed-form planar reaching solver. We do
    NOT use ikpy's numerical ``inverse_kinematics`` for reaching: it branch-jumps
    (non-deterministically hops elbow-up/elbow-down between calls). Instead we
    reduce the arm to ``shoulder_pan`` (azimuth) + a planar RRR in the pan-frame
    vertical plane and solve it analytically, picking a single fixed elbow branch.

All geometry constants (link lengths, rest angles, pan sign, the pitch mapping)
are EXTRACTED ONCE from the ikpy FK chain at import — nothing is hardcoded — so
this stays consistent with FK by construction. We validate the two against each
other in scripts/stage1_ik_check.py.

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


# --------------------------------------------------------------------------- #
# Joint limits (from the URDF, via the ikpy chain)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def joint_limits() -> dict[str, tuple[float, float]]:
    """{joint: (lo, hi)} in radians, read from the URDF joint limits."""
    chain = build_chain()
    out: dict[str, tuple[float, float]] = {}
    for link, active in zip(chain.links, _ACTIVE_MASK):
        if active:
            lo, hi = link.bounds
            out[link.name] = (
                -math.pi if lo is None or not math.isfinite(lo) else lo,
                math.pi if hi is None or not math.isfinite(hi) else hi,
            )
    return out


def _wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def in_limits(joints: dict[str, float], tol: float = 1e-6) -> bool:
    lim = joint_limits()
    return all(lim[j][0] - tol <= joints[j] <= lim[j][1] + tol for j in lim)


# --------------------------------------------------------------------------- #
# Closed-form reaching solver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReachSolution:
    joints: dict[str, float]  # {joint: radians} for the 5 active joints
    reachable: bool           # False => target clamped to nearest feasible point
    in_limits: bool           # False => no joint-limit-respecting pose exists


def solve_reach(
    x: float,
    y: float,
    z: float,
    pitch: float = -math.pi / 2,
    wrist_roll: float = 0.0,
    current: dict[str, float] | None = None,
) -> ReachSolution:
    """Closed-form IK: tip at (x, y, z) m (ikpy base frame) with approach ``pitch``.

    ``pitch`` is the in-plane elevation of the gripper APPROACH axis — the
    direction of the wrist_flex->tip (fingertip) link: ``-pi/2`` = straight down
    for a top-down grasp, ``0`` = horizontal forward.

    Deterministic. Of the two elbow branches it keeps those within the URDF joint
    limits and, per the build guide, prefers elbow-up; if ``current`` is given it
    instead prefers the least-motion branch (continuity). Out-of-reach targets
    clamp to the nearest feasible planar point (``reachable=False``).
    """
    g = geometry()
    px, py = g.pan_axis_xy

    # --- shoulder_pan: aim the planar slice at the target's azimuth -----------
    azimuth = math.atan2(y - py, x - px)
    pan = _wrap(azimuth / g.pan_gain)
    xt = math.hypot(x - px, y - py)  # horizontal distance from the pan axis

    # --- pitch is the wrist_flex->tip link direction; back off to the wrist ----
    xw = xt - g.L3 * math.cos(pitch)
    zw = z - g.L3 * math.sin(pitch)

    # --- planar 2-link (L1, L2) from the lift axis to the wrist center ---------
    dx = xw - g.x0
    dz = zw - g.z0
    d = math.hypot(dx, dz)
    reachable = g.reach_min <= d <= g.reach_max
    if not reachable and d > 0:
        d = min(max(d, g.reach_min + 1e-9), g.reach_max - 1e-9)
        xw = g.x0 + dx * d / math.hypot(dx, dz)
        zw = g.z0 + dz * d / math.hypot(dx, dz)
        dx, dz = xw - g.x0, zw - g.z0

    base_angle = math.atan2(dz, dx)
    cos_in = (g.L1 * g.L1 + d * d - g.L2 * g.L2) / (2 * g.L1 * d)
    cos_in = max(-1.0, min(1.0, cos_in))
    interior = math.acos(cos_in)

    # Both elbow branches; recover ikpy joint angles for each.
    candidates: list[tuple[dict[str, float], float]] = []  # (joints, elbow_z)
    for sign in (+1.0, -1.0):
        phi1 = base_angle + sign * interior
        ex = g.x0 + g.L1 * math.cos(phi1)
        ez = g.z0 + g.L1 * math.sin(phi1)
        phi2 = math.atan2(zw - ez, xw - ex)
        s = _wrap(g.a1 - phi1)
        e = _wrap(g.a2 - (g.a1 - phi1) - phi2)
        w = _wrap((g.a3 - pitch) - (g.a1 - phi1) - (g.a2 - (g.a1 - phi1) - phi2))
        joints = {
            "shoulder_pan": pan,
            "shoulder_lift": s,
            "elbow_flex": e,
            "wrist_flex": w,
            "wrist_roll": wrist_roll,
        }
        candidates.append((joints, ez))

    valid = [(j, ez) for j, ez in candidates if in_limits(j)]
    pool = valid if valid else candidates
    if current is not None:
        keys = ("shoulder_lift", "elbow_flex", "wrist_flex")
        chosen = min(pool, key=lambda c: max(abs(c[0][k] - current[k]) for k in keys))[0]
    else:
        # prefer elbow-up: the branch whose elbow sits higher (larger elbow z).
        chosen = max(pool, key=lambda c: c[1])[0]

    return ReachSolution(joints=chosen, reachable=reachable, in_limits=bool(valid))
