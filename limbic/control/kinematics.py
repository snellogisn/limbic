"""Inverse / forward kinematics: table-frame mm <-> this arm's joint degrees.

Built on two trusted layers that stay consistent by construction:
    * geometry + FK = the ikpy chain (``ik_chain``), parsed from the SO-101 URDF.
    * reaching IK   = ik_chain's deterministic closed-form planar solver.

This module is the thin, MEASURED bridge on top of both (§A.3/§A.6 of
CLAUDE.md, constants in ``calibration``): the table-frame <-> ikpy-frame
offset, the ikpy<->arm joint sign/offset, and an empirical top-down accuracy
correction. Nothing hardware-specific is invented here -- it all comes from
``calibration``.

THE TABLE FRAME (memorize this -- every Cartesian call uses it):
    origin : directly under the shoulder-pan axis, on the table surface
    +x     : forward  (the direction the arm reaches when pan = 0)
    +y     : left
    +z     : up from the table
Units: millimetres for position, degrees for joint angles (arm convention).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from . import calibration
from .ik_chain import ACTIVE_JOINTS
from .ik_chain import fk as _ikpy_fk
from .safety import ARM_JOINTS, within_limits

# ------------------------------------------------------------------------- #
# Reaching-IK engine. Default = mink (MuJoCo): an open-source differential-IK
# solver verified on this rig (FK->IK 0.0008 mm, MuJoCo-vs-ikpy frame 0.0000 mm).
# The closed-form planar solver is kept only as an explicit fallback
# (LIMBIC_IK=planar). Both expose ``.solve(x_mm, y_mm, z_mm, pitch_deg) -> deg[5]``
# in ACTIVE_JOINTS order, so the rest of this module is engine-agnostic.
#
# The engine is built lazily on first use so importing ``limbic.control`` never
# requires mujoco/mink (preserving the arch-split: arm code imports anywhere).
# ------------------------------------------------------------------------- #
_ENGINE = None
_USING_MINK = False

# Live-tuned droop biases for the mink path (this rig, wrist_roll locked). The arm
# sags under gravity: the tip lands SHORT in reach and LOW in z, so we command a
# bit farther + higher. Seeded from ruler measurements; refine at the rig via
# $LIMBIC_FWD_BIAS_MM / $LIMBIC_Z_BIAS_MM. (The reach-dependent z fit in
# ``calibration.command_for_real`` does the bulk of z; Z_BIAS is the final trim.)
_MINK_FWD_BIAS_MM = float(os.environ.get("LIMBIC_FWD_BIAS_MM", "50.0"))
_MINK_Z_BIAS_MM = float(os.environ.get("LIMBIC_Z_BIAS_MM", "0.0"))

# Wrist-TILT position fix: mink solves position-only, so the gripper tilts as it
# reaches out, and the claw then overhangs the IK tip horizontally (calibration.
# claw_overhang_offset). We read the achieved tilt back and re-solve a couple of
# times so the CLAW — not the bare tip — lands on the target. A few iterations
# converge (each reach nudge only slightly changes the tilt); below the tolerance
# we stop. No-op when the claw overhang is 0 (unmeasured) or the wrist is vertical.
_CLAW_COMP_ITERS = 3
_CLAW_COMP_TOL_MM = 0.5


def _engine():
    """Return the reaching-IK engine, building it once on first call."""
    global _ENGINE, _USING_MINK
    if _ENGINE is not None:
        return _ENGINE
    pref = os.environ.get("LIMBIC_IK", "mink").lower()
    if pref != "planar":
        try:
            from .mink_ik import MinkSO101IK
            _ENGINE = MinkSO101IK()
            _USING_MINK = True
            return _ENGINE
        except Exception as exc:  # mink/mujoco absent or model failed -> fall back
            import warnings
            warnings.warn(
                f"mink IK unavailable ({exc!r}); falling back to the closed-form "
                "planar solver. Set LIMBIC_IK=planar to silence this.",
                RuntimeWarning, stacklevel=2,
            )
    from ._prep_planar_ik import PlanarSO101IK
    _ENGINE = PlanarSO101IK()
    _USING_MINK = False
    return _ENGINE


def active_engine() -> str:
    """Name of the reaching-IK engine actually in use: ``"mink"`` or ``"planar"``.

    Builds the engine on first call (so it reflects the REAL choice, including a
    silent fallback to planar when mink/mujoco can't import). Useful for a startup
    banner that verifies the demo box is on the proper solver.
    """
    _engine()
    return "mink" if _USING_MINK else "planar"


@dataclass(frozen=True)
class IKSolution:
    """A full joint solution (arm-convention degrees) plus whether it was exactly reachable."""

    joints: dict[str, float]  # {joint_name: degrees} for the 5 arm joints
    reachable: bool           # False => outside reach or no in-limits pose existed
    # The tool pitch the solution ACTUALLY achieves (deg; -90 = straight down).
    # mink tilts the wrist as it reaches out, so this is the realised angle, not the
    # requested one — used to compensate the claw overhang (the tilt-grasp fix) and
    # available to callers that want to know the true approach. Equals the requested
    # pitch on the closed-form planar solver (which controls pitch directly).
    achieved_pitch_deg: float = -90.0


def solve_ik(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    approach_pitch_deg: float = -90.0,
) -> IKSolution:
    """Inverse kinematics for the tool tip at table-frame ``(x, y, z)`` mm.

    Args:
        x_mm, y_mm, z_mm: Desired tool-tip position in the table frame.
        approach_pitch_deg: Tool approach angle. ``-90`` = pointing straight
            down (top-down grasp, the default); ``0`` = horizontal forward.

    Returns:
        An :class:`IKSolution` in the arm's own degree convention.
        ``reachable`` is ``False`` when the target was outside the arm's
        reach or no joint-limit-respecting pose existed; the closed-form
        solver still returns its nearest/best-effort pose in that case.
    """
    azimuth = math.atan2(y_mm, x_mm)
    reach_mm = math.hypot(x_mm, y_mm)

    eng = _engine()

    def _solve_at(radius_mm: float, cmd_z: float) -> dict[str, float] | None:
        b = eng.solve(
            radius_mm * math.cos(azimuth), radius_mm * math.sin(azimuth), cmd_z, approach_pitch_deg
        )
        return None if b is None else {j: float(b[i]) for i, j in enumerate(ACTIVE_JOINTS)}

    def _command_for(real_reach: float, real_z: float) -> tuple[float, float]:
        """(cmd_reach, cmd_z) to feed the solver so the IK TIP lands at real (reach, z)."""
        if _USING_MINK:
            # mink kinematics, with the rig's MEASURED z-droop compensation only:
            # forward is left raw (it lands within ~1 cm on this arm), but the tip
            # sags under gravity in z, so we aim higher in z via the measured fit.
            # Forward's separate fit is NOT used -- it was tuned to the old solver.
            _, cmd_z = calibration.command_for_real(real_reach, real_z + _MINK_Z_BIAS_MM, approach_pitch_deg)
            return real_reach + _MINK_FWD_BIAS_MM, cmd_z
        # Closed-form path: turn the desired REAL position into the command to send.
        return calibration.command_for_real(real_reach, real_z, approach_pitch_deg)

    # First pass: aim the IK TIP at the requested point.
    real_reach, real_z = reach_mm, z_mm
    cmd_reach, cmd_z = _command_for(real_reach, real_z)
    joints = _solve_at(cmd_reach, cmd_z)
    achieved_pitch = approach_pitch_deg

    # WRIST-TILT POSITION FIX (mink only): mink lets the wrist tilt as it reaches,
    # and the claw then overhangs the IK tip horizontally — so a tip aimed at the
    # block lands the CLAW beyond it. Read the achieved tilt back and pull the TARGET
    # in by the claw overhang, re-solving until it settles, so the CLAW hits the
    # block. All terms are 0 at top-down, so a vertical grasp is unchanged. The
    # planar solver controls pitch directly, so it needs no read-back.
    if joints is not None and _USING_MINK:
        achieved_pitch = eng.tool_pitch_deg(joints)
        for _ in range(_CLAW_COMP_ITERS):
            d_reach, d_z = calibration.claw_overhang_offset(achieved_pitch)
            if abs(d_reach) <= _CLAW_COMP_TOL_MM and abs(d_z) <= _CLAW_COMP_TOL_MM:
                break
            real_reach, real_z = reach_mm - d_reach, z_mm - d_z
            cmd_reach, cmd_z = _command_for(real_reach, real_z)
            adjusted = _solve_at(cmd_reach, cmd_z)
            if adjusted is None:  # the inward nudge went unreachable -> keep last good
                break
            joints = adjusted
            achieved_pitch = eng.tool_pitch_deg(joints)

    reachable = joints is not None
    if joints is None:  # out of reach -> clamp inward to the nearest feasible radius
        for scale in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5):
            joints = _solve_at(cmd_reach * scale, cmd_z)
            if joints is not None:
                if _USING_MINK:
                    achieved_pitch = eng.tool_pitch_deg(joints)
                break
    if joints is None:  # still unreachable -> safe neutral pose, flagged unreachable
        return IKSolution(joints={j: 0.0 for j in ACTIVE_JOINTS}, reachable=False,
                          achieved_pitch_deg=approach_pitch_deg)

    in_limits_ok = all(within_limits(j, joints[j]) for j in ARM_JOINTS)
    return IKSolution(joints=joints, reachable=reachable and in_limits_ok,
                      achieved_pitch_deg=achieved_pitch)


def forward_kinematics(joints: dict[str, float]) -> tuple[float, float, float]:
    """Tool-tip position (table-frame mm) for a set of arm-convention joint degrees.

    Used by the mock backend to report where the simulated tip "is", and to
    report the achieved position after a real move. Pure kinematics only --
    it does NOT undo the §5 accuracy correction, since that correction
    compensates for real-world droop that this model doesn't otherwise know
    about; it only ever biases the command, never the read-back interpretation.
    """
    eng = _engine()
    if _USING_MINK:
        return eng.fk({j: joints[j] for j in ACTIVE_JOINTS})
    ikpy_rad = calibration.arm_to_ikpy_rad({j: joints[j] for j in ACTIVE_JOINTS})
    x_m, y_m, z_m = _ikpy_fk(ikpy_rad)
    return calibration.ikpy_to_table_mm(x_m, y_m, z_m)
