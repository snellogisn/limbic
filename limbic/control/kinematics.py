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
from dataclasses import dataclass

from . import calibration
from .ik_chain import ACTIVE_JOINTS
from .ik_chain import fk as _ikpy_fk
from .safety import ARM_JOINTS, within_limits
from ._prep_planar_ik import PlanarSO101IK

# Hardware-validated closed-form planar IK engine (loads the URDF chain once).
# It takes a table-frame command and returns arm-convention degrees directly --
# pan/azimuth and the table<->arm frame handling all live inside it.
_PREP_IK = PlanarSO101IK()


@dataclass(frozen=True)
class IKSolution:
    """A full joint solution (arm-convention degrees) plus whether it was exactly reachable."""

    joints: dict[str, float]  # {joint_name: degrees} for the 5 arm joints
    reachable: bool           # False => outside reach or no in-limits pose existed


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

    # Empirical correction: turn the desired REAL position into the command to send.
    cmd_reach, cmd_z = calibration.command_for_real(reach_mm, z_mm, approach_pitch_deg)

    def _solve_at(radius_mm: float) -> dict[str, float] | None:
        b = _PREP_IK.solve(
            radius_mm * math.cos(azimuth), radius_mm * math.sin(azimuth), cmd_z, approach_pitch_deg
        )
        return None if b is None else {j: float(b[i]) for i, j in enumerate(ACTIVE_JOINTS)}

    joints = _solve_at(cmd_reach)
    reachable = joints is not None
    if joints is None:  # out of reach -> clamp inward to the nearest feasible radius
        for scale in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5):
            joints = _solve_at(cmd_reach * scale)
            if joints is not None:
                break
    if joints is None:  # still unreachable -> safe neutral pose, flagged unreachable
        return IKSolution(joints={j: 0.0 for j in ACTIVE_JOINTS}, reachable=False)

    in_limits_ok = all(within_limits(j, joints[j]) for j in ARM_JOINTS)
    return IKSolution(joints=joints, reachable=reachable and in_limits_ok)


def forward_kinematics(joints: dict[str, float]) -> tuple[float, float, float]:
    """Tool-tip position (table-frame mm) for a set of arm-convention joint degrees.

    Used by the mock backend to report where the simulated tip "is", and to
    report the achieved position after a real move. Pure kinematics only --
    it does NOT undo the §5 accuracy correction, since that correction
    compensates for real-world droop that this model doesn't otherwise know
    about; it only ever biases the command, never the read-back interpretation.
    """
    ikpy_rad = calibration.arm_to_ikpy_rad({j: joints[j] for j in ACTIVE_JOINTS})
    x_m, y_m, z_m = _ikpy_fk(ikpy_rad)
    return calibration.ikpy_to_table_mm(x_m, y_m, z_m)
