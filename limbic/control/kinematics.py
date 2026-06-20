"""Closed-form inverse kinematics: table coordinates -> joint angles.

Pure Python (``math`` only, no numpy/ikpy), so it imports and runs identically on
macOS, Windows and Linux with zero binary dependencies. That portability is the
whole point: the reference project was blocked on Windows/ARM because its IK
backends (placo, mujoco) had no wheels for that platform. A small analytical
solver sidesteps that entirely.

THE TABLE FRAME (memorize this — every Cartesian call uses it):
    origin : directly under the shoulder-pan axis, on the table surface
    +x     : forward  (the direction the arm reaches when pan = 0)
    +y     : left
    +z     : up from the table
Units: millimetres for position, degrees for joint angles.

The arm is treated as:
    * shoulder_pan  -> azimuth (which way the whole arm faces)
    * shoulder_lift + elbow_flex -> a planar 2-link reach in the vertical plane,
      solving for radius ``r`` (horizontal distance from the pan axis) and height
      ``z``
    * wrist_flex    -> sets the approach pitch of the tool (e.g. straight down)
    * wrist_roll    -> tool roll; left at 0 for top-down grasps

LINK_*_MM and the offsets below are NOMINAL values for a small tabletop arm and
are meant to be measured/calibrated against your specific hardware (the reference
project did this with a ruler). They are good enough to drive the mock backend
and to validate the whole pipeline end to end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Nominal arm geometry (millimetres). Calibrate against your hardware.
LINK_UPPER_MM = 116.0   # shoulder_lift pivot -> elbow pivot
LINK_FORE_MM = 135.0    # elbow pivot -> wrist pivot
WRIST_MM = 70.0         # wrist pivot -> tool tip
BASE_HEIGHT_MM = 100.0  # height of the shoulder_lift pivot above the table


@dataclass(frozen=True)
class IKSolution:
    """A full joint solution (degrees) plus whether it was exactly reachable."""

    joints: dict[str, float]   # {joint_name: degrees} for the 5 arm joints
    reachable: bool            # False => clamped to the nearest feasible pose


def _planar_two_link(
    r_mm: float, z_mm: float, elbow_up: bool
) -> tuple[float, float] | None:
    """Solve the vertical-plane 2-link reach for the wrist pivot at (r, z).

    Returns ``(shoulder_lift_deg, elbow_flex_deg)`` or ``None`` if (r, z) is
    outside the annulus the two links can reach. The convention is chosen to be
    exactly consistent with :func:`forward_kinematics`: the upper link's absolute
    angle from horizontal is ``shoulder_lift``, and the fore link's absolute angle
    is ``shoulder_lift + elbow_flex`` (i.e. ``elbow_flex`` is the relative joint
    angle, positive = counter-clockwise). 0 = pointing straight out horizontally.
    """
    # Target of the wrist pivot, relative to the shoulder_lift pivot.
    dz = z_mm - BASE_HEIGHT_MM
    dist = math.hypot(r_mm, dz)

    reach_max = LINK_UPPER_MM + LINK_FORE_MM
    reach_min = abs(LINK_UPPER_MM - LINK_FORE_MM)
    if dist > reach_max or dist < reach_min or dist == 0:
        return None

    # Standard 2-link IK. cos(q2) from the law of cosines; the sign of q2 selects
    # the elbow-up vs elbow-down branch.
    cos_q2 = (r_mm**2 + dz**2 - LINK_UPPER_MM**2 - LINK_FORE_MM**2) / (
        2 * LINK_UPPER_MM * LINK_FORE_MM
    )
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    sin_q2 = math.sqrt(1.0 - cos_q2**2)
    elbow = math.atan2(-sin_q2 if elbow_up else sin_q2, cos_q2)

    # Shoulder so that the two links chain to (r, dz).
    shoulder = math.atan2(dz, r_mm) - math.atan2(
        LINK_FORE_MM * math.sin(elbow), LINK_UPPER_MM + LINK_FORE_MM * math.cos(elbow)
    )

    return math.degrees(shoulder), math.degrees(elbow)


def solve_ik(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    approach_pitch_deg: float = -90.0,
) -> IKSolution:
    """Inverse kinematics for the tool tip at table-frame ``(x, y, z)`` mm.

    Args:
        x_mm, y_mm, z_mm: Desired tool-tip position in the table frame.
        approach_pitch_deg: Tool approach angle. ``-90`` = pointing straight down
            (top-down grasp, the default); ``0`` = pointing horizontally forward.

    Returns:
        An :class:`IKSolution`. ``reachable`` is ``False`` when the target lay
        outside the arm's reach and the solution is the closest feasible pose;
        callers (and the safety layer) decide whether that's acceptable.

    The wrist target is computed by stepping back from the tip along the approach
    direction, so the tool *tip* — not the wrist — lands on (x, y, z).
    """
    pitch = math.radians(approach_pitch_deg)

    # Azimuth: which way the whole arm faces. atan2(y, x) keeps +x forward / +y left.
    pan_deg = math.degrees(math.atan2(y_mm, x_mm))

    # Horizontal reach to the tip, then step back along the approach direction to
    # find where the wrist pivot must sit.
    r_tip = math.hypot(x_mm, y_mm)
    r_wrist = r_tip - WRIST_MM * math.cos(pitch)
    z_wrist = z_mm - WRIST_MM * math.sin(pitch)

    planar = _planar_two_link(r_wrist, z_wrist, elbow_up=True)
    reachable = planar is not None
    if planar is None:
        # Out of reach: aim at the closest point on the reach circle so the arm
        # extends toward the target instead of failing outright.
        planar = _closest_feasible(r_wrist, z_wrist)

    shoulder_deg, elbow_deg = planar

    # Wrist flex makes the tool hold the requested approach pitch regardless of
    # how the two links ended up oriented.
    wrist_flex_deg = approach_pitch_deg - (shoulder_deg + elbow_deg)

    return IKSolution(
        joints={
            "shoulder_pan": pan_deg,
            "shoulder_lift": shoulder_deg,
            "elbow_flex": elbow_deg,
            "wrist_flex": wrist_flex_deg,
            "wrist_roll": 0.0,
        },
        reachable=reachable,
    )


def _closest_feasible(r_mm: float, z_mm: float) -> tuple[float, float]:
    """Aim the arm at the nearest point on its outer reach circle for (r, z)."""
    dz = z_mm - BASE_HEIGHT_MM
    angle = math.atan2(dz, r_mm)
    reach = LINK_UPPER_MM + LINK_FORE_MM
    r_clamped = reach * math.cos(angle)
    z_clamped = BASE_HEIGHT_MM + reach * math.sin(angle)
    fallback = _planar_two_link(r_clamped, z_clamped, elbow_up=True)
    # Almost-fully-extended; if rounding still fails, just lay the arm flat-ish.
    return fallback if fallback is not None else (math.degrees(angle), 0.0)


def forward_kinematics(joints: dict[str, float]) -> tuple[float, float, float]:
    """Tool-tip position (table-frame mm) for a set of joint angles.

    Used by the mock backend to report where the simulated tip "is", and as a
    round-trip check on the IK. The inverse of :func:`solve_ik`.
    """
    pan = math.radians(joints["shoulder_pan"])
    shoulder = math.radians(joints["shoulder_lift"])
    elbow = math.radians(joints["elbow_flex"])
    wrist = math.radians(joints["wrist_flex"])

    # Build the reach radius / height by walking out each link in the vertical plane.
    r = LINK_UPPER_MM * math.cos(shoulder)
    z = BASE_HEIGHT_MM + LINK_UPPER_MM * math.sin(shoulder)
    r += LINK_FORE_MM * math.cos(shoulder + elbow)
    z += LINK_FORE_MM * math.sin(shoulder + elbow)
    r += WRIST_MM * math.cos(shoulder + elbow + wrist)
    z += WRIST_MM * math.sin(shoulder + elbow + wrist)

    x = r * math.cos(pan)
    y = r * math.sin(pan)
    return x, y, z
