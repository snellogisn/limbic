"""Guardrails: the single source of truth for what the arm is allowed to do.

Every motion — whether issued by a human, a scripted demo, or the LLM brain —
passes through these checks before any command reaches the motors. Keeping the
limits in one module (rather than scattered through the primitives) means there
is exactly one place to audit, tighten, or explain the robot's safe envelope.

Two kinds of guardrail live here:
    1. Per-joint soft limits  (JOINT_SOFT_LIMITS) — degrees each joint may reach.
    2. Cartesian workspace     (WORKSPACE)        — the box/dome the tip may enter.

All angles are in DEGREES in the robot's own convention (centre = 0). Distances
are in MILLIMETRES in the table frame (see ``kinematics`` for the frame
definition).
"""

from __future__ import annotations

from dataclasses import dataclass

# Joint order, base -> tip. The gripper is driven separately (open/close), so it
# is not part of the reaching chain but still has a soft limit.
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper"
ALL_JOINTS: tuple[str, ...] = ARM_JOINTS + (GRIPPER_JOINT,)


# --------------------------------------------------------------------------- #
# Per-joint soft limits (degrees). Deliberately tighter than the motors'
# mechanical range to protect the hardware and anything mounted near it.
# `shoulder_pan` is clamped to +-80 deg to keep the arm from swinging into a
# side-mounted camera, exactly as the reference arm did.
# --------------------------------------------------------------------------- #
JOINT_SOFT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-80.0, 80.0),    # protect a side-mounted camera/sensor
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-95.0, 95.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-165.0, 165.0),
    # Gripper uses a 0..100 "percent open" scale rather than degrees.
    "gripper": (0.0, 100.0),
}


def clamp_joint(name: str, value: float) -> float:
    """Clamp a single joint command to its soft limit. Unknown joints pass through."""
    if name not in JOINT_SOFT_LIMITS:
        return value
    low, high = JOINT_SOFT_LIMITS[name]
    return max(low, min(high, value))


def within_limits(name: str, value: float) -> bool:
    """True if ``value`` is inside joint ``name``'s soft limit (always True if unknown)."""
    if name not in JOINT_SOFT_LIMITS:
        return True
    low, high = JOINT_SOFT_LIMITS[name]
    return low <= value <= high


# --------------------------------------------------------------------------- #
# Cartesian workspace envelope (table frame, millimetres).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Workspace:
    """The reachable region the tool tip is allowed to enter.

    Modelled as a dome: a horizontal reach radius (distance from the pan axis)
    plus a height band. Targets outside the dome are clamped to the nearest
    point inside it — a bad request just stops short instead of crashing.
    """

    reach_min_mm: float = 40.0    # too close folds the arm into itself
    reach_max_mm: float = 310.0   # true physical reach of a small tabletop arm
    z_min_mm: float = -50.0       # below 0 lets a drooping arm still touch the table
    z_max_mm: float = 250.0       # ceiling


WORKSPACE = Workspace()


def clamp_to_workspace(
    x_mm: float, y_mm: float, z_mm: float, workspace: Workspace = WORKSPACE
) -> tuple[float, float, float, bool]:
    """Clamp a Cartesian target to the workspace dome.

    Returns ``(x, y, z, was_clamped)``. The azimuth (direction the arm points) is
    preserved; only the reach radius and height are pulled in if needed.
    """
    import math

    radius = math.hypot(x_mm, y_mm)
    clamped = False

    if radius > workspace.reach_max_mm:
        scale = workspace.reach_max_mm / radius
        x_mm, y_mm = x_mm * scale, y_mm * scale
        clamped = True
    elif 1e-6 < radius < workspace.reach_min_mm:
        scale = workspace.reach_min_mm / radius
        x_mm, y_mm = x_mm * scale, y_mm * scale
        clamped = True

    z_clamped = min(max(z_mm, workspace.z_min_mm), workspace.z_max_mm)
    if z_clamped != z_mm:
        z_mm, clamped = z_clamped, True

    return x_mm, y_mm, z_mm, clamped
