"""Connection and motion configuration for the arm — all overridable by env var.

Nothing here is OS-specific: the actual port-finding lives in
``limbic.platform_support`` so this module reads the same way on every machine.
Every value can be overridden from the environment so a user never edits code to
retarget hardware:

    LIMBIC_PORT     serial port (e.g. COM7, /dev/cu.usbserial-10). If unset we
                    auto-detect; if detection fails we fall back to the mock.
    LIMBIC_ROBOT_ID robot id/name used by the underlying SDK (default "bronny",
                    which is the calibration this physical arm is registered under
                    — see ~/.cache/huggingface/lerobot/calibration/robots/).
    LIMBIC_BACKEND  "auto" | "real" | "mock" — which hardware backend to use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..platform_support import detect_serial_port


# Gripper "percent open" scale (0..100, not raw servo ticks, so it reads the same
# regardless of the underlying motor): 100 = fully open, 0 = fully closed.
# CLOSED is 0 so a grip closes ALL THE WAY: with nothing in the claw the fingers
# shut completely; with an object the servo (position mode) simply stops on it and
# holds — i.e. "close all the way unless something is blocking it".
GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 0.0

# Smooth-motion profile. Moves are interpolated into fine sub-steps with an
# ease-in/ease-out velocity curve, so the arm accelerates and decelerates instead
# of jerking step-to-step.
#
# Two independent knobs, BOTH env-overridable:
#   * STEP (deg per sub-step) — how far the arm is told to move each command.
#   * DT   (seconds between sub-steps) — how often commands are streamed.
# Speed ≈ STEP / DT. SMOOTHNESS comes from a SMALL step at a HIGH stream rate
# (small dt): the servo always chases a target just ahead of it and never snaps-
# then-waits. To go slow AND smooth, shrink STEP and keep DT small — do NOT just
# inflate DT (big step + long pause = visible jerk). A simulation can set tiny dt
# (e.g. LIMBIC_SMOOTH_DT=0.001) to run near-instantly.
SMOOTH_STEP_DEG = float(os.environ.get("LIMBIC_SMOOTH_STEP", "1.5"))  # transit: deg per sub-step
SMOOTH_DT_S = float(os.environ.get("LIMBIC_SMOOTH_DT", "0.02"))       # seconds between sub-steps (~50 Hz)
SLOW_STEP_DEG = float(os.environ.get("LIMBIC_SLOW_STEP", "1.0"))      # precision (descend/grasp/place): finer ...
SLOW_DT_S = float(os.environ.get("LIMBIC_SLOW_DT", "0.06"))           # ... and slower for controlled fine motion
CONVERGE_TOL_DEG = 1.2  # hold the goal until servos settle within this band
GRIPPER_SETTLE_S = float(os.environ.get("LIMBIC_GRIPPER_SETTLE", "0.5"))  # let the claw fully actuate


@dataclass(frozen=True)
class ArmConfig:
    """Everything needed to connect to (or simulate) one arm."""

    port: str | None        # None => no port found; auto backend uses the mock
    robot_id: str
    backend: str            # "auto" | "real" | "mock"
    hold_torque: bool = True  # keep torque on between commands so the arm holds pose


def load_config() -> ArmConfig:
    """Build an :class:`ArmConfig` from environment variables + auto-detection."""
    return ArmConfig(
        port=detect_serial_port(),                        # honours $LIMBIC_PORT
        robot_id=os.environ.get("LIMBIC_ROBOT_ID", "bronny"),
        backend=os.environ.get("LIMBIC_BACKEND", "auto").lower(),
    )
