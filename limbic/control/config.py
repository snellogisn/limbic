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


# Gripper "percent open" scale, verified on the reference hardware:
# 100 = fully open, 10 = closed enough to grip. Kept on a 0..100 scale (not raw
# servo ticks) so it reads the same regardless of the underlying motor.
GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 10.0

# Smooth-motion profile. Moves are interpolated into fine sub-steps with an
# ease-in/ease-out velocity curve streamed at ~50 Hz, so the arm accelerates and
# decelerates instead of jerking step-to-step.
SMOOTH_STEP_DEG = 1.5   # transit speed: degrees of travel per sub-step
SMOOTH_DT_S = 0.02      # seconds between sub-steps (~50 Hz)
SLOW_STEP_DEG = 1.0     # precision speed for descend/grasp/place: finer ...
SLOW_DT_S = 0.06        # ... and ~3x slower for controlled fine motion
CONVERGE_TOL_DEG = 1.2  # hold the goal until servos settle within this band
GRIPPER_SETTLE_S = 0.5  # let the claw fully actuate before moving on


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
