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
# "Home" is a COARSE move to a known-safe neutral pose, so it favours SPEED over
# precision — "close enough", no fine convergence pass. Bigger sub-steps (fewer
# setpoints => quicker) and a brief settle. Both env-tunable; raise HOME_STEP for
# a faster home, lower it if the move looks jerky on the rig.
HOME_STEP_DEG = float(os.environ.get("LIMBIC_HOME_STEP", "3.0"))     # deg per sub-step (~3x transit)
HOME_SETTLE_S = float(os.environ.get("LIMBIC_HOME_SETTLE", "0.15"))  # brief settle after homing
GRIPPER_SETTLE_S = float(os.environ.get("LIMBIC_GRIPPER_SETTLE", "0.5"))  # final claw settle once it's stopped
# The claw is one slow servo with no "done" signal, and it's slower than any fixed
# guess. So instead of sleeping a constant we DRIVE it to its target and watch the
# read-back until it physically STOPS moving — reached full open/close, or
# stalled/clamped on an object. Only then is the claw genuinely done, so a
# following arm move can't begin mid-grip ("grab and lift at the same time").
GRIPPER_POLL_S = float(os.environ.get("LIMBIC_GRIPPER_POLL", "0.05"))       # re-command/re-read cadence
GRIPPER_STOP_TOL = float(os.environ.get("LIMBIC_GRIPPER_STOP_TOL", "1.0"))  # "stopped" when read-back moves < this (% open)
GRIPPER_MAX_S = float(os.environ.get("LIMBIC_GRIPPER_MAX", "2.5"))          # hard cap so we never wait forever
# Every motor lags its setpoint — open-loop servos take a moment to physically
# arrive after the command stream stops. So pause briefly between KEY POINTS (the
# end of each point-to-point move) before the next action begins. This is what
# keeps motions DISCRETE: the arm is truly settled before the claw actuates, so
# the claw acts in real isolation (§0.6) instead of closing mid-drift.
MOVE_SETTLE_S = float(os.environ.get("LIMBIC_MOVE_SETTLE", "0.5"))  # settle after each point-to-point move


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
