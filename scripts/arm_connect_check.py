"""Stage 0 hardware check — READ-ONLY, commands no motion.

Confirms the per-machine hardware seam before any kinematics work:
  1. prints the machine profile (serial ports + cameras by name),
  2. connects to the real SO-101 on its serial port,
  3. reads and prints all six joints in DEGREES (gripper on its 0..100 scale),
  4. disconnects.

It NEVER moves the arm and NEVER (re)calibrates: it connects with
``calibrate=False`` and refuses to connect unless a calibration file already
exists for the chosen robot id. A MISSING calibration file is what triggers
lerobot's interactive range-of-motion routine (it drives every joint to its
limits) — so we abort with a clear message instead of surprising the arm.

The follower on this rig is calibrated under the id "bronny" (see
~/.cache/huggingface/lerobot/calibration/robots/so_follower/bronny.json), so we
default to that. Override with $LIMBIC_ROBOT_ID.

Safety: the arm must be on BARREL-JACK power (never USB). Connecting enables
torque, so the arm holds its current pose stiffly — it will not move to a
commanded position. Leave the arm in a stable resting pose before running.

    python scripts/arm_connect_check.py
"""

from __future__ import annotations

import os
import pathlib
import sys

# Make `limbic` importable when run directly from scripts/ even if not installed.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from limbic.platform_support import detect_serial_port, format_profile

EXPECTED_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def calibration_path(robot_id: str) -> pathlib.Path:
    """Where lerobot stores this follower's calibration file."""
    root = os.environ.get("HF_LEROBOT_HOME") or (
        pathlib.Path.home() / ".cache" / "huggingface" / "lerobot"
    )
    return pathlib.Path(root) / "calibration" / "robots" / "so_follower" / f"{robot_id}.json"


def main() -> None:
    print(format_profile())
    print()

    robot_id = os.environ.get("LIMBIC_ROBOT_ID", "bronny")
    port = detect_serial_port() or os.environ.get("LIMBIC_PORT")
    if port is None:
        raise SystemExit("No serial port found. Plug in the arm or set $LIMBIC_PORT.")

    cal = calibration_path(robot_id)
    if not cal.exists():
        raise SystemExit(
            f"No calibration file for robot id {robot_id!r} at {cal}.\n"
            "Connecting would launch lerobot's interactive calibration (it moves "
            "every joint to its limits). Aborting. Set LIMBIC_ROBOT_ID to a "
            "calibrated id, or calibrate deliberately."
        )

    print(f"Connecting to SO-101 on {port} as id {robot_id!r} (read-only, no motion)...")
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cfg = SO101FollowerConfig(
        port=port,
        id=robot_id,
        max_relative_target=None,
        disable_torque_on_disconnect=False,  # hold pose; never go limp mid-test
    )
    robot = SO101Follower(cfg)
    robot.connect(calibrate=False)  # never (re)calibrate
    try:
        obs = robot.get_observation()
        joints = {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}
        print("\nJoint readings (degrees; gripper = 0..100 open%):")
        for name in EXPECTED_JOINTS:
            val = joints.get(name)
            unit = "%" if name == "gripper" else "deg"
            print(f"   {name:14s} {val:8.2f} {unit}" if val is not None
                  else f"   {name:14s}   MISSING")
        missing = [j for j in EXPECTED_JOINTS if j not in joints]
        extra = [j for j in joints if j not in EXPECTED_JOINTS]
        print()
        if missing:
            print(f"WARNING: missing joints: {missing}")
        if extra:
            print(f"NOTE: extra channels reported: {extra}")
        if not missing:
            print("OK: all six joints report.")
    finally:
        robot.disconnect()
        print("Disconnected (torque left engaged; arm holds its pose).")


if __name__ == "__main__":
    main()
