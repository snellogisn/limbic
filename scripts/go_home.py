"""Move the arm to HOME — every motor centred (all joints 0 deg, gripper halfway).

This is what "go home" means on this rig: each motor at the middle of its range,
a known neutral pose (handy before powering down, tightening hardware, or
re-zeroing). Connects the real SO-101 (bronny, calibrate=False), drives there
smoothly, holds (torque engaged), and disconnects without going limp.

Safety: BARREL-JACK power only. The arm sweeps up/forward to the centred pose —
keep the workspace clear.

    python scripts/go_home.py
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def calibration_path(robot_id: str) -> pathlib.Path:
    root = os.environ.get("HF_LEROBOT_HOME") or (
        pathlib.Path.home() / ".cache" / "huggingface" / "lerobot"
    )
    return pathlib.Path(root) / "calibration" / "robots" / "so_follower" / f"{robot_id}.json"


def main() -> None:
    from limbic.control.arm import RobotArm
    from limbic.control.backends import RealBackend
    from limbic.control.config import load_config

    robot_id = os.environ.get("LIMBIC_ROBOT_ID", "bronny")
    cal = calibration_path(robot_id)
    if not cal.exists():
        raise SystemExit(
            f"No calibration file for id {robot_id!r} at {cal}. Aborting "
            "(connecting without it would launch interactive calibration)."
        )

    cfg = load_config()
    if cfg.port is None:
        raise SystemExit("No serial port found. Plug in the arm or set $LIMBIC_PORT.")

    print(f"Connecting real SO-101 on {cfg.port} as {robot_id!r}; moving to HOME "
          "(all motors centred)...")
    arm = RobotArm(config=cfg, backend=RealBackend(cfg), verbose=True)
    arm.connect()
    try:
        joints = arm.go_home()
        print("\nAt HOME. Joint readings (deg; gripper 0..100):")
        for name, val in joints.items():
            print(f"   {name:14s} {val:8.2f}")
    finally:
        arm.disconnect()
        print("Disconnected (torque left engaged; arm holds its pose).")


if __name__ == "__main__":
    main()
