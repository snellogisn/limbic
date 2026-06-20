"""Stage 2 HARDWARE ruler check (§A.6 / §5) — the first real top-down move.

Confirms on the PHYSICAL arm that the open-loop accuracy correction points the
right way: command a known REAL table target, then ruler-measure where the tip
actually lands. It must land AT the target, not low/short. If it lands LOW, the
§5 z-correction is inverted — STOP and fix before any grasp.

Two modes:
  * DRY RUN (default): no hardware, no motion. Prints the IK plan for the target
    — the corrected COMMAND the solver will send and the model tip that command
    maps to — so you can eyeball it before anything moves.
  * GO (`--go`): connects the real SO-101 (bronny, calibrate=False) and streams a
    SLOW top-down move to the target, then holds (torque engaged) so you can
    measure. Disconnects leaving torque engaged.

Usage:
    python scripts/stage2_ruler_check.py                # dry run, default target
    python scripts/stage2_ruler_check.py 180 0 60       # dry run, custom target
    python scripts/stage2_ruler_check.py 180 0 60 --go  # MOVE the real arm

Target is REAL table-frame mm: x=forward from the pan axis, y=left, z=up from the
table. Default pitch is -90 (straight down). Safety: BARREL-JACK power only, arm
clear of obstacles, ready to catch it. Start HIGH (z=60-80) so a wrong z sign is
caught with margin before the tip reaches the table; only then step down.
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from limbic.control import calibration
from limbic.control.kinematics import forward_kinematics, solve_ik
from limbic.control.safety import ARM_JOINTS, clamp_to_workspace, within_limits

DEFAULT_TARGET = (180.0, 0.0, 60.0, -90.0)  # x, y, z mm; pitch deg (start high & safe)


def calibration_path(robot_id: str) -> pathlib.Path:
    root = os.environ.get("HF_LEROBOT_HOME") or (
        pathlib.Path.home() / ".cache" / "huggingface" / "lerobot"
    )
    return pathlib.Path(root) / "calibration" / "robots" / "so_follower" / f"{robot_id}.json"


def parse_args(argv: list[str]) -> tuple[float, float, float, float, bool]:
    go = "--go" in argv
    nums = [a for a in argv if a != "--go"]
    x, y, z, pitch = DEFAULT_TARGET
    if len(nums) >= 3:
        x, y, z = float(nums[0]), float(nums[1]), float(nums[2])
    if len(nums) >= 4:
        pitch = float(nums[3])
    return x, y, z, pitch, go


def print_plan(x: float, y: float, z: float, pitch: float) -> bool:
    """Print the IK plan for the target. Returns True if it's safely solvable."""
    cx, cy, cz, clamped = clamp_to_workspace(x, y, z)
    if clamped:
        print(f"  [safety] target ({x:.0f},{y:.0f},{z:.0f}) outside workspace -> "
              f"nearest ({cx:.0f},{cy:.0f},{cz:.0f})")

    sol = solve_ik(cx, cy, cz, pitch)
    model = forward_kinematics(sol.joints)
    in_lim = all(within_limits(j, sol.joints[j]) for j in ARM_JOINTS)

    print(f"\n  desired REAL target : ({cx:.0f}, {cy:.0f}, {cz:.0f}) mm, pitch {pitch:.0f} deg")
    print(f"  arm-degree command  : " + ", ".join(
        f"{j}={sol.joints[j]:.1f}" for j in ARM_JOINTS))
    print(f"  -> model tip of that command (where the §5 correction aims): "
          f"({model[0]:.0f}, {model[1]:.0f}, {model[2]:.0f}) mm")
    print(f"     (the command intentionally aims FARTHER/HIGHER than the target;")
    print(f"      the real arm's droop should pull the tip back onto the target.)")
    print(f"  reachable={sol.reachable}  within_soft_limits={in_lim}")
    if not (sol.reachable and in_lim):
        print("  !! NOT safely solvable (out of reach or a joint hits its soft limit).")
        return False
    return True


def main() -> None:
    x, y, z, pitch, go = parse_args(sys.argv[1:])
    print("=" * 72)
    print("  Stage 2 ruler check — first real top-down move (§5 direction confirm)")
    print("=" * 72)

    ok = print_plan(x, y, z, pitch)

    if not go:
        print("\nDRY RUN — nothing moved. Re-run with --go to drive the real arm.")
        return
    if not ok:
        raise SystemExit("\nRefusing to move: target not safely solvable (see above).")

    robot_id = os.environ.get("LIMBIC_ROBOT_ID", "bronny")
    cal = calibration_path(robot_id)
    if not cal.exists():
        raise SystemExit(
            f"\nNo calibration file for id {robot_id!r} at {cal}. Aborting "
            "(connecting without it would launch interactive calibration)."
        )

    # Import here so a dry run never needs lerobot/hardware.
    from limbic.control.arm import RobotArm
    from limbic.control.backends import RealBackend
    from limbic.control.config import load_config

    cfg = load_config()
    if cfg.port is None:
        raise SystemExit("\nNo serial port found. Plug in the arm or set $LIMBIC_PORT.")

    print(f"\nConnecting real SO-101 on {cfg.port} as {robot_id!r} and moving SLOW...")
    arm = RobotArm(config=cfg, backend=RealBackend(cfg), verbose=True)
    arm.connect()
    try:
        arm.move_to_xyz(x, y, z, approach_pitch_deg=pitch, slow=True)
        print("\nMOVE COMPLETE — arm holding pose (torque engaged).")
        print("RULER-MEASURE the tool tip now, in the table frame (origin under the")
        print("shoulder-pan axis, on the table surface):")
        print(f"   forward from the pan axis : expect {x:.0f} mm")
        print(f"   lateral (+y = left)       : expect {y:.0f} mm")
        print(f"   height above the table    : expect {z:.0f} mm")
        print("\nIf the height lands LOW/short of target, the §5 correction is "
              "inverted — report back and STOP before any grasp.")
    finally:
        arm.disconnect()
        print("Disconnected (torque left engaged; arm holds its pose).")


if __name__ == "__main__":
    main()
