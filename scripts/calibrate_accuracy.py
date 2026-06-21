"""Systematic open-loop accuracy calibration (§A.6 / §5) — the move/measure/fit loop.

This replaces ad-hoc "tweak a constant, move, get surprised" with a reproducible
pipeline backed by a persistent dataset:

  plan                      print a deliberate grid of targets to measure
  collect X Y Z --go        move the real arm to (X,Y,Z); then ruler-measure
  collect X Y Z --measured MX MY MZ
                            record where it actually landed (appends a sample)
  fit                       refit from the WHOLE dataset (robust + LOO cross-
                            validation), print honest expected accuracy, save it
  check X Y Z --go          move to a held-out target to see the live error

Samples are (aim -> real) pairs in planar (forward-reach, z) mm, stored in
calibration/accuracy_samples.csv. 'aim' is the model tip the IK hit (correction-
independent), so the dataset accumulates cleanly forever. The fitted model in
calibration/accuracy_model.json supersedes the affine constants automatically.

Typical first run:
    python scripts/calibrate_accuracy.py plan
    # for each printed target:
    python scripts/calibrate_accuracy.py collect 160 0 60 --go     # arm moves, you measure
    python scripts/calibrate_accuracy.py collect 160 0 60 --measured 165 0 84
    ...
    python scripts/calibrate_accuracy.py fit                       # see expected accuracy
    python scripts/calibrate_accuracy.py check 175 0 45 --go       # validate on a new point

Safety: BARREL-JACK power only, workspace clear, hand ready. Moves are SLOW and
top-down; start high (z>=50) so a bad z is caught before the tip nears the table.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from limbic.control import accuracy_model as am
from limbic.control import calibration
from limbic.control.kinematics import forward_kinematics, solve_ik
from limbic.control.safety import ARM_JOINTS, clamp_to_workspace, within_limits

CSV_PATH = pathlib.Path(__file__).resolve().parent.parent / "calibration" / "accuracy_samples.csv"
MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "calibration" / "accuracy_model.json"

# A deliberate grid spanning the demo envelope: centerline reach sweep x z levels,
# plus a few off-axis points to anchor lateral behaviour. ~16 measurements.
PLAN_CENTER_REACH = (130, 160, 190, 210)
PLAN_Z = (25, 60, 100)
PLAN_OFFAXIS = [(160, 60, 40), (160, -60, 40), (190, 80, 40), (190, -80, 40)]


def _aim_for_target(x: float, y: float, z: float, pitch: float):
    """The model tip (= aim) the IK hits for this target, and whether it's safe."""
    cx, cy, cz, clamped = clamp_to_workspace(x, y, z)
    sol = solve_ik(cx, cy, cz, pitch)
    ax, ay, az = forward_kinematics(sol.joints)
    in_lim = all(within_limits(j, sol.joints[j]) for j in ARM_JOINTS)
    return (ax, ay, az), sol, in_lim, clamped


def cmd_plan() -> None:
    print("Planned calibration grid (run each, then `fit`). y=0 is the centerline.\n")
    targets = [(r, 0, z) for r in PLAN_CENTER_REACH for z in PLAN_Z] + PLAN_OFFAXIS
    for (x, y, z) in targets:
        (ax, ay, az), sol, in_lim, clamped = _aim_for_target(x, y, z, -90.0)
        flag = "" if (sol.reachable and in_lim and not clamped) else "  [!] check reach/limits"
        print(f"  python scripts/calibrate_accuracy.py collect {x} {y} {z} --go{flag}")
    print(f"\n{len(targets)} targets. After moving to each and measuring, record with "
          "`collect X Y Z --measured MX MY MZ`, then run `fit`.")


def cmd_collect(argv: list[str]) -> None:
    go = "--go" in argv
    measured = None
    if "--measured" in argv:
        i = argv.index("--measured")
        measured = tuple(float(v) for v in argv[i + 1:i + 4])
    note = ""
    if "--note" in argv:
        note = argv[argv.index("--note") + 1]
    nums = []
    skip = 0
    for j, a in enumerate(argv):
        if skip:
            skip -= 1
            continue
        if a == "--measured":
            skip = 3
            continue
        if a == "--note":
            skip = 1
            continue
        if a == "--go":
            continue
        nums.append(a)
    x, y, z = (float(nums[0]), float(nums[1]), float(nums[2]))
    pitch = float(nums[3]) if len(nums) >= 4 else -90.0

    (ax, ay, az), sol, in_lim, clamped = _aim_for_target(x, y, z, pitch)
    aim_fwd, aim_z = math.hypot(ax, ay), az
    print(f"target ({x:.0f},{y:.0f},{z:.0f}) pitch {pitch:.0f}")
    print(f"  IK aim (model tip) = reach {aim_fwd:.1f}, z {aim_z:.1f}  "
          f"reachable={sol.reachable} in_limits={in_lim}")

    if measured is not None:
        mx, my, mz = measured
        am.record_landing(CSV_PATH, (x, y, z), (mx, my, mz), pitch,
                          source="collect", note=note)
        print(f"  RECORDED: aim(reach {aim_fwd:.1f}, z {aim_z:.1f}) -> "
              f"real(reach {math.hypot(mx, my):.1f}, z {mz:.1f}). Now run `fit`.")
        return

    if not go:
        print("\nDry run. Add --go to move the arm, or --measured MX MY MZ to record a landing.")
        return
    if not (sol.reachable and in_lim):
        raise SystemExit("Refusing to move: target not safely solvable (out of reach / joint limit).")

    _drive(x, y, z, pitch)
    print("\nMOVE COMPLETE — arm holding. Ruler-measure the tip (table frame), then run:")
    print(f"  python scripts/calibrate_accuracy.py collect {x:.0f} {y:.0f} {z:.0f} --measured MX MY MZ")


def cmd_fit() -> None:
    samples = am.load_samples(CSV_PATH)
    model, report = am.fit_model(samples)
    print(f"Loaded {report['n_total']} samples ({report['n_vertical']} vertical-approach).")
    if report["status"] != "ok":
        print("  " + report["message"])
        return
    print(f"  used {report['n_used']}, dropped {report['n_outliers']} outlier(s).")
    print(f"  feature set: fwd={report['feature_set']['fwd']}, z={report['feature_set']['z']}")
    acc = report["expected_accuracy_mm"]
    print(f"\n  EXPECTED ACCURACY (leave-one-out CV): ±{acc['fwd']} mm reach, ±{acc['z']} mm z")
    print(f"  (training residual: {report['train_rms_mm']['fwd']}/{report['train_rms_mm']['z']} mm)")
    if report["outliers"]:
        print("\n  Outliers dropped (re-measure these if you want them back):")
        for o in report["outliers"]:
            print(f"    real={o['real']} resid={o['resid_mm']}mm  {o['source']} {o['note']}")
    print("\n  Per-point residuals (aim-space mm; large = noisy/inconsistent point):")
    for r in report["residuals"]:
        k = " " if r["kept"] else "X"
        print(f"   [{k}] real={r['real']}  d_fwd={r['resid_fwd']:+.1f} d_z={r['resid_z']:+.1f}  {r['source']}")
    model.save(MODEL_PATH)
    calibration.reload_accuracy_model()
    print(f"\n  Saved {MODEL_PATH.name}; it now supersedes the affine constants.")
    if max(acc["fwd"], acc["z"]) > 12:
        print("  NOTE: expected error > ~1cm. Collect more / cleaner points (repeats per "
              "target average out ruler noise), or lean on the visual closed-loop for the last mm.")


def cmd_check(argv: list[str]) -> None:
    go = "--go" in argv
    nums = [a for a in argv if a != "--go"]
    x, y, z = float(nums[0]), float(nums[1]), float(nums[2])
    pitch = float(nums[3]) if len(nums) >= 4 else -90.0
    cf, cz = calibration.command_for_real(math.hypot(x, y), z, pitch)
    print(f"check target ({x:.0f},{y:.0f},{z:.0f}): correction aims reach {cf:.1f}, z {cz:.1f}")
    model = am.AccuracyModel.load(MODEL_PATH)
    if model is not None:
        print(f"  fitted-model expected error: ±{model.expected_accuracy_mm[0]:.1f}/"
              f"{model.expected_accuracy_mm[1]:.1f} mm (reach/z)")
    if not go:
        print("  Dry run. Add --go to move and ruler-check the landing.")
        return
    (ax, ay, az), sol, in_lim, _ = _aim_for_target(x, y, z, pitch)
    if not (sol.reachable and in_lim):
        raise SystemExit("Refusing to move: not safely solvable.")
    _drive(x, y, z, pitch)
    print(f"\nMOVE COMPLETE — measure the tip; it should land at ({x:.0f},{y:.0f},{z:.0f}).")
    print("  If it's off, record it as a sample: "
          f"collect {x:.0f} {y:.0f} {z:.0f} --measured MX MY MZ  (then re-fit).")


def _drive(x: float, y: float, z: float, pitch: float) -> None:
    """Connect the real arm and stream a SLOW top-down move; leaves torque on."""
    from limbic.control.arm import RobotArm
    from limbic.control.backends import RealBackend
    from limbic.control.config import load_config

    cfg = load_config()
    if cfg.port is None:
        raise SystemExit("No serial port found. Plug in the arm or set $LIMBIC_PORT.")
    print(f"Connecting real arm on {cfg.port} and moving SLOW...")
    arm = RobotArm(config=cfg, backend=RealBackend(cfg), verbose=True)
    arm.connect()
    try:
        arm.move_to_xyz(x, y, z, approach_pitch_deg=pitch, slow=True)
    finally:
        arm.disconnect()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "plan":
        cmd_plan()
    elif cmd == "collect":
        cmd_collect(rest)
    elif cmd == "fit":
        cmd_fit()
    elif cmd == "check":
        cmd_check(rest)
    else:
        print(f"unknown command {cmd!r}\n")
        print(__doc__)


if __name__ == "__main__":
    main()
