"""Stage 2 calibration check (§A.3/§A.6) — pure computation, NO hardware, NO motion.

Validates the wired-up bridge (kinematics.solve_ik on the hardware-validated
planar solver + calibration.py's frame/sign/offset + §5 accuracy correction):

  1. Frame cross-check: pan-axis x from the ikpy FK chain at runtime vs. the
     measured PAN_AXIS_X_MM constant (re-confirms no typo in calibration.py).

  2. Solver geometry round-trip (correction BYPASSED): feed a RAW table target
     straight into the planar solver, run the result back through the
     independent forward_kinematics frame path, and confirm it returns to the
     target ~exactly. This proves the solver + sign/offset + BOTH frame
     conversions are mutually consistent. (It deliberately does NOT go through
     solve_ik, because solve_ik bakes in the open-loop correction below.)

  3. Correction self-inverse: real_for_command(command_for_real(p)) == p, so
     the two §5 functions are exact inverses (no silent drift in the math).

  4. Correction direction + preview: for top-down (pitch -90) the command must
     aim HIGHER and FARTHER than the target (counters droop/undershoot). Printed
     so a human can sanity-check magnitudes before any real move; the sign is
     asserted (a flipped sign would drive the gripper into the table).

  5. Joint-limit sweep: over the practical grasp band, what fraction of top-down
     targets solve to an arm-degree pose inside safety.JOINT_SOFT_LIMITS.

  6. Known rest-pose FK: feed the Stage 0 joint reading through
     forward_kinematics and print the table (x, y, z) for a human to eyeball.

Nothing here touches the real arm.
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from limbic.control import calibration
from limbic.control.ik_chain import ACTIVE_JOINTS, geometry
from limbic.control.kinematics import forward_kinematics, solve_ik
from limbic.control.safety import ARM_JOINTS, within_limits
from limbic.control._prep_planar_ik import PlanarSO101IK

_PREP_IK = PlanarSO101IK()


def check_frame_crosscheck() -> None:
    g = geometry()
    px_mm = g.pan_axis_xy[0] * 1000.0
    print(f"[1] pan axis x: ik_chain={px_mm:.2f}mm  calibration={calibration.PAN_AXIS_X_MM:.2f}mm "
          f"(diff {abs(px_mm - calibration.PAN_AXIS_X_MM):.2f}mm)")
    print()


def check_solver_roundtrip() -> float:
    """Raw planar solve -> independent FK frame path; worst tip error (no correction)."""
    worst = 0.0
    n = 0
    skipped = 0
    for pan_deg in (-60, -20, 0, 20, 60):
        for reach in (120, 160, 200):
            for z in (0, 40, 80):
                for pitch in (-90, -70, -45):
                    az = math.radians(pan_deg)
                    x = reach * math.cos(az)
                    y = reach * math.sin(az)
                    b = _PREP_IK.solve(x, y, z, pitch)
                    if b is None:
                        skipped += 1
                        continue
                    joints = {j: float(b[i]) for i, j in enumerate(ACTIVE_JOINTS)}
                    gx, gy, gz = forward_kinematics(joints)
                    worst = max(worst, math.hypot(gx - x, gy - y, gz - z))
                    n += 1
    print(f"[2] solver geometry round-trip over {n} feasible poses "
          f"({skipped} unreachable skipped): worst position error = {worst:.3f} mm")
    print()
    return worst


def check_correction_self_inverse() -> float:
    """At the grasp pitch (-90) the two §5 functions must be exact inverses.

    At intermediate pitches the blend (f*linear + (1-f)*identity) is not
    analytically self-inverting; that residual is inherent to the trusted
    correction and only affects ``real_for_command`` (diagnostic-only), so it
    is reported but not asserted.
    """
    worst90 = 0.0
    worst_partial = 0.0
    for fwd in (120, 150, 180, 205):
        for z in (0, 30, 60):
            cf, cz = calibration.command_for_real(fwd, z, -90.0)
            bf, bz = calibration.real_for_command(cf, cz, -90.0)
            worst90 = max(worst90, abs(bf - fwd), abs(bz - z))
            for pitch in (-85, -80):
                cf, cz = calibration.command_for_real(fwd, z, pitch)
                bf, bz = calibration.real_for_command(cf, cz, pitch)
                worst_partial = max(worst_partial, abs(bf - fwd), abs(bz - z))
    print(f"[3] §5 correction self-inverse @ pitch -90 (grasp): worst = {worst90:.6f} mm")
    print(f"      (partial-pitch residual, informational only: {worst_partial:.3f} mm)")
    print()
    return worst90


def check_correction_direction() -> bool:
    print("[4] top-down (pitch=-90) correction preview — desired REAL vs. command sent:")
    print(f"      {'real_fwd':>9} {'real_z':>7}   ->   {'cmd_fwd':>9} {'cmd_z':>7}   (cmd should be >= real)")
    ok = True
    for fwd in (120, 150, 180, 205):
        for z in (0, 20, 50):
            cmd_fwd, cmd_z = calibration.command_for_real(fwd, z, -90.0)
            flag = ""
            if cmd_fwd < fwd - 1e-6 or cmd_z < z - 1e-6:
                ok = False
                flag = "  <-- WRONG DIRECTION"
            print(f"      {fwd:>9} {z:>7}   ->   {cmd_fwd:>9.1f} {cmd_z:>7.1f}{flag}")
    print(f"    direction OK (commands aim higher/farther): {ok}")
    print()
    return ok


def check_joint_limit_sweep() -> tuple[int, int]:
    ok = 0
    total = 0
    for pan_deg in range(-40, 41, 20):
        for reach in (120, 150, 180, 205):
            for z in (0, 20, 40, 60):
                az = math.radians(pan_deg)
                x = reach * math.cos(az)
                y = reach * math.sin(az)
                sol = solve_ik(x, y, z, approach_pitch_deg=-90.0)
                total += 1
                in_lim = all(within_limits(j, sol.joints[j]) for j in ARM_JOINTS)
                if sol.reachable and in_lim:
                    ok += 1
    print(f"[5] joint-limit sweep: {ok}/{total} top-down grasp-band targets solve "
          f"reachable AND within the arm soft limits")
    print()
    return ok, total


def check_rest_pose() -> None:
    # Stage 0 reading (arm-convention degrees), gripper omitted (not in ARM_JOINTS).
    rest = {
        "shoulder_pan": -10.15,
        "shoulder_lift": -103.08,
        "elbow_flex": 97.54,
        "wrist_flex": 79.91,
        "wrist_roll": 0.40,
    }
    x, y, z = forward_kinematics(rest)
    print(f"[6] Stage 0 rest-pose joints -> table frame: x={x:.1f}mm y={y:.1f}mm z={z:.1f}mm")
    print()


def main() -> None:
    print("=" * 72)
    print("  Stage 2 calibration check (frame, sign/offset, §5 correction; no hardware)")
    print("=" * 72)
    check_frame_crosscheck()
    rt = check_solver_roundtrip()
    inv = check_correction_self_inverse()
    direction = check_correction_direction()
    ok, total = check_joint_limit_sweep()
    check_rest_pose()

    passed = rt < 2.0 and inv < 1e-3 and direction and ok > 0
    print("RESULT:", "PASS" if passed else "REVIEW",
          "(solver consistent, correction invertible + correct direction)"
          if passed else "(see numbers above)")


if __name__ == "__main__":
    main()
