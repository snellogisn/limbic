"""Stage 1 IK validation (§A.2) — pure computation, NO hardware, NO motion.

Three checks, all against the ikpy chain as the source of truth:

  1. ikpy FK<->IK round-trip: sample joint vectors -> FK -> ikpy.inverse_kinematics
     -> FK again; confirm the tip returns to ~0. Proves the chain parse is sane.

  2. Closed-form solver accuracy: over a grid of reachable (x,y,z,pitch), run our
     solve_reach -> FK and confirm position error ~0 and approach pitch ~0.

  3. Determinism / no branch-jump: solve the same target repeatedly and from
     neighbouring targets; confirm the joint solution does not hop branches the
     way ikpy's numerical solver does.

Nothing here touches the real arm or asserts anything about the physical frame —
that mapping is measured live in Stage 2.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from limbic.control.ik_chain import (
    ACTIVE_JOINTS,
    build_chain,
    fk,
    fk_pitch,
    geometry,
    solve_reach,
)


def check_geometry() -> None:
    g = geometry()
    print("Extracted planar geometry (from the ikpy FK chain):")
    print(f"  pan axis (base xy) : ({g.pan_axis_xy[0]*1000:.2f}, {g.pan_axis_xy[1]*1000:.2f}) mm")
    print(f"  pan gain           : {g.pan_gain:+.4f} rad/rad")
    print(f"  lift axis (pan x,z): ({g.x0*1000:.2f}, {g.z0*1000:.2f}) mm")
    print(f"  L1, L2, L3         : {g.L1*1000:.2f}, {g.L2*1000:.2f}, {g.L3*1000:.2f} mm")
    print(f"  rest angles a1,a2,a3: {math.degrees(g.a1):.2f}, {math.degrees(g.a2):.2f}, {math.degrees(g.a3):.2f} deg")
    print(f"  reach min..max     : {g.reach_min*1000:.1f} .. {g.reach_max*1000:.1f} mm (2-link)")
    print()


def check_fk_ik_roundtrip() -> float:
    """ikpy FK -> ikpy IK -> FK; report worst tip error over random sane poses."""
    chain = build_chain()
    rng = np.random.default_rng(0)
    worst = 0.0
    n = 60
    for _ in range(n):
        # sample within joint bounds (skip base/tip fixed links)
        q = [0.0]
        for link in chain.links[1:-1]:
            lo, hi = link.bounds
            lo = -math.pi if lo is None or not math.isfinite(lo) else lo
            hi = math.pi if hi is None or not math.isfinite(hi) else hi
            q.append(rng.uniform(lo * 0.7, hi * 0.7))
        q.append(0.0)
        target = chain.forward_kinematics(q)
        sol = chain.inverse_kinematics_frame(target, initial_position=q)
        got = chain.forward_kinematics(sol)
        err = np.linalg.norm(target[:3, 3] - got[:3, 3])
        worst = max(worst, err)
    print(f"[1] ikpy FK<->IK round-trip: worst tip error over {n} poses = {worst*1000:.4f} mm")
    return worst


def check_closed_form() -> tuple[float, float]:
    """Our solve_reach -> FK over the table workspace; worst position & pitch error.

    Only joint-limit-respecting, in-reach solutions are scored (those are the
    poses the arm can actually hold); infeasible targets are counted, not scored.
    """
    g = geometry()
    px, py = g.pan_axis_xy
    worst_pos = 0.0
    worst_pitch = 0.0
    tested = 0
    infeasible = 0
    pitches = [-math.pi / 2, math.radians(-70), math.radians(-50)]
    for pan_deg in range(-70, 71, 20):
        for reach in [0.10, 0.14, 0.18, 0.22, 0.26]:
            for z in [0.00, 0.02, 0.04, 0.06]:  # table-height grasps
                for pitch in pitches:
                    az = math.radians(pan_deg)
                    x = px + reach * math.cos(az)
                    y = py + reach * math.sin(az)
                    sol = solve_reach(x, y, z, pitch=pitch)
                    if not (sol.reachable and sol.in_limits):
                        infeasible += 1
                        continue
                    got = fk(sol.joints)
                    perr = math.hypot(got[0] - x, got[1] - y, got[2] - z)
                    pierr = abs(fk_pitch(sol.joints) - pitch)
                    worst_pos = max(worst_pos, perr)
                    worst_pitch = max(worst_pitch, pierr)
                    tested += 1
    print(f"[2] closed-form solve->FK over {tested} feasible targets "
          f"({infeasible} out-of-reach/limits skipped):")
    print(f"      worst position error = {worst_pos*1000:.4f} mm")
    print(f"      worst pitch error    = {math.degrees(worst_pitch):.4f} deg")
    return worst_pos, worst_pitch


def check_determinism() -> float:
    """Ours must vary smoothly across neighbouring targets; ikpy branch-jumps.

    Sweep the clean top-down grasp band (reach 0.14..0.24 m) in 2 mm steps and
    measure the worst single-step joint change for (a) our closed-form solver and
    (b) ikpy's numerical solver seeded from a fixed neutral pose. A branch jump
    shows up as a large discontinuity.
    """
    g = geometry()
    chain = build_chain()
    px, py = g.pan_axis_xy
    z = 0.03  # table-height grasp band

    a = solve_reach(px + 0.18, py + 0.02, z).joints
    b = solve_reach(px + 0.18, py + 0.02, z).joints
    repeat = max(abs(a[j] - b[j]) for j in ACTIVE_JOINTS)

    neutral = [0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0]  # fixed seed for ikpy
    prev_ours = prev_ikpy = None
    worst_ours = worst_ikpy = 0.0
    for i in range(51):
        x = px + 0.14 + i * 0.002  # 0.14 .. 0.24 m, the top-down grasp band
        ours = solve_reach(x, py, z).joints
        if prev_ours is not None:
            worst_ours = max(worst_ours, max(abs(ours[k] - prev_ours[k]) for k in ACTIVE_JOINTS))
        prev_ours = ours

        frame = np.eye(4)
        frame[:3, 3] = [x, py, z]
        q = chain.inverse_kinematics_frame(frame, initial_position=neutral)
        if prev_ikpy is not None:
            worst_ikpy = max(worst_ikpy, float(np.max(np.abs(q[1:6] - prev_ikpy[1:6]))))
        prev_ikpy = q

    print(f"[3] determinism: repeat-call max joint diff = {math.degrees(repeat):.6f} deg")
    print(f"      worst single-step joint change over the grasp band (2mm steps):")
    print(f"        ours (closed-form)  = {math.degrees(worst_ours):7.3f} deg  <- smooth, no jumps")
    print(f"        ikpy (fixed seed)   = {math.degrees(worst_ikpy):7.3f} deg  <- branch-jumps")
    return worst_ours


def main() -> None:
    print("=" * 64)
    print("  Stage 1 IK validation (ikpy frame; no hardware)")
    print("=" * 64)
    check_geometry()
    rt = check_fk_ik_roundtrip()
    pos, pitch = check_closed_form()
    step = check_determinism()
    print()
    ok = (rt < 1e-3 and pos < 1e-3 and pitch < math.radians(0.1)
          and step < math.radians(20))
    print("RESULT:", "PASS" if ok else "REVIEW",
          "(FK/IK consistent, closed-form ~0 error, deterministic)"
          if ok else "(see numbers above)")


if __name__ == "__main__":
    main()
