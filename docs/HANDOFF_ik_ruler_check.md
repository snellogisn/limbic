# Handoff — IK Verification by Ruler Check (Part A §A.2/§A.6), for the Mac rig

**You are an AI assistant** helping a human verify the SO-101 inverse kinematics
on a **Mac-connected arm**. The IK and its open-loop accuracy correction were
built and ruler-verified on a *different physical arm*. Your job: command known
table targets, have the human ruler-measure where the tip lands, **characterize
any per-axis error, and — only with the user's approval — add a correction curve
for the one direction that needs it.** Read this whole doc before moving anything.

---

## The golden rules (do not break these)
- **The arm is PHYSICAL; the human is your eyes and hands.** For every step, tell
  them the exact command to run and exactly what to ruler-measure. They report
  numbers back. **Never invent or assume a hardware number.** If unsure, ask.
- **Barrel-jack power only — never USB.**
- **Ask before any risky/fast/extreme motion.** Start every probe **HIGH and
  SLOW** (high z, slow profile) so a wrong sign can't drive the tip into the table.
- **Verify with the user BEFORE editing any IK / correction file.** Show the data,
  state your hypothesis, get explicit approval, change ONE thing, re-measure.
- **Commit each working checkpoint.** Never break a known-good state without a
  commit to fall back to.
- **Don't change the §0.3 shared interfaces** (the team API).

---

## The table frame (memorize — every target uses it)
Origin directly **under the shoulder-pan axis, on the table surface**.
**+x = forward** (away from the base), **+y = left**, **+z = up**. **Units = mm.**

## How the IK is layered (so you know what you're testing)
1. **Geometry** — ikpy from the SO-101 URDF + a closed-form planar solver. This is
   universal SO-101; it should be correct on any SO-101.
2. **ikpy ↔ arm sign/offset** (`limbic/control/calibration.py`, `_SIGN`/`_OFFSET`,
   `arm_deg = sign·ikpy_deg + offset`). These come from the lerobot calibration
   convention. They were measured on the *other* arm — if your arm is calibrated
   the standard way they should be close, but **that is part of what you're
   verifying.**
3. **Open-loop accuracy correction** (`calibration.py` §5) — compensates the real
   tip landing short/low under load (slack + gravity droop). **This is
   rig-specific and most likely to differ on your arm.**

A discrepancy could live in any layer — your job is to find *which*.

---

## Setup on the Mac (prerequisites)
- **Serial port:** not `COM7`. Find it: `ls /dev/cu.usbserial-* /dev/cu.usbmodem*`.
  Set `export LIMBIC_PORT=/dev/cu.usbserial-XXXX`.
- **Calibration / robot id:** the default `robot_id` is `bronny`, whose lerobot
  calibration file belongs to the *other* arm. **Your arm needs its own lerobot
  calibration** (or a copied one) under its own id; set
  `export LIMBIC_ROBOT_ID=<your_id>`. A *missing* calibration makes lerobot launch
  its interactive range-of-motion routine (it drives every joint to its limits) —
  don't connect until a calibration exists for your id.
- **Sanity-connect read-only first** to confirm comms + that joints read sanely
  before any motion: `python scripts/arm_connect_check.py` (reads all six joints,
  no motion).

---

## The procedure
1. **Home it:** `python scripts/go_home.py` — drives every motor to the centre of
   its range (all arm joints 0°, gripper halfway). Confirm it *looks* centred.
2. **One ruler probe:**
   ```
   python scripts/stage2_ruler_check.py 180 0 60          # DRY RUN: prints the IK plan
   python scripts/stage2_ruler_check.py 180 0 60 --go     # SLOW move, then HOLDS for measuring
   ```
   Args are REAL table-frame mm: `x y z [pitch]` (default pitch −90 = straight
   down). It clamps to the workspace, applies the §5 correction, streams a slow
   move, holds torque so the tip stays put, and prints exactly what to measure.
3. **Measure** (have the human ruler these, in the table frame):
   - forward distance from the pan axis (**x**),
   - lateral offset (**y**, +left),
   - height above the table (**z**).
   Record **commanded vs measured** for each.
4. **Start safe, then grid.** First probe high + centerline: `180 0 60`. Then
   sweep once you trust the z direction:
   - centerline, varying reach: `x ∈ {120,160,200,240}, y=0, z=40`
   - off-centerline: `y ∈ {−120,−60,+60,+120}` at `x≈180, z=40`
   - height: `z ∈ {20,40,60}` at `x≈180, y=0`
   Keep the human in the loop; ask before each new far/low target.

---

## Interpreting the data — ISOLATE ONE cardinal direction
Build a small table of **error = measured − commanded** for **x, y, z
separately**. Then classify the *dominant* axis (focus on one at a time):

| Pattern in one axis | Most likely cause | Fix (don't guess — confirm first) |
|---|---|---|
| **Constant** offset everywhere | a frame/constant, **not** the IK. A constant **z** miss everywhere = **base height** (`BASE_HEIGHT_ABOVE_TABLE_MM`), per §A.6. | adjust the single constant, re-measure |
| **Grows with reach / height** (pose-dependent) | droop / slack — needs a **correction curve** for that axis (exactly like the z curve below) | build + invert a fit for that axis |
| **Linear scale** (e.g. lands at 0.85× commanded) | a slope error | a linear correction (slope+intercept) |
| **Wildly wrong / wrong sign / jumps** | a **sign/offset** (calibration convention) problem — a *different* class of bug, **not** a correction curve | re-check `_SIGN`/`_OFFSET` with the user |

**Rules of isolation:**
- Change **one axis at a time**. Tune it, re-measure the *same* grid, confirm it
  improved, commit — *then* look at the next axis. Coupled changes make it
  impossible to tell what helped.
- **y (lateral) has no correction by design** (azimuth is preserved). If **y** is
  systematically off, suspect the **base not being level** or a **pan sign/offset**
  *before* adding any y curve — bring it to the user.

---

## The z correction is your worked template
The existing open-loop correction is in **`limbic/control/calibration.py` §5**:
- `command_for_real(real_fwd, real_z, pitch) -> (cmd_fwd, cmd_z)` — **the one you
  call**: pass the *desired REAL* position, get back the *command* to send so the
  drooping tip lands on target.
- `real_for_command(...)` — the forward fit (predict where a raw command lands).
  **Diagnostic only — never drive the arm with it.**
- It was built by: command a **sweep** along the axis, ruler the **real** landing,
  fit `real = f(command)`, then **invert** it (solve for the command that yields a
  desired real). The z path also has a **reach-dependent dropoff** table and a
  **pitch blend** (droop fades as the wrist tilts off vertical). The forward axis
  already has a linear fit (`REAL_FWD_COEF`) you can read as an example.

To build the same kind of curve for whichever axis your arm needs:
1. With the user, sweep that axis (command a ladder, ruler each real landing).
2. Fit `real = f(command)` for that axis.
3. Invert it and fold it into `command_for_real` for that axis only.
4. Re-measure the grid, confirm, commit.

---

## CRITICAL — confirm before changing IK
Before editing **any** IK / correction file:
1. Show the user the per-axis **error table**.
2. Name the **single** direction you believe is off and your **hypothesis**
   (constant vs curve vs scale vs sign).
3. Get **explicit approval**.
4. Make the **isolated** change, re-measure the **same** targets, confirm it
   improved, and **commit**.

And the first-move safety check that gated this on the original arm: the §5
**z-direction must be confirmed on YOUR arm** before any near-table grasp —
command a known target, confirm the tip lands **AT** it (not low/short). **If it
lands low, a correction is inverted — stop and fix before continuing.**
