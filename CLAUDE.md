# Project Guide

## How to use this document
This is the shared spec for the whole team. It has two kinds of content:

- **Part 0 — Shared foundations** (everyone reads): what we're building, the architecture,
  the **cross-component interfaces**, the environment, and the safety + physical rules.
- **Parts A / B / C — the three components**, each owned and edited by one person/track:
  - **Part A — The Arm:** kinematics, motion primitives, and table localization (the *Body*).
  - **Part B — Vision:** open-vocabulary object detection (the *world model / Senses*).
  - **Part C — The Brain:** the LLM orchestrator that decides and commands (the *Brain*).

**Editing rule:** edit **your own component section** freely as your design evolves — the
sections are self-contained so you won't step on anyone else. The **one thing you must NOT
change alone is §0.3 (the Shared Interfaces)** — those are the team's API; everyone builds
against them, so change them only by team agreement. If you're working on one component, read
**Part 0 + your Part**; skim the others for context.

---

# Part 0 — Shared foundations (everyone)

## 0.1 What we're building
A general-purpose **tabletop manipulation** system. A person gives a natural-language command;
the **Brain (Claude, via the API)** reasons and issues **motion primitives as tool calls**; a
**vision model** locates objects in the camera image; a one-time **camera-to-table
calibration** converts image pixels into table coordinates; and an **IK solver** turns those
coordinates into joint angles for an **SO-101 robot arm**. Scope is deliberately bounded to
tabletop **pick / place / move / stack** (richer verbs like push/throw are *composed*
from the same primitives, not built separately).

## 0.2 Architecture & data flow
Three layers, with a single clean data path:

```
  [overhead camera frame]
        |
        ├──> (Part B) DETECTION:    image + object name  ->  object pixel (u, v)
        |                                                        |
        └──> (Part A) LOCALIZATION: pixel (u, v)         ->  table coordinate (x, y)
                                                                 |
        (Part C) BRAIN: command + object (x,y)  ->  motion primitive tool calls
                                                                 |
        (Part A) ARM:   primitives -> IK -> joint angles -> the arm moves
```

- **Body** (Part A) = arm + IK + motion primitives **+ pixel→table localization**.
- **Senses** (Part B) = the vision model that finds objects in the image.
- **Brain** (Part C) = the Claude-API reasoning loop that composes the task.

## 0.3 Shared interfaces — the team's API (change only by agreement)
These are the seams between components. They are intentionally small and stable so everyone can
build in parallel. **Do not change a signature here without telling the people on both sides.**

1. **Table coordinate frame.** Origin directly under the **shoulder-pan axis** at the **table
   surface**; **+x = forward** (away from the base), **+y = left**, **+z = up**. **Units =
   millimeters.** Every component that talks about a position uses this frame.

2. **Motion primitive tool surface** (Part A exposes → Part C calls). Small, composable,
   low-level — positions in table-frame mm:
   - `move_to_xyz(x, y, z, pitch)` — pitch sets approach angle (e.g. straight-down for a grasp)
   - `open_gripper()` / `close_gripper()`
   - `lift_by(dz)`
   - `go_home()`
   Out-of-range targets **clamp to the nearest reachable point** (never crash).

3. **Detection → Localization seam** (Part B → Part A). Detection returns
   `[(label, (u, v)), ...]` — the **pixel location** of each found object (the bounding-box
   center). **That pixel is exactly the "click"** the arm side turns into a table coordinate:
   localization converts each `(u, v) -> (x, y)`. A human click and a detected box are
   interchangeable inputs here — same pixel, same path — so vision is a *drop-in* for a click.

4. **Brain-facing perception tool** (→ Part C calls). A single tool
   `detect_objects(prompt) -> [(label, (x, y)), ...]` returning **table coordinates** = Part B's
   detection chained into Part A's localization. **Part C (the brain/integration layer) owns the
   composition** — it calls detection to get the pixel + label, then localization to get the
   table coordinate. Ownership is explicit: **Part B produces the pixel, Part A converts it, Part
   C chains them.** The Brain never sees pixels; it gets object positions in the arm's frame.

## 0.4 Environment & machines
- **Arm stack:** `lerobot[feetech]`, arm on a serial **COM port**, **barrel-jack power — NEVER
  power the arm from USB.** Runs fine on ARM64 or x64.
- **Arch split (important):** the **vision model needs PyTorch**, which installs on **x64
  Windows / macOS / Linux but NOT on ARM64 (Snapdragon) Windows.** So vision (Part B) runs on
  an x64/Mac/Linux machine; the arm + localization + brain run anywhere. Keep `torch` /
  `ultralytics` **out of the base requirements** (an optional extra) so non-x64 installs don't
  break.
- **Per-machine setup:** confirm the **COM port** and the **camera device names** (indices
  shuffle between machines — resolve cameras by name); reinstall the USB-serial driver if the
  port doesn't appear.
- **Where the demo runs:** each component is developed on its owner's machine (the arm +
  localization where the arm is plugged in; vision on the x64 box). The **integrated demo runs on
  the x64 box**, so vision and everything else are co-located. When the arm moves onto the demo
  box, the only things to re-confirm are the **COM port** and the **camera device names** — the
  calibrated rig geometry (IK offsets, base height, extrinsics) **stays valid as long as the
  physical camera/arm/table setup isn't disturbed.** If the rig is physically moved or bumped,
  re-run the extrinsics + a quick z-check.
- Public asset: the SO-101 URDF (`so101_new_calib.urdf` from `TheRobotStudio/SO-ARM100`) — just
  download it.

## 0.5 Operating & safety rules (everyone, but mostly the arm)
- **The arm is physical.** A human is required as eyes/hands for every calibration/measurement
  step — tell them exactly what to run and what to measure; they report back.
- **Ask before destructive/irreversible hardware actions** (re-calibration, extreme or fast
  poses). A bad joint command can drive the arm into the table or itself.
- Every **hardware-specific value is measured on THIS rig** — do not assume numbers from
  anywhere. (Each component lists what it must calibrate live.)

## 0.6 Physical rules — how the arm must behave
Behavioral principles for clean manipulation. **The arm (Part A) enforces these in the
primitives; the Brain (Part C) embeds them in its system prompt** so it composes motions
correctly.
- **Gripper acts in ISOLATION.** The claw opens/closes only with the **arm fully stopped** — no
  other joints moving while it actuates. Settle → actuate → settle. Each grab/drop is a discrete
  settled step.
- **Gripper has actuation LATENCY.** The claw isn't instant. For a **timed/dynamic** release
  (e.g. throwing) the open command must **LEAD** the desired release moment.
- **Speed by task.** SLOW & controlled for **precision/contact** moves (descend-to-grasp,
  lower-to-place, push); normal & smooth for **point-to-point transit**.
- **Soft limits — obey by default, override only when forced.** These rules (gripper-in-isolation,
  slow-for-precision, the joint/reach clamps) are the **norm** — follow them; they're what makes
  manipulation clean. But they're **soft: when, and *only* when, a task genuinely can't be
  accomplished within them, break them.** The worked example is a **throw** — it *requires* moving
  the arm and opening the gripper together, fast, so it deliberately breaks both gripper-isolation
  and slow-for-precision; that coordinated release *is* the throw. Don't break a limit for
  convenience, and still weigh real harm (e.g. a pan limit may be a camera-collision guard, not
  just a preference).
- **Use the arm's FULL capability.** Reason about how the arm physically moves — it tilts,
  extends, flicks the wrist to reach far. Use the full envelope; don't clamp rigidly; verify
  limits by testing into them.
- **Tracing/drawing is done on the HORIZONTAL plane.** Treat a **constant-z plane as a sheet of
  paper lying flat on the table** and draw on it from above: vary **x and y** to draw the shape
  while holding **z fixed**, tool pointing straight down (pitch −90) like a pen on paper. Do
  **not** trace in a vertical (y-z) plane standing up in the air.
- **Grasp rules.** Aim a small offset to one side of the object (claw geometry) and **descend
  WELL into** it (a couple cm, not a tap). This offset applies to **grab/drop ONLY — not push**.
  "Move X cm" = move the claw **tip** that real distance.
- **Lift before retract.** After a grasp, raise the object **straight up** clear of the surface
  before moving sideways (and lower it into place before opening on a drop) — don't drag it
  across the table or clip neighboring objects.

## 0.7 Integration order & protected checkpoints (highest risk — read this)
The components can be built in isolation, but **bringing them together is the scariest, most
failure-prone part — so do it FIRST and incrementally, never as a final step.** The natural
instinct ("finish my part, integrate at the end") is how demos die at hour 22. Connect the pieces
early on stubs, then upgrade each stub to the real thing.

**Order — each step is a working, committed checkpoint:**
1. **Brain → real arm.** Claude issues a primitive for a **hardcoded/typed coordinate** → the arm
   moves. Proves loop → dispatch → primitives → hardware with zero perception. *(The single
   scariest connect — do it first.)*
2. **Brain → arm via a click.** Claude calls `detect_objects`, but the pixel comes from a
   **manual click** (a stand-in) → Part A localization → table → arm. Proves the full
   command → coordinate → motion path with perception stubbed.
3. **Swap in real vision.** Replace the manual click with **Part B detection**. Nothing downstream
   changes — the detected box center is the same pixel a click gave (§0.3 #3). Now it's
   prompt → detection → localization → arm.
4. **Expand + website.** Grow from one task to multi-verb composition (pick/place/push/stack) and
   build the reasoning-display website (§C.5).

**Protected-checkpoint discipline:** the moment a checkpoint works (typed-reach, Brain→arm,
click→arm loop, vision-driven loop, multi-verb), **commit it.** **Never break a working
checkpoint to chase the next feature without committing first** — always keep a known-good state
to return to. This is what saves the demo at hour 22.

---

# Part A — The Arm: Kinematics, Motion & Localization (the Body)
*Owner edits this section. Exposes: the motion primitives (§0.3 #2) and localization (§0.3 #3).*

## A.1 Goal
Type/receive a table-frame XYZ → the arm moves there accurately; and convert any camera pixel
into a correct table coordinate. End state: an object's pixel → table coord → the arm reaches it.

## A.2 Inverse kinematics
- **Forward kinematics + geometry = ikpy** (pure Python; runs on any arch). Build the chain from
  the URDF with the **5 active joints** `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
  wrist_roll`; the IK tip is the **gripper tool point**; the jaw open/close joint is **not** in
  the chain.
- **Reaching IK = a closed-form analytical PLANAR solver, NOT ikpy's optimizer.** ikpy's
  numerical `inverse_kinematics` **branch-jumps** (non-deterministically hops between
  elbow-up/elbow-down → same target, different pose). Replace it with a deterministic closed-form
  solve (0 model error, no jumping):
  - `shoulder_pan` → **azimuth** (which vertical slice the arm reaches in), set from the target.
  - `shoulder_lift + elbow_flex + wrist_flex` → an **analytic 2-link-plus-wrist** problem in that
    vertical plane, solving tip **(radius, height) + approach pitch**.
  - `wrist_roll` → held fixed.
  - Extract the planar geometry (link lengths, frame offsets) **once from the ikpy FK chain** so
    the two stay consistent.

## A.3 Conventions (get these right or nothing works)
- **UNITS LANDMINE:** ikpy/URDF use **radians**; the arm uses **degrees**. Convert on every
  solve. This is the #1 bug source.
- **ikpy↔arm frame offset:** the IK joint frame and the arm's joint frame differ per joint by a
  **sign and an offset** (`arm_deg = sign·ikpy_deg + offset`). MEASURED (see A.6).
- **Top-down grasp:** gripper pointing straight down (the grasp pitch).

## A.4 Motion primitives
Implement the §0.3 tool surface in real table-frame mm, applying IK + the calibrated corrections
+ the soft clamps inside. Smooth ease-in/out interpolation + a convergence check so open-loop
targets settle. A **slow profile** for precision moves, normal for transit (per §0.6). Enforce
the §0.6 physical rules here (gripper isolation, etc.). Add a dynamic fast-release primitive for
throwing (timed gripper open that leads the release).

## A.5 Localization (pixel → table)
- **Intrinsics:** calibrate each camera with a checkerboard.
- **Extrinsics:** detect an **AprilTag** of known size/position with `solvePnP`; compose
  camera→base using the **tag's measured position/orientation in the base frame**. This ties the
  camera to the **same base frame the arm uses** — confirm `camera frame == arm frame` against
  the real arm.
- **pixel→table:** undistort the pixel → ray in the camera → rotate to base → intersect the
  **z = 0 table plane** → `(x, y)`.
- **Multiple cameras:** if used, select the camera on the **side the object is closest to** and
  take its reading entirely (don't average across cameras once each is accurate).

## A.6 Calibrate live (methods; every value is hardware-specific — measure on THIS rig)
Command a pose, a human ruler-measures, repeat.
- **ikpy↔arm sign/offset per joint** — centered pose; measure link/tip positions; resolve each
  sign with a small single-joint move; fit the offsets.
- **Base height above the table** — command a known z on the centerline, ruler the tip, adjust
  the base-height constant until commanded z == measured z. (A *constant* z miss everywhere is
  THIS, not the IK.)
- **Empirical accuracy correction** — the real tip lands short/low of the model under load
  (slack + gravity droop). Command/measure a forward+z sweep and fit a real↔command map. Make
  the z correction **pitch-aware** (droop fades as the wrist tilts). Add a **reach-dependent
  z-dropoff**: close to the base the arm folds and the tip runs low — sweep z across reaches and
  compensate.
- **Claw grip offset** — the claw grips offset from the IK tip; command-and-measure (the lateral
  side offset + any forward offset). Grab/drop only.
- **Camera intrinsics** — checkerboard, ~20 varied images covering the frame edges, target ≲1 px.
- **Camera extrinsics** — `solvePnP` on the tag; verify by projecting the base axes back into the
  image.
- **pixel→table** — validate against known measured points; tune the camera-selection rule.
- **Behavior tuning** — grasp depth, push contact height, throw wind-up/release + the release
  lead time.

**Expected starting points — rough estimates (RE-MEASURE; never trust these).** The values below
are **ballpark estimates** — a head start to narrow the search and sanity-check your own numbers,
**not constants.** Mounting, claw alignment, and camera position all shift them, and a wrong
baked-in number is worse than none, so confirm each by measurement (above); if your measurement
disagrees, it wins.
- **Claw grip offset:** ~**1 cm to the RIGHT** of the object, and **descend ~2 cm INTO** it
  (grab/drop only).
- **Base height:** a constant model-vs-table **z-offset of order ~10 cm**; a *constant* z miss
  everywhere is this, not the IK.
- **Reach-dependent z-dropoff:** close to the base the tip runs **low** (the arm folds) — needs a
  positive z correction that grows as reach shrinks.
- **Reach envelope:** clean **top-down** grasps to ~**23 cm** from the pan axis; the arm
  tilts/extends to ~**31 cm** for coarse moves.
- **ikpy↔arm offsets:** big offsets on **shoulder_lift & elbow_flex (~−80 to −90°)**, **wrist_flex
  small (~−8°)**, **shoulder_pan & wrist_roll ~0**; signs pan **+**, lift/elbow/wrist_flex **−**,
  roll **+**.
- **Off-centerline:** if the base isn't level the tip rides a **couple cm high** away from the
  centerline — keep demo grasps near y≈0 until leveled.

## A.7 Pitfalls
- **AprilTag pose-FLIP:** a single small tag viewed near-straight-down has two near-equal
  `solvePnP` solutions that flip frame-to-frame → the camera position jumps by cm. **Fix: mount
  the tag so the camera sees it OBLIQUELY** (forward of the camera's nadir) for one clean
  solution; and/or take a robust median-cluster extrinsic over many frames.
- **tag→base rotation:** the tag-frame → base-frame rotation is fiddly (a 90°/180° error is easy
  and silently wrong). Verify by overlaying the projected base axes on the live image.
- **Off-centerline accuracy:** grasps degrade away from the centerline if the base isn't level;
  keep the base level / demo near y≈0 if accuracy slips.
- **Reach:** the arm tilts to reach past the top-down zone — precise top-down grasping is only
  accurate within the inner reach band; beyond that the tip tilts and accuracy loosens (fine for
  pushing/coarse moves).

---

# Part B — Vision: Object Detection (the world model)
*Owner edits this section. Exposes: detection (§0.3 #3).*

## B.1 What it does
Given a camera frame and an object name/prompt, return **where each object is in the image** —
its label and **pixel location (the bounding-box center)** — per the §0.3 #3 seam. That pixel is
the **"click"** the arm side turns into a table coordinate, so vision is a **drop-in
replacement for a human click**: produce the same `(label, (u, v))` and everything downstream
just works. It does **not** do the pixel→table math (that's Part A) and does **not** touch arm
code; it stays a standalone module behind the interface until integration.

## B.2 Stack
- **Open-vocabulary detector** (e.g. **YOLO-World** via `ultralytics` + `torch`) so it can be
  asked for arbitrary object names without retraining.
- Runs on the **x64 / Mac / Linux** machine (PyTorch requirement — see §0.4).

## B.3 Approach
- Expose `detect(frame, prompt) -> [(label, (u, v)), ...]` — label + bounding-box-center pixel per
  object, matching §0.3 #3 exactly so it drops straight into the arm's click→coordinate path.
- **Validate** that detection is reliable on the **real overhead feed** under the demo's lighting
  and objects — accuracy here directly drives grasp success, so measure it on the rig.

## B.4 Tune live
- Detection confidence threshold, prompt phrasing (object names), and robustness on the actual
  demo objects under the real lighting/camera. These are tuned at the rig, not assumed.

---

# Part C — The Brain: LLM Orchestrator
*Owner edits this section. Consumes: the primitives (§0.3 #2). Owns: the `detect_objects` composition (§0.3 #4).*

## C.1 The loop
A standard Claude **Messages-API tool-use loop**: send the user command + tool definitions →
Claude thinks and emits tool calls → execute them → return results → repeat until the task is
done. Zero hardware risk in the loop logic itself.

## C.2 Tool surface — KEY decision
The tools are **only** the small composable set: the §0.3 motion primitives + `detect_objects`.
**Do NOT build `pick`/`place`/`push`/`stack`/`throw` as tools.** The Brain **composes** those
verbs from the low-level primitives, guided by the physical rules in its system prompt. A
tool-per-verb defeats the general-purpose/zero-shot goal and can't scale to un-anticipated
commands. *(Optional, not architectural: one tested `pick`/`place` convenience tool as a
reliability hedge for the highest-stakes action — the grasp.)*

## C.3 Dispatch (and the perception composition)
Glue that maps each tool call to the **real Python** (Part A's primitives). Keep our code in the
middle so the **safety clamps still apply** to anything Claude asks for. **This layer also owns
the `detect_objects` composition (§0.3 #4):** it calls Part B's detector to get the object pixel +
label, then Part A's localization to convert that pixel to a table coordinate, and returns it to
Claude. Part B and Part A each own one half; Part C chains them.

## C.4 System prompt
Carries: the table coordinate frame (§0.3 #1), the **physical rules (§0.6)** as hardware guidance
(grasp offset + descend-into, lift-before-retract, gripper-in-isolation, speed-by-task, the
reach/pitch envelope, soft-limits-are-overridable), and a couple of **few-shot examples** of
composing a verb (e.g. how a "stack" decomposes into detect → move above → descend → grasp →
lift → move → place). This is where most of the "reasoning quality" lives.

## C.5 Reasoning display (the demo website)
A **website** is the demo surface: type a prompt → watch Claude **think**, see **which tools it's
calling** and with what arguments, and watch the task get accomplished live (ideally with a
detection overlay / camera view). This is what demonstrates *general-purpose AI reasoning* (not a
scripted arm) — the highest-value piece of the presentation.

## C.6 Build & tune notes
- Build against **mock/printing primitives + a mock `detect_objects`** first (no hardware needed)
  — this lets the Brain be developed fully in parallel with the rig, then swapped to real
  dispatch at integration.
- Iterate the **system prompt + few-shots** for verb-composition quality once on the real arm.
