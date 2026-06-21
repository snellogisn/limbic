# limbic — Architecture

`limbic` is a control stack for a tabletop robot arm that a person drives in
plain language: you give an instruction, an LLM perceives the scene and compiles
a list of **motion primitives**, and the arm executes them. It is built to run on
**macOS, Windows and Linux**, with or without physical hardware.

## The three layers (plus the mind)

```
   instruction ("pick up the block and put it on the left")
        │
        ▼
 ┌──────────────┐   browses catalogs, queries senses, emits an ordered plan
 │   THE MIND   │   limbic/brain/     (Claude API: perceive → plan → run)
 └──────┬───────┘
        │ list[ {primitive, args} ]
        ▼
 ┌──────────────┐   reusable arm skills, one file each, LLM-authorable
 │  THE SKILLS  │   limbic/primitives/   (home, move_to, pick, place, push, …)
 └──────┬───────┘
        │ RobotArm method calls (already safety-clamped + smoothed)
        ▼
 ┌──────────────┐   movement, gripper, GUARDRAILS — the only thing that
 │   THE BODY   │   limbic/control/     touches motors; auto mock⇄real backend
 └──────┬───────┘
        │ joint commands (degrees / 0..100 gripper)
        ▼
   MockBackend (software sim)   or   RealBackend (LeRobot SO-101 over USB)

 ┌──────────────┐   read-only perceptions the mind can query while planning
 │  THE SENSES  │   limbic/inputs/    (joint_state, tip_position, camera, …)
 └──────────────┘
```

### The Body — `limbic/control/`
The only layer that commands motors. Everything else goes through its `RobotArm`
class, so the safety and smoothing logic is written once and shared.

| File | Role |
|---|---|
| `arm.py` | `RobotArm` — the stable tool surface: `move_to_xyz`, `reach_above`, `descend_to`, `lift_by`, `set_joint`, `go_home`, `open_gripper`/`close_gripper`/`set_gripper`, `current_xyz`, `read_joints`. Every move is workspace-clamped, soft-limit-clamped, and streamed with an ease-in/ease-out velocity profile. |
| `safety.py` | The **single source of truth for guardrails**: per-joint soft limits + the Cartesian workspace dome. Targets outside the safe region are clamped to the nearest reachable point, never sent raw. |
| `kinematics.py` | Pure-Python closed-form IK/FK (table-mm ⇄ joint-degrees). No numpy/ikpy/placo, so it runs identically on every OS — this is what unblocks macOS, where the reference project's solvers had no binaries. |
| `backends.py` | The hardware seam: `HardwareBackend` interface, `MockBackend` (software sim), `RealBackend` (LeRobot SO-101). `make_backend()` auto-picks: real if a serial port is found and `lerobot` is installed, else mock. |
| `config.py` | Env-driven connection + motion config (`$LIMBIC_PORT`, `$LIMBIC_BACKEND`, …). |

### The Skills — `limbic/primitives/`
A folder of motion primitives, **one per file**, each a `Primitive` subclass
declaring `name`, `summary`, `parameters`, and a `run(arm, **kwargs)` that calls
only the `RobotArm` tool surface (so it inherits all safety). The `registry`
auto-discovers every file, so the catalog the LLM browses is always in sync with
what's on disk. `run_sequence.py` executes a `list[{primitive, args}]` in one go.

### The Senses — `limbic/inputs/`
Read-only perceptions, **one per file**, each an `Input` subclass with the same
self-describing shape. The `registry` auto-discovers them and injects runtime
context (the live arm) so motor senses work without the LLM having to wire them.
Cameras use the cross-platform `open_camera` helper (not Windows DirectShow).

### The Mind — `limbic/brain/`
Accomplishes an instruction as a **plan → execute → verify → retry cycle**:

1. **Plan** — Claude perceives via `sense_*` tools if needed, may **create or edit
   a motion primitive** (`create_primitive` / `edit_primitive`) when no existing
   skill fits, then commits one ordered list of steps (`submit_plan`). Every step
   is **validated against the registry** before anything moves.
2. **Execute** — the list runs through the sequence runner and `RobotArm`, so the
   safety layer governs every motor command (our code drives the arm, not Claude).
3. **Verify** — a pluggable check (`verifier=`) decides *satisfied* vs *incomplete*
   from a fresh `inputs.snapshot()` + the execution results.
4. **Retry** — if incomplete, the reason + snapshot are fed back and the cycle
   repeats (revise the plan, or author a better primitive), up to `max_attempts`.

It routes models by difficulty (quick model for short/urgent, large model for
complex spatial reasoning) and accepts an injected `client`/`verifier` so the whole
cycle is testable offline without an API key.

**Live-input seam:** `inputs.snapshot()` reads *every* registered sense, so when a
streaming object detector (e.g. YOLO: what/where/size) is later dropped into
`inputs/library/`, it automatically joins every snapshot and makes verification
detection-grounded — no change to the brain. That detector + the `detect_objects`
composition are intentionally integrated separately.

## How the LLM composes — and evolves — primitives

The core idea: the LLM doesn't get one tool per verb. It gets a small set of
composable primitives and the catalogs that describe them, and it **composes**
pick/place/stack/arrange itself. Because primitives are one-file, uniform, and
auto-discovered:

- **Pick & choose** — the LLM reads `registry.catalog()` and selects primitives.
- **Invent** — it can author a new primitive by writing one new file in
  `primitives/library/` in the same mould; `registry.reload()` picks it up.
- **Revise/retire** — if a better primitive supersedes an old one, the LLM can
  edit or delete that single file without touching anything else.

A standard, stable set ships by default; treat it as the baseline and only grow
it when a genuinely new capability is needed.

## Why it runs everywhere (the cross-platform refactor)

The reference (`bronny`/LeRobot) was Windows-locked in three ways, each fixed here:

| Windows-locked in the reference | Cross-platform in limbic |
|---|---|
| Hard-coded `COM7` serial port | `platform_support.detect_serial_port()` auto-finds the USB-serial bridge on any OS (`$LIMBIC_PORT` overrides) |
| `pygrabber.dshow_graph` + `cv2.CAP_DSHOW` (DirectShow) cameras | `platform_support.open_camera()` selects AVFoundation / DirectShow / V4L2 per OS |
| IK backends with no macOS/ARM wheels (placo, mujoco) | pure-Python `kinematics.py`, zero binary deps |
| PowerShell-only setup, no-hardware = no-run | env-var config + a `MockBackend` so the whole pipeline runs on a bare laptop |
