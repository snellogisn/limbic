# limbic

A **cross-platform** (macOS · Windows · Linux), **LLM-driven** control stack for a
tabletop robot arm. Speak an instruction; an LLM perceives the scene, compiles a
list of **motion primitives**, and the arm carries it out — and it all runs on a
plain laptop with **no physical arm** thanks to a built-in software mock.

It is a clean, cross-platform reimagining of a Windows-locked LeRobot SO-101
control stack. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design.

---

## Why it runs anywhere

The earlier code only ran on Windows: hard-coded `COM7` serial ports, Windows-only
DirectShow cameras, and IK solvers with no macOS binaries. limbic fixes all of
that:

- **Serial ports auto-detected** on every OS (`$LIMBIC_PORT` to override).
- **Cameras** open with the right backend per OS (AVFoundation / DirectShow / V4L2).
- **Kinematics are pure Python** — zero binary dependencies.
- **A mock backend** simulates the arm, so the entire pipeline runs on a bare
  machine with nothing plugged in.

---

## Install

The core stack needs **nothing** — it's pure Python and runs as-is. Install extras
only for the capabilities you want:

```bash
pip install -r requirements.txt          # serial + camera + LLM brain
# or pick à la carte:
pip install pyserial          # real-arm USB port detection
pip install opencv-python     # the camera sense
pip install anthropic         # the runtime LLM brain
pip install "lerobot[feetech]"  # drive the physical SO-101 arm
```

Requires Python 3.10+.

---

## Quick start

### Drive the arm directly (auto mock ⇄ real)

```python
from limbic import RobotArm

with RobotArm() as arm:          # real arm if one is attached, else the mock
    arm.go_home()
    arm.open_gripper()
    arm.move_to_xyz(180, 0, 60)  # table-frame mm: +x forward, +y left, +z up
    arm.close_gripper()
    arm.lift_by(80)
```

### Run a plan (a list of motion primitives)

```bash
python -m limbic.primitives.example_plan      # pick & place, on the mock arm
```

### Let an LLM compile and run the plan

```bash
export ANTHROPIC_API_KEY=sk-...
python examples/run_mock_demo.py              # perceive → plan → execute (mock)
```

`run_mock_demo.py` also runs **offline** (no API key) by executing a canned plan,
so you can always see the whole pipeline move the arm.

---

## Selecting hardware vs. mock

Everything is environment-driven — no code edits:

| Variable | Meaning | Example |
|---|---|---|
| `LIMBIC_BACKEND` | `auto` (default), `real`, or `mock` | `mock` |
| `LIMBIC_PORT` | serial port (else auto-detected) | `COM7` · `/dev/cu.usbserial-10` |
| `LIMBIC_ROBOT_ID` | robot id for the SDK | `limbic` |
| `ANTHROPIC_API_KEY` | needed only for the runtime brain | `sk-...` |

`auto` uses the real arm when a serial port is found **and** `lerobot` is
installed; otherwise it transparently falls back to the mock and tells you so.

> **Safety:** every motion — human, scripted, or LLM-issued — passes through the
> workspace clamp and per-joint soft limits in `limbic/control/safety.py` before
> any command reaches a motor. An out-of-reach target stops at the nearest
> reachable point; it is never sent raw.

---

## Layout

```
limbic/
  control/      The Body   — movement, gripper, guardrails, mock⇄real backend
  primitives/   The Skills — one motion primitive per file + the plan runner
  inputs/       The Senses — motor + camera readings the LLM can query
  brain/        The Mind   — instruction → validated list of primitives → run
  platform_support.py      — the cross-platform seam (ports, cameras, OS)
examples/
  run_mock_demo.py         — end-to-end demo on the mock arm
```

The motion primitives and sensory inputs are **auto-discovered**: add a capability
by dropping a single new file in `primitives/library/` or `inputs/library/` —
nothing else to wire up. This is also how the LLM invents or revises primitives.
