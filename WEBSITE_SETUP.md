# Website / Demo-Box Setup — pip installs

Everything you need to `pip install` to run the limbic **web interface** on the
**x64 demo computer** with the **mink (MuJoCo) IK solver**, the **Claude brain**,
and the **real SO-101 arm**.

> ⚠️ x64 only. `mujoco`/`mink` and `torch` have **no ARM64-Windows wheels**, so this
> setup does **not** run on the Snapdragon/ARM64 dev box — that machine falls back
> to the planar IK solver. Run the website on the x64 box.

> Use **Python 3.13 (64-bit)** — the verified `mujoco`/`mink` wheels are `cp313 win_amd64`.

---

## 1. Create + activate a venv

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. The pip installs

```powershell
# --- Inverse kinematics ----------------------------------------------------
# numpy + ikpy: the FK chain (always imported, both IK engines build off it)
pip install numpy ikpy

# mink (MuJoCo) reaching IK — the NEW default solver the website uses
pip install mujoco mink
# mink solves the QP with the "daqp" backend; install it explicitly
pip install qpsolvers daqp

# --- The Claude brain (natural-language planning) --------------------------
pip install "anthropic>=0.40"

# --- The real arm ----------------------------------------------------------
pip install "pyserial>=3.5"          # COM-port detection
pip install "lerobot[feetech]"        # drives the physical SO-101 (Feetech servos)
```

### One-liner

```powershell
pip install numpy ikpy mujoco mink qpsolvers daqp "anthropic>=0.40" "pyserial>=3.5" "lerobot[feetech]"
```

---

## 3. (Optional) Camera vision — only for `detect_objects`

Needed **only** if the demo uses live object detection (Part B) instead of typed
`(x, y)` coordinates. Heavy; skip it if you're driving by coordinates.

```powershell
pip install "opencv-python>=4.8" pygrabber        # capture + by-name camera enum (Windows)
pip install "torch>=2.0" transformers              # Grounding DINO detector
```

---

## 4. Run it

```powershell
$env:ANTHROPIC_API_KEY = 'sk-ant-...'   # inline ONLY — never write the key to a file
$env:LIMBIC_BACKEND = 'real'            # drive the physical arm
$env:LIMBIC_PORT = 'COM5'               # the COM port on THIS machine (Device Manager > Ports)
python web\server.py                     # then open http://localhost:8765
```

`LIMBIC_IK` already defaults to `mink`, so the website uses the new solver with no
extra flag.

## 5. Confirm it's actually using mink (not the fallback)

Watch the server startup log:

- ✅ Planner line says **`planner: CLAUDE`** (key was picked up).
- ✅ **No** warning like `mink IK unavailable (...); falling back to the closed-form planar solver`.

If you see that fallback warning, `mujoco`/`mink`/`daqp` didn't import — re-check
step 2 in the **same venv** you're launching `web/server.py` from.

---

## Why each one

| Package | Why it's needed |
|---|---|
| `numpy` | array math under the IK chain + mink |
| `ikpy` | builds the SO-101 FK chain from the URDF (imported on every run) |
| `mujoco` | physics/kinematics engine mink runs on |
| `mink` | the differential-IK reaching solver (the new default) |
| `qpsolvers`, `daqp` | the QP backend mink calls each iteration (`"daqp"`) |
| `anthropic` | the Claude brain that plans natural-language tasks |
| `pyserial` | finds/opens the arm's USB serial (COM) port |
| `lerobot[feetech]` | the driver for the physical SO-101 (Feetech STS servos) |
| `opencv-python`, `pygrabber` | *(optional)* camera capture + Windows by-name enumeration |
| `torch`, `transformers` | *(optional)* Grounding DINO object detection |
