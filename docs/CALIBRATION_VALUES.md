# Camera calibration values (CAM_A / CAM_B)

**Why this file exists:** `calib/` is in `.gitignore` (see the "Camera/calibration
artefacts" block), so the `.npz` files that hold the real calibration **never get
pushed to GitHub**. Anyone who clones the repo gets the code but *not* the numbers,
and the vision pipeline can't localize without them. This file is the tracked,
human-readable copy of those numbers, plus a script to rebuild the `.npz` from them.

These are **measured on THIS rig** (per CLAUDE.md §0.5 / §A.6). If a fresh live
calibration disagrees with a value here, the **live measurement wins** — re-run the
calibration scripts and update this file.

- Capture resolution: **1280 × 720** (boxes/pixels are only valid at this size).
- Units: **mm** for all table-frame / extrinsic quantities (CLAUDE.md §0.3 #1).
- Frame: origin under the shoulder-pan axis at the table surface; +x forward,
  +y left, +z up.

---

## Camera registry (from `limbic/control/calibration.py` §8)

| Role | Camera (device name)      | Side  | AprilTag id | Tag position (x, y, z) mm |
|------|---------------------------|-------|-------------|---------------------------|
| A    | `Logitech Webcam C930e`   | RIGHT | 0           | (60, −145, 5)             |
| B    | `HD Pro Webcam C920`      | LEFT  | 1           | (60, 145, 5)              |

- AprilTag family: **36h11**
- AprilTag size: **57.5 mm**
- Calibration checkerboard: **9 × 6** inner corners, **22.5 mm** squares.

The `.npz` filename suffix matches the role: `intrinsics_CAM_A.npz`,
`extrinsics_CAM_A.npz`, etc. Place them in `calib/` (the default `--calib-dir`).

---

## CAM_A — `Logitech Webcam C930e` (RIGHT)

### Intrinsics (`intrinsics_CAM_A.npz`)
RMS reprojection error: **0.891 px** · image_size: 1280×720 · grid 9×6 · square 22.5 mm

```
camera_matrix =
[[772.47605884,   0.0,          648.00428436],
 [  0.0,          769.02114301, 383.33832325],
 [  0.0,          0.0,            1.0        ]]

dist_coeffs = [0.11197489, -0.33401960, 0.00217911, -0.00089367, 0.22996597]
```

### Extrinsics (`extrinsics_CAM_A.npz`)
solvePnP residual: **0.300 mm** · inliers 150/150 frames · centre_std (4.85, 3.95, 0.92) mm · tag_yaw 90°

```
R_cam2base =
[[-0.00114728, -0.96709662,  0.25440677],
 [-0.99987497, -0.00290291, -0.01554413],
 [ 0.01577119, -0.25439280, -0.96697238]]

t_cam2base = [2.38059507, -106.54621747, 339.96464857]   # camera centre in base frame, mm
tag_yaw_deg = 90.0
```

---

## CAM_B — `HD Pro Webcam C920` (LEFT)

### Intrinsics (`intrinsics_CAM_B.npz`)
RMS reprojection error: **1.467 px** · image_size: 1280×720 · grid 9×6 · square 22.5 mm

```
camera_matrix =
[[954.72093962,   0.0,          627.96483124],
 [  0.0,          949.04428938, 394.79532316],
 [  0.0,          0.0,            1.0        ]]

dist_coeffs = [0.08277258, -0.29172100, 0.00214850, -0.00435495, 0.23720871]
```

### Extrinsics (`extrinsics_CAM_B.npz`)
solvePnP residual: **0.204 mm** · inliers 150/150 frames · centre_std (1.99, 1.05, 0.41) mm · tag_yaw 90°

```
R_cam2base =
[[ 0.01489072, -0.95634775,  0.29185142],
 [-0.99515961,  0.01418098,  0.09724328],
 [-0.09713713, -0.29188676, -0.95150749]]

t_cam2base = [3.85179001, 86.78350374, 344.95687294]   # camera centre in base frame, mm
tag_yaw_deg = 90.0
```

---

## Rebuild the `.npz` files from these values

`calib/` is gitignored, so after cloning, recreate it with this script (run from the
repo root). It writes exactly the four files the loaders expect.

```python
# scripts-free: paste into `python -` or save as a throwaway, run from repo root.
import os, numpy as np
os.makedirs("calib", exist_ok=True)

# ---- CAM_A (Logitech C930e, RIGHT) ----
np.savez("calib/intrinsics_CAM_A.npz",
    camera_matrix=np.array([[772.47605884, 0.0, 648.00428436],
                            [0.0, 769.02114301, 383.33832325],
                            [0.0, 0.0, 1.0]]),
    dist_coeffs=np.array([[0.11197489, -0.33401960, 0.00217911, -0.00089367, 0.22996597]]),
    rms=np.array(0.89126434), image_size=np.array([1280, 720]),
    grid=np.array([9, 6]), square_mm=np.array(22.5))

np.savez("calib/extrinsics_CAM_A.npz",
    R_cam2base=np.array([[-0.00114728, -0.96709662, 0.25440677],
                         [-0.99987497, -0.00290291, -0.01554413],
                         [ 0.01577119, -0.25439280, -0.96697238]]),
    t_cam2base=np.array([2.38059507, -106.54621747, 339.96464857]),
    tag_yaw_deg=np.array(90.0), resid_mm=np.array(0.29968607),
    inliers=np.array(150), frames_solved=np.array(150),
    centre_std_mm=np.array([4.85240204, 3.95487802, 0.91685343]))

# ---- CAM_B (HD Pro C920, LEFT) ----
np.savez("calib/intrinsics_CAM_B.npz",
    camera_matrix=np.array([[954.72093962, 0.0, 627.96483124],
                            [0.0, 949.04428938, 394.79532316],
                            [0.0, 0.0, 1.0]]),
    dist_coeffs=np.array([[0.08277258, -0.29172100, 0.00214850, -0.00435495, 0.23720871]]),
    rms=np.array(1.46704213), image_size=np.array([1280, 720]),
    grid=np.array([9, 6]), square_mm=np.array(22.5))

np.savez("calib/extrinsics_CAM_B.npz",
    R_cam2base=np.array([[ 0.01489072, -0.95634775, 0.29185142],
                         [-0.99515961,  0.01418098, 0.09724328],
                         [-0.09713713, -0.29188676, -0.95150749]]),
    t_cam2base=np.array([3.85179001, 86.78350374, 344.95687294]),
    tag_yaw_deg=np.array(90.0), resid_mm=np.array(0.20437323),
    inliers=np.array(150), frames_solved=np.array(150),
    centre_std_mm=np.array([1.98862467, 1.04852412, 0.40633490]))

print("wrote calib/{intrinsics,extrinsics}_CAM_{A,B}.npz")
```

---

## How to re-measure (if the rig moved)

- **Intrinsics** (optics; only if the camera/lens itself changed):
  `scripts/stage3_intrinsics.py` — checkerboard, ~20 varied images, target ≲1 px RMS.
- **Extrinsics** (pose; re-run whenever a camera is bumped/moved):
  `scripts/stage3_extrinsics.py` (or `_robust.py`) — solvePnP on the AprilTag.
  The dual viewer also **re-solves extrinsics live** every detection pass, so a
  small bump is tracked automatically; these saved values are the fallback.
- **Verify**: `scripts/click_localize.py` — click the same point in both feeds and
  confirm the two table coords agree (cross-check), and that base axes overlay
  correctly on the live image (CLAUDE.md §A.7).
