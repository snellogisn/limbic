# Handoff — Camera & Localization (Part A §A.5), for the x64 box

This is the pixel → table-frame localization half of Part A. The arm/IK half is
done and verified on hardware; this is what's left, and it runs on the **x64
machine** (where the integrated demo lives and where vision/PyTorch run).

## Goal
Turn a camera pixel `(u, v)` — a human click **or** a detection bounding-box
centre (they are the same input, §0.3 #3) — into a **table coordinate `(x, y)`**
in the *same frame the arm uses* (§0.3 #1: origin under the shoulder-pan axis on
the table surface, **+x forward, +y left, +z up, mm**).

**End state:** click a pixel → correct table coord, and `camera frame == arm
frame` confirmed against the real arm. After that, click → pick is just chaining
this into the (already working) motion primitives.

## What's already built in this repo
- **`limbic/control/localization.py`** — the pixel→table math. `CameraIntrinsics`
  / `CameraExtrinsics` loaders + `pixel_to_table()` (undistort → camera ray →
  rotate into base → intersect the `z=0` table plane). Pure geometry; validated
  to **0.000 mm** in a synthetic round-trip. `load_camera(role, calib_dir)` loads
  an intrinsics+extrinsics pair.
- **`limbic/control/calibration.py` §8** — the measured §8 constants and a
  `CAMERAS` registry so role ↔ camera name ↔ tag never drift:
  - role **`A`** = RIGHT = *Logitech Webcam C930e*, tag **id 0** at `(60, -145, 5)` mm
  - role **`B`** = LEFT  = *HD Pro Webcam C920*,    tag **id 1** at `(60, +145, 5)` mm
  - `camera_for_y(y)` / `camera_role_for_y(y)` — §8 selection rule (`y>0` → LEFT/B,
    `y≤0` → RIGHT/A); use that camera's reading entirely (don't average).
  - AprilTag family **36h11**, size **57.5 mm**. Capture at **1280×720**.
- **`scripts/stage3_extrinsics.py`** — re-derives a camera's pose in the base
  frame from its tag (cv2.aruco + solvePnP). Math validated synthetically
  (recovers a known oblique camera centre to 0.000 mm).

Cameras are confirmed to open **by name** at 1280×720 on the arm machine. Resolve
by name everywhere — indices shuffle between machines.

## What's left to do (in order)
1. **Transfer the intrinsics `.npz`** into a calibration dir, named
   `intrinsics_CAM_A.npz` and `intrinsics_CAM_B.npz`. Intrinsics are the camera's
   *optics* — unaffected by where it's mounted — so they transfer between rigs as-is.
   (The loader is tolerant of key spellings: `camera_matrix`/`mtx`/`K`,
   `dist_coeffs`/`dist`/`D`.)
2. **Re-run extrinsics for BOTH cameras** — the rig was physically moved, so the
   old extrinsics are stale (intrinsics still hold):
   ```
   python scripts/stage3_extrinsics.py A B --calib-dir <DIR>            # dry run
   python scripts/stage3_extrinsics.py A B --calib-dir <DIR> --tag-yaw 0
   # sweep --tag-yaw 0 / 90 / 180 / 270 until the overlay + camera centre look right
   python scripts/stage3_extrinsics.py A B --calib-dir <DIR> --go       # save .npz
   ```
   This writes `extrinsics_CAM_{A,B}.npz` and an `_overlay.png` per camera.
3. **Wire pixel→table**: `intr, extr = localization.load_camera("A", dir)` then
   `localization.pixel_to_table(u, v, intr, extr)`.
4. **Verify `camera frame == arm frame`** (the Stage-3 milestone): put an object
   at a *known* table point, click it, confirm the returned `(x, y)` matches the
   ruler; ideally have the arm reach it and check it lands on the object.

## Read these before trusting an extrinsic (§A.7 traps)
- **Pose flip:** a tag viewed near-straight-down has two near-equal solvePnP
  solutions that flip frame-to-frame (camera position jumps by cm). **Mount/aim so
  each camera sees its tag OBLIQUELY**, not at nadir. The script already keeps the
  *camera-above-table* solution with the lowest reprojection error.
- **tag→base 90°/180° is SILENT.** Because the tag is square, a wrong `--tag-yaw`
  still fits the corners perfectly (it just relabels them) — so the script's
  corner-residual is a *gross-error gate only* and **cannot** catch a yaw error.
  Disambiguate it two ways, both human:
  1. the printed **recovered camera centre** (base mm) must match where the camera
     physically sits — a 90° error puts it somewhere obviously wrong;
  2. the saved **`*_overlay.png`** — the red **+x** arrow must point **forward**
     (away from the base), green **+y** to the **left**.
  Only `--go`-save once *both* look right.
- **Off-centerline / base level:** accuracy degrades away from `y≈0` if the base
  isn't level; keep demo objects near the centerline if it slips.

## OpenCV gotcha on the x64 box (likely bites you)
Both `opencv-python` and `opencv-python-headless` are installed (headless is a
**lerobot** dependency). On the arm machine the GUI build currently wins so
`cv2.imshow` works — but *which build wins depends on install order*, so on a
fresh x64 install the **headless build can win and silently break `cv2.imshow`**
(black/no window) for the click UI. Robust fix when you build the click UI:
```
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python          # GUI is a superset; lerobot's camera code still works
```
Or sidestep entirely with a **browser-based** click UI (matches the §C.5 demo
website) and leave headless in place.

## Environment knobs
- `LIMBIC_CAMERA` — override camera (name substring or index).
- `LIMBIC_PORT`, `LIMBIC_ROBOT_ID`, `LIMBIC_BACKEND` — arm side (only needed for
  the `camera frame == arm frame` cross-check).

## Don't touch
- The §0.3 shared interfaces (the team's API) — change only by agreement.
- The arm/IK internals (`kinematics.py`, `_prep_*`, `calibration.py` §1–§5) — those
  are measured/validated; localization only *reads* the §8 constants + frame.
