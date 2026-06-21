"""Stage 3: measure a camera's INTRINSICS from a checkerboard (§A.5 / §A.6).

Intrinsics are the camera's optics (focal length, principal point, lens
distortion) — independent of where the camera is mounted, so they only need
re-measuring per physical camera, not per rig move. This is the step the
extrinsics tool (``stage3_extrinsics.py``) assumes is already done: it writes
``intrinsics_CAM_<role>.npz`` in the same ``--calib-dir``.

CRITICAL: calibrate at the SAME resolution the rig captures at. §8 captures at
1280x720, so the default here is 1280x720 — a K measured at one resolution is
wrong at another (it scales with pixels).

How it works (auto-capture, §A.6 "~20 varied views covering the frame edges"):
    Hold the printed checkerboard up and sweep it SLOWLY through the frame. The
    script auto-grabs a view only when the board is (a) sharp (not motion-blurred)
    and (b) meaningfully different from views it already has — so you naturally
    collect varied distances/tilts/positions. It computes once it has --target
    good views (or ESC to stop early), then saves K + dist + RMS.

    Target RMS reprojection error: < 1.0 px (§A.6). > 1.5 px -> re-run.

Usage (one camera at a time — the board is camera-specific):
    python scripts/stage3_intrinsics.py A --calib-dir PATH
    python scripts/stage3_intrinsics.py B --calib-dir PATH --grid 9x6 --square-mm 25
    python scripts/stage3_intrinsics.py A --calib-dir PATH --headless   # no GUI window

If the live window is black (the headless OpenCV build won the install — see the
camera-localization handoff), re-run with --headless, or fix the cv2 build:
    pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python

Safety: no arm motion. Camera only.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from limbic.control import calibration


# Sub-pixel corner refinement + a robust board-finder flag set.
def _criteria():
    import cv2

    return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def _find_flags():
    import cv2

    return (cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK)


def object_points(grid: tuple[int, int], square_mm: float) -> np.ndarray:
    """3-D corners of a flat checkerboard (z=0), in mm. Square size only scales
    the (discarded) extrinsics — K/dist are unaffected — but keep it honest."""
    cols, rows = grid
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return objp


def board_signature(corners, w: int, h: int) -> np.ndarray:
    """Normalised (cx, cy, apparent-width, apparent-height) of the board in the
    frame — used to reject views too similar to ones already captured."""
    pts = corners.reshape(-1, 2)
    cx, cy = pts[:, 0].mean() / w, pts[:, 1].mean() / h
    bw = (pts[:, 0].max() - pts[:, 0].min()) / w   # apparent size ~ distance
    bh = (pts[:, 1].max() - pts[:, 1].min()) / h   # aspect ~ tilt
    return np.array([cx, cy, bw, bh])


def novelty(sig, signatures) -> float:
    if not signatures:
        return float("inf")
    return min(float(np.linalg.norm(sig - s)) for s in signatures)


def sharpness(gray, corners) -> float:
    """Laplacian variance over the board region — low = blurry."""
    import cv2

    pts = corners.reshape(-1, 2)
    x1, y1 = pts.min(axis=0).astype(int)
    x2, y2 = pts.max(axis=0).astype(int)
    x1, y1 = max(x1, 0), max(y1, 0)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def collect_views(cap, grid, objp, target, headless, sharp_min, novelty_min, cooldown):
    """Drive the camera until ``target`` good checkerboard views are collected.
    Returns (objpoints, imgpoints, (w, h)). ESC (GUI mode) stops early."""
    import cv2

    objpoints: list = []
    imgpoints: list = []
    signatures: list = []
    last_t = 0.0
    w = h = 0

    while len(imgpoints) < target:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("  camera read failed (busy / unplugged?) — stopping.")
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, grid, _find_flags())

        reason, color = "show the checkerboard", (0, 60, 220)
        if found:
            sig = board_signature(corners, w, h)
            nov = novelty(sig, signatures)
            sharp = sharpness(gray, corners)
            now = time.time()
            if sharp < sharp_min:
                reason, color = "hold steady (blurry)", (0, 140, 240)
            elif nov < novelty_min:
                reason, color = "move to a new angle/position", (0, 200, 240)
            elif (now - last_t) < cooldown:
                reason, color = "...", (0, 200, 0)
            else:
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _criteria())
                objpoints.append(objp)
                imgpoints.append(refined)
                signatures.append(sig)
                last_t = now
                reason, color = "CAPTURED", (0, 255, 0)
                print(f"  captured {len(imgpoints)}/{target} "
                      f"(sharpness {sharp:.0f}, novelty {nov:.2f})")

        if headless:
            continue
        try:
            disp = frame.copy()
            if found:
                cv2.drawChessboardCorners(disp, grid, corners, found)
            cv2.putText(disp, f"captures: {len(imgpoints)}/{target}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 200, 0), 2)
            cv2.putText(disp, reason, (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.imshow("Intrinsics calibration  |  ESC = stop", disp)
            if (cv2.waitKey(1) & 0xFF) == 27:
                print("  stopped early by user (ESC).")
                break
        except cv2.error:
            print("  (no GUI window available — switching to headless capture; "
                  "re-run with --headless to silence this.)")
            headless = True

    if not headless:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return objpoints, imgpoints, (w, h)


def run_role(role, calib_dir, grid, square_mm, width, height, target, headless,
             overwrite, sharp_min, novelty_min, cooldown) -> bool:
    import cv2

    from limbic.platform_support import open_camera

    cam = calibration.CAMERAS[role]
    name = cam["name"]
    out = calib_dir / f"intrinsics_CAM_{role}.npz"
    print(f"\n--- CAM_{role} ({cam['side']}, {name}) ---")
    print(f"  checkerboard: {grid[0]}x{grid[1]} inner corners, {square_mm:g} mm squares")
    print(f"  capturing at {width}x{height} (MUST match the rig capture resolution)")
    if out.exists() and not overwrite:
        print(f"  {out} already exists — pass --overwrite to replace it. (skipping)")
        return False

    try:
        cap = open_camera(name, width=width, height=height)
    except (ImportError, RuntimeError) as exc:
        print(f"  cannot open camera {name!r}: {exc}")
        return False

    try:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (width, height):
            print(f"  WARNING: driver negotiated {actual_w}x{actual_h}, not "
                  f"{width}x{height}. Intrinsics are resolution-specific — the "
                  f"extrinsics/localization MUST then also run at {actual_w}x{actual_h}.")
        objp = object_points(grid, square_mm)
        objpoints, imgpoints, (w, h) = collect_views(
            cap, grid, objp, target, headless, sharp_min, novelty_min, cooldown)
    finally:
        cap.release()

    n = len(imgpoints)
    if n < 5:
        print(f"  only {n} views — need >=5. Re-run and sweep the board more.")
        return False
    if n < target:
        print(f"  note: stopped with {n}/{target} views; accuracy may be reduced.")

    print(f"  computing calibration from {n} views ...")
    rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    print(f"  RMS reprojection error: {rms:.4f} px  "
          f"({'excellent' if rms < 0.5 else 'good' if rms < 1.0 else 'RE-RUN (>1.0)'})")
    print(f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f}  cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    np.savez(str(out), camera_matrix=K, dist_coeffs=dist, rms=float(rms),
             image_size=np.array([w, h]), grid=np.array(grid),
             square_mm=float(square_mm))
    print(f"  SAVED {out}")
    if rms >= 1.0:
        print("  ^ RMS is high — re-run with more varied views before trusting it.")
    return rms < 1.0


def _parse_grid(s: str) -> tuple[int, int]:
    cols, rows = (int(x) for x in s.lower().split("x"))
    return cols, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkerboard intrinsics for one rig camera.")
    ap.add_argument("role", choices=sorted(calibration.CAMERAS), help="camera role (A=RIGHT, B=LEFT)")
    ap.add_argument("--calib-dir", required=True, help="dir to save intrinsics_CAM_<role>.npz")
    ap.add_argument("--grid", type=_parse_grid, default=(9, 6),
                    help="inner corners as COLSxROWS (default 9x6)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="square size in mm (default 25)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--target", type=int, default=20, help="good views to collect (default 20)")
    ap.add_argument("--headless", action="store_true", help="no GUI window (auto-capture only)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing intrinsics file")
    ap.add_argument("--sharp-min", type=float, default=60.0)
    ap.add_argument("--novelty-min", type=float, default=0.12)
    ap.add_argument("--cooldown", type=float, default=0.6)
    args = ap.parse_args()

    calib_dir = pathlib.Path(args.calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("  Stage 3 intrinsics (checkerboard -> camera_matrix + dist_coeffs)")
    print("=" * 72)
    print(f"calib dir: {calib_dir}")
    ok = run_role(args.role, calib_dir, args.grid, args.square_mm, args.width,
                  args.height, args.target, args.headless, args.overwrite,
                  args.sharp_min, args.novelty_min, args.cooldown)
    print("\nsummary:", {args.role: "ok" if ok else "review"})


if __name__ == "__main__":
    main()
