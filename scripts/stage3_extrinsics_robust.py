"""ROBUST multi-frame extrinsics measurement for the rig cameras (§A.5 / §A.7).

The single-frame solver (``stage3_extrinsics.py``) takes ONE frame's solvePnP.
That's vulnerable to the §A.7 pose-FLIP (a square tag has two near-equal
solutions that hop frame-to-frame, jumping the camera centre by cm) and to
per-frame detection noise. This tool instead:

  1. grabs MANY frames per camera,
  2. solves camera->base on each (keeping the camera-above-table solution),
  3. CLUSTERS the recovered camera centres and rejects outliers (the flips /
     bad detections fall outside the cluster),
  4. averages the surviving inliers — translation by mean, rotation by the
     quaternion eigen-mean (Markley) — for one steady pose,
  5. saves it, overwriting extrinsics_CAM_<role>.npz.

It reports the inlier count and the centre spread (std) so you can see how
stable the measurement actually was. tag-yaw defaults to 90 (the verified-good
orientation for this rig); the corner residual is still only a gross-error gate
(it cannot catch a yaw error — that was eyeballed already).

Usage (intrinsics_CAM_<role>.npz must exist in --calib-dir):
    python scripts/stage3_extrinsics_robust.py A B --calib-dir calib            # measure + save
    python scripts/stage3_extrinsics_robust.py A B --calib-dir calib --dry-run  # don't write
    python scripts/stage3_extrinsics_robust.py A B --calib-dir calib --frames 200 --tag-yaw 90

Safety: cameras only, no arm motion. Keep each tag visible, lit, and OBLIQUE.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent))
sys.path.insert(0, str(_SCRIPTS))

import stage3_extrinsics as ext  # noqa: E402  reuse detect_tag / solve_extrinsics / self_check / overlay
from limbic.control import calibration  # noqa: E402
from limbic.control.localization import CameraExtrinsics, CameraIntrinsics  # noqa: E402


# --------------------------------------------------------------------------- #
# Rotation averaging helpers (numpy-only; no scipy dependency)
# --------------------------------------------------------------------------- #
def rotmat_to_quat(R) -> np.ndarray:
    """3x3 rotation -> unit quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_rotmat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def average_quaternions(quats) -> np.ndarray:
    """Markley's quaternion average: eigenvector of sum(q q^T) for the largest
    eigenvalue. Sign-invariant, so antipodal duplicates don't bias it."""
    Q = np.asarray(quats, dtype=np.float64)
    M = Q.T @ Q
    vals, vecs = np.linalg.eigh(M)
    q = vecs[:, int(np.argmax(vals))]
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


# --------------------------------------------------------------------------- #
# Per-camera robust measurement
# --------------------------------------------------------------------------- #
def measure_role(role, calib_dir, tag_yaw, frames, reject_mm, min_inliers, dry_run) -> bool:
    import cv2

    from limbic.platform_support import open_camera

    cam = calibration.CAMERAS[role]
    name, tag_id, tag_xyz = cam["name"], cam["tag_id"], cam["tag_xyz_mm"]
    size = calibration.APRILTAG_SIZE_MM
    intr_path = calib_dir / f"intrinsics_CAM_{role}.npz"
    print(f"\n--- CAM_{role} ({cam['side']}, {name}) | tag id {tag_id} @ {tag_xyz} mm | yaw {tag_yaw:g} ---")
    if not intr_path.exists():
        print(f"  intrinsics not found: {intr_path}  (run stage3_intrinsics first)")
        return False
    intr = CameraIntrinsics.load(intr_path)

    try:
        cap = open_camera(name, width=1280, height=720)
    except (ImportError, RuntimeError) as exc:
        print(f"  cannot open camera {name!r}: {exc}")
        return False

    centres: list = []
    quats: list = []
    resids: list = []
    last_color = None
    last_corners = None
    seen = 0
    try:
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = ext.detect_tag(gray, tag_id)
            if corners is None:
                continue
            extr = ext.solve_extrinsics(corners, intr, tag_xyz, size, tag_yaw)
            if extr is None:
                continue
            seen += 1
            centres.append(np.asarray(extr.t_cam2base, dtype=np.float64))
            quats.append(rotmat_to_quat(extr.R_cam2base))
            resids.append(ext.self_check(corners, intr, extr, tag_xyz, size, tag_yaw))
            last_color, last_corners = frame, corners
    finally:
        cap.release()

    if seen < min_inliers:
        print(f"  only {seen} frames with a valid solve (need >={min_inliers}). "
              "Check the tag is visible, lit, and oblique. Not saving.")
        return False

    centres = np.array(centres)            # (seen, 3)
    quats = np.array(quats)                # (seen, 4)
    med = np.median(centres, axis=0)
    dist = np.linalg.norm(centres - med, axis=1)
    inl = dist <= reject_mm
    n_in = int(inl.sum())
    if n_in < min_inliers:
        # Relax: the cluster is looser than reject_mm (still report the spread).
        print(f"  WARNING: only {n_in}/{seen} frames within {reject_mm:g} mm of the median "
              f"centre — the pose is jittery (possible flips). Using all {seen} frames.")
        inl = np.ones(seen, dtype=bool)
        n_in = seen

    centre = centres[inl].mean(axis=0)
    spread = centres[inl].std(axis=0)
    R = quat_to_rotmat(average_quaternions(quats[inl]))
    extr = CameraExtrinsics(R_cam2base=R, t_cam2base=centre)

    # Gross-error gate (cannot catch yaw — that was eyeballed) + overlay for the record.
    resid = ext.self_check(last_corners, intr, extr, tag_xyz, size, tag_yaw)
    overlay = calib_dir / f"extrinsics_CAM_{role}_overlay.png"
    ext.save_axes_overlay(last_color, intr, extr, tag_xyz, overlay)

    print(f"  frames solved: {seen}/{frames} | inliers: {n_in} (<= {reject_mm:g} mm)")
    print(f"  camera centre: ({centre[0]:.1f}, {centre[1]:.1f}, {centre[2]:.1f}) mm "
          f"+/- ({spread[0]:.1f}, {spread[1]:.1f}, {spread[2]:.1f}) mm (inlier std)")
    print(f"  median per-frame corner residual: {np.median(resids):.2f} mm | "
          f"final-pose residual: {resid:.2f} mm")
    print(f"  overlay (eyeball axes: red +x FORWARD, green +y LEFT): {overlay}")

    if resid >= 10.0:
        print("  gross residual too large — NOT saving. Fix intrinsics/tag-size/detection.")
        return False
    if dry_run:
        print("  dry run — not saved. Drop --dry-run to overwrite the .npz.")
        return True
    out = calib_dir / f"extrinsics_CAM_{role}.npz"
    extr.save(out, tag_yaw_deg=float(tag_yaw), resid_mm=float(resid),
              inliers=int(n_in), frames_solved=int(seen),
              centre_std_mm=spread.astype(float))
    print(f"  SAVED {out}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Robust multi-frame extrinsics measurement.")
    ap.add_argument("roles", nargs="*", default=["A", "B"], help="camera roles (default A B)")
    ap.add_argument("--calib-dir", required=True, help="dir with intrinsics_CAM_<role>.npz")
    ap.add_argument("--tag-yaw", type=float, default=90.0, help="tag +x vs base +x (deg); verified 90 on this rig")
    ap.add_argument("--frames", type=int, default=150, help="frames to capture per camera")
    ap.add_argument("--reject-mm", type=float, default=25.0, help="centre-cluster outlier radius (mm)")
    ap.add_argument("--min-inliers", type=int, default=30, help="min good frames required to save")
    ap.add_argument("--dry-run", action="store_true", help="measure + report, but do not write the .npz")
    args = ap.parse_args()

    roles = args.roles or ["A", "B"]
    calib_dir = pathlib.Path(args.calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("  Stage 3 ROBUST extrinsics (multi-frame solvePnP -> clustered camera->base)")
    print("=" * 72)
    print(f"calib dir: {calib_dir} | frames/cam: {args.frames} | tag-yaw: {args.tag_yaw:g}")
    results = {r: measure_role(r, calib_dir, args.tag_yaw, args.frames, args.reject_mm,
                               args.min_inliers, args.dry_run) for r in roles}
    print("\nsummary:", {r: ("ok" if v else "review") for r, v in results.items()})


if __name__ == "__main__":
    main()
