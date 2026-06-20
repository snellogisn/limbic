"""Stage 3: re-measure a camera's EXTRINSICS from its AprilTag (§A.5 / §A.7).

When a camera is physically moved, its intrinsics still hold but its pose in the
base/table frame changes. This re-derives that pose: detect the camera's 36h11
tag, solvePnP the tag->camera transform, compose with the tag's KNOWN base pose
to get camera->base, and save extrinsics_CAM_<role>.npz.

Robustness (the §A.7 pitfalls):
  * Pose flip: a square tag has two solvePnP solutions. We compute both
    (IPPE_SQUARE) and keep the one with the camera ABOVE the table
    (t_cam2base z > 0) and the lower reprojection error.
  * tag->base rotation is fiddly and a 90/180 error is SILENT. Because the tag
    is square, a wrong --tag-yaw still fits the corners perfectly (it just
    relabels them) — so the corner residual CANNOT catch a yaw error; it only
    catches gross errors (wrong intrinsics, mis-detection, wrong tag size). The
    yaw is disambiguated two ways the human must eyeball:
      1. the recovered CAMERA CENTRE in base mm — must match where the camera
         physically sits (a 90 deg yaw error puts it somewhere obviously wrong);
      2. a saved OVERLAY PNG with the base axes (origin + X/Y/Z arrows) drawn on
         the frame — +x must point forward (away from the base), +y to the left.
    Try --tag-yaw in {0, 90, 180, 270} until BOTH look right, then --go to save.

The tag is assumed flat on the table (local +z up). --tag-yaw is the rotation of
the tag's printed +x about vertical relative to base +x.

Usage (intrinsics_CAM_<role>.npz must already exist in --calib-dir):
    python scripts/stage3_extrinsics.py A --calib-dir PATH            # dry run
    python scripts/stage3_extrinsics.py A --calib-dir PATH --tag-yaw 90
    python scripts/stage3_extrinsics.py A --calib-dir PATH --go       # save .npz

Safety: no arm motion. BARREL-JACK power is irrelevant here (camera only), but
keep the tag fully visible and lit, viewed OBLIQUELY (not straight down).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from limbic.control import calibration
from limbic.control.localization import CameraExtrinsics, CameraIntrinsics, pixel_to_table


def _aruto_detector():
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # API moved between OpenCV versions; support both.
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, params), None
    return None, dictionary


def detect_tag(gray, want_id: int):
    """Return the 4 image corners (TL,TR,BR,BL) for ``want_id``, or None."""
    import cv2

    detector, dictionary = _aruto_detector()
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if int(i) == want_id:
            return c.reshape(4, 2).astype(np.float64)
    return None


def tag_local_corners(size_mm: float) -> np.ndarray:
    """The 4 tag corners in the tag's own frame (center origin, +x right, +y up,
    z=0), in the aruco corner order TL, TR, BR, BL."""
    h = size_mm / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def tag_to_base(tag_xyz_mm, tag_yaw_deg: float):
    """Known tag pose in base: flat on the table, rotated tag_yaw about +z."""
    yaw = np.radians(tag_yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    R_t2b = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    t_t2b = np.asarray(tag_xyz_mm, dtype=np.float64)
    return R_t2b, t_t2b


def solve_extrinsics(img_corners, intr, tag_xyz_mm, size_mm, tag_yaw_deg):
    """tag->camera (solvePnP, flip-resistant) composed with known tag->base."""
    import cv2

    obj = tag_local_corners(size_mm)
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, img_corners, intr.camera_matrix, intr.dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok or len(rvecs) == 0:
        return None
    R_t2b, t_t2b = tag_to_base(tag_xyz_mm, tag_yaw_deg)

    best = None
    for rvec, tvec, err in zip(rvecs, tvecs, errs.flatten()):
        R_t2c, _ = cv2.Rodrigues(rvec)
        R_c2b = R_t2b @ R_t2c.T
        t_c2b = (-R_t2b @ R_t2c.T @ tvec.reshape(3)) + t_t2b
        if t_c2b[2] <= 0:  # camera must be above the table
            continue
        extr = CameraExtrinsics(R_cam2base=R_c2b, t_cam2base=t_c2b)
        if best is None or err < best[0]:
            best = (float(err), extr)
    return best[1] if best else None


def self_check(img_corners, intr, extr, tag_xyz_mm, size_mm, tag_yaw_deg) -> float:
    """Map each detected corner pixel -> table, compare to its known base position.
    Returns the worst residual (mm). A wrong tag-yaw makes this large."""
    R_t2b, t_t2b = tag_to_base(tag_xyz_mm, tag_yaw_deg)
    known_base = (tag_local_corners(size_mm) @ R_t2b.T) + t_t2b  # (4,3)
    worst = 0.0
    for (u, v), kb in zip(img_corners, known_base):
        x, y = pixel_to_table(u, v, intr, extr, table_z_mm=float(kb[2]))
        worst = max(worst, float(np.hypot(x - kb[0], y - kb[1])))
    return worst


def capture_frame(cam_name: str, frames: int):
    """Grab ``frames`` frames, return (last good colour BGR, grayscale, (w,h))."""
    import cv2

    from limbic.platform_support import open_camera

    cap = open_camera(cam_name, width=1280, height=720)
    last = None
    wh = (0, 0)
    try:
        for _ in range(frames):
            ok, f = cap.read()
            if ok and f is not None:
                last = f
                wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    if last is None:
        return None, None, wh
    return last, cv2.cvtColor(last, cv2.COLOR_BGR2GRAY), wh


def save_axes_overlay(color_frame, intr, extr, tag_xyz_mm, out_path) -> None:
    """Draw the base-frame origin + X/Y/Z axis arrows on the frame and save it.

    THIS is the §A.7 yaw check: +x must point forward (away from the base), +y
    left, with the origin under the shoulder-pan axis. Also marks the tag centre.
    """
    import cv2

    R_b2c = extr.R_cam2base.T
    rvec, _ = cv2.Rodrigues(R_b2c)
    tvec = (-R_b2c @ extr.t_cam2base).reshape(3, 1)

    L = 100.0  # axis arrow length, mm
    pts = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L],
                    list(tag_xyz_mm)], dtype=np.float64)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, intr.camera_matrix, intr.dist_coeffs)
    proj = proj.reshape(-1, 2).astype(int)
    o, px, py, pz, ptag = (tuple(p) for p in proj)

    img = color_frame.copy()
    cv2.arrowedLine(img, o, px, (0, 0, 255), 3, tipLength=0.2)   # +x red
    cv2.arrowedLine(img, o, py, (0, 255, 0), 3, tipLength=0.2)   # +y green
    cv2.arrowedLine(img, o, pz, (255, 0, 0), 3, tipLength=0.2)   # +z blue
    cv2.circle(img, o, 5, (255, 255, 255), -1)
    cv2.circle(img, ptag, 6, (0, 255, 255), 2)                   # tag centre
    cv2.putText(img, "+x", px, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "+y", py, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), img)


def run_role(role: str, calib_dir: pathlib.Path, tag_yaw_deg: float, frames: int, go: bool) -> bool:
    cam = calibration.CAMERAS[role]
    name, tag_id, tag_xyz = cam["name"], cam["tag_id"], cam["tag_xyz_mm"]
    size = calibration.APRILTAG_SIZE_MM
    intr_path = calib_dir / f"intrinsics_CAM_{role}.npz"
    print(f"\n--- CAM_{role} ({cam['side']}, {name}) | tag id {tag_id} @ {tag_xyz} mm ---")
    if not intr_path.exists():
        print(f"  intrinsics not found: {intr_path}  (skipping)")
        return False
    intr = CameraIntrinsics.load(intr_path)

    color, gray, wh = capture_frame(name, frames)
    if gray is None:
        print(f"  no frame from {name!r} (camera busy / unplugged?)")
        return False
    corners = detect_tag(gray, tag_id)
    if corners is None:
        print(f"  tag id {tag_id} NOT seen in {wh[0]}x{wh[1]} frame "
              "(check it's visible, lit, and in view).")
        return False

    extr = solve_extrinsics(corners, intr, tag_xyz, size, tag_yaw_deg)
    if extr is None:
        print("  solvePnP failed / no camera-above-table solution.")
        return False

    resid = self_check(corners, intr, extr, tag_xyz, size, tag_yaw_deg)
    cam_pos = extr.t_cam2base
    consistent = resid < 10.0  # gross-error gate ONLY (cannot catch a yaw error)

    overlay = calib_dir / f"extrinsics_CAM_{role}_overlay.png"
    save_axes_overlay(color, intr, extr, tag_xyz, overlay)

    print(f"  corner consistency residual (tag-yaw={tag_yaw_deg:g}): {resid:.1f} mm "
          f"({'ok' if consistent else 'TOO LARGE — gross error, fix before trusting'})")
    print(f"  >> recovered CAMERA CENTRE in base: "
          f"({cam_pos[0]:.0f}, {cam_pos[1]:.0f}, {cam_pos[2]:.0f}) mm "
          f"-- does this match where {cam['side']} camera physically sits?")
    print(f"  >> EYEBALL the axes overlay: {overlay}")
    print(f"     +x (red) must point FORWARD (away from base), +y (green) LEFT. "
          f"If not, re-run with a different --tag-yaw (0/90/180/270).")

    if go:
        if not consistent:
            print("  NOT saving (gross residual). Fix intrinsics/detection/tag-size first.")
            return False
        out = calib_dir / f"extrinsics_CAM_{role}.npz"
        extr.save(out, tag_yaw_deg=float(tag_yaw_deg), resid_mm=float(resid))
        print(f"  SAVED {out}  (only trust it if the overlay + camera centre look right)")
    else:
        print("  dry run — re-run with --go to save once the overlay looks right.")
    return consistent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roles", nargs="*", default=["A", "B"],
                    help="camera roles to process (default: A B)")
    ap.add_argument("--calib-dir", required=True,
                    help="dir holding intrinsics_CAM_<role>.npz (extrinsics saved here)")
    ap.add_argument("--tag-yaw", type=float, default=0.0,
                    help="tag +x rotation about vertical vs base +x (deg)")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--go", action="store_true", help="save the .npz (else dry run)")
    args = ap.parse_args()

    roles = args.roles or ["A", "B"]
    calib_dir = pathlib.Path(args.calib_dir)
    print("=" * 72)
    print("  Stage 3 extrinsics re-measure (AprilTag solvePnP -> camera->base)")
    print("=" * 72)
    print(f"calib dir: {calib_dir}")
    results = {r: run_role(r, calib_dir, args.tag_yaw, args.frames, args.go) for r in roles}
    print("\nsummary:", {r: ("ok" if v else "review") for r, v in results.items()})


if __name__ == "__main__":
    main()
