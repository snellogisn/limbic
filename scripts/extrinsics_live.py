"""LIVE extrinsics viewer/tuner for both rig cameras (§A.5 / §A.7).

Opens CAM_A and CAM_B at once, detects each camera's AprilTag every frame,
solvePnP -> camera->base, and draws the recovered BASE-FRAME AXES live on each
feed so you can eyeball the §A.7 yaw check in real time instead of squinting at a
saved PNG:
    red  +x  must point FORWARD (away from the base)
    green +y must point LEFT
    blue +z  must point UP (out of the table)

Because the tag is square, the corner residual CANNOT catch a 90/180 yaw error
(it fits any relabeling) — so you disambiguate by EYE here: rotate each camera's
assumed tag-yaw with a hotkey until the axes point the right way, then save.

Hotkeys (focus a camera window first):
    a : rotate CAM_A tag-yaw by +90 deg
    b : rotate CAM_B tag-yaw by +90 deg
    s : SAVE both extrinsics (extrinsics_CAM_{A,B}.npz) at the current yaws
    r : reload (re-print) the recovered camera centres to the console
    ESC / q : quit (does NOT auto-save)

Usage (intrinsics_CAM_{A,B}.npz must exist in --calib-dir):
    python scripts/extrinsics_live.py --calib-dir calib
    python scripts/extrinsics_live.py --calib-dir calib --yaw-a 90 --yaw-b 90

Safety: cameras only, no arm motion.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent))   # repo root (for limbic.*)
sys.path.insert(0, str(_SCRIPTS))          # scripts dir (to reuse stage3_extrinsics)

import stage3_extrinsics as ext  # noqa: E402  reuse detect_tag / solve_extrinsics / self_check
from limbic.control import calibration  # noqa: E402
from limbic.control.localization import CameraIntrinsics  # noqa: E402


def draw_axes(img, intr, extr, tag_xyz_mm, axis_len_mm: float = 100.0) -> None:
    """Project the base-frame origin + X/Y/Z axes onto the live frame."""
    import cv2

    R_b2c = extr.R_cam2base.T
    rvec, _ = cv2.Rodrigues(R_b2c)
    tvec = (-R_b2c @ extr.t_cam2base).reshape(3, 1)
    L = axis_len_mm
    pts = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], dtype=np.float64)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, intr.camera_matrix, intr.dist_coeffs)
    proj = proj.reshape(-1, 2).astype(int)
    o, px, py, pz = (tuple(p) for p in proj)
    cv2.arrowedLine(img, o, px, (0, 0, 255), 3, tipLength=0.2)   # +x red
    cv2.arrowedLine(img, o, py, (0, 255, 0), 3, tipLength=0.2)   # +y green
    cv2.arrowedLine(img, o, pz, (255, 0, 0), 3, tipLength=0.2)   # +z blue
    cv2.circle(img, o, 5, (255, 255, 255), -1)
    cv2.putText(img, "+x fwd", px, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "+y left", py, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_tag(img, corners) -> None:
    import cv2

    pts = corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [pts], True, (0, 255, 255), 2)


def banner(img, lines) -> None:
    import cv2

    y = 30
    for text, color in lines:
        cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 32


def process(role, cap, intr, yaw, size):
    """Grab a frame, detect the tag, solve at ``yaw``, draw the overlay.

    Returns (display_frame, extr_or_None, resid_or_None). Never raises on a
    missing tag / failed solve — it just annotates the frame.
    """
    import cv2

    cam = calibration.CAMERAS[role]
    tag_id, tag_xyz = cam["tag_id"], cam["tag_xyz_mm"]
    ok, frame = cap.read()
    if not ok or frame is None:
        blank = np.zeros((720, 1280, 3), np.uint8)
        banner(blank, [(f"CAM_{role} ({cam['side']}): no frame", (0, 0, 255))])
        return blank, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = ext.detect_tag(gray, tag_id)
    lines = [(f"CAM_{role} {cam['side']} {cam['name']}", (255, 255, 0)),
             (f"tag id {tag_id} | tag-yaw {yaw:g}  (press '{role.lower()}' to rotate)",
              (255, 255, 255))]

    extr = resid = None
    if corners is None:
        lines.append((f"tag id {tag_id} NOT seen", (0, 0, 255)))
    else:
        draw_tag(frame, corners)
        extr = ext.solve_extrinsics(corners, intr, tag_xyz, size, yaw)
        if extr is None:
            lines.append(("solvePnP failed (no above-table soln)", (0, 0, 255)))
        else:
            draw_axes(frame, intr, extr, tag_xyz)
            resid = ext.self_check(corners, intr, extr, tag_xyz, size, yaw)
            c = extr.t_cam2base
            lines.append((f"cam centre (mm): ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})",
                          (0, 255, 0)))
            lines.append((f"corner resid {resid:.1f} mm (cannot catch yaw)",
                          (200, 200, 200)))
            lines.append(("check: red=+x FORWARD, green=+y LEFT", (0, 255, 255)))
    banner(frame, lines)
    return frame, extr, resid


def main() -> None:
    import cv2

    from limbic.platform_support import open_camera

    ap = argparse.ArgumentParser(description="Live extrinsics viewer/tuner for both cameras.")
    ap.add_argument("--calib-dir", required=True, help="dir with intrinsics_CAM_{A,B}.npz")
    ap.add_argument("--yaw-a", type=float, default=90.0, help="initial CAM_A tag-yaw (deg)")
    ap.add_argument("--yaw-b", type=float, default=90.0, help="initial CAM_B tag-yaw (deg)")
    ap.add_argument("--roles", nargs="*", default=["A", "B"])
    args = ap.parse_args()

    calib_dir = pathlib.Path(args.calib_dir)
    size = calibration.APRILTAG_SIZE_MM
    roles = args.roles
    yaws = {"A": args.yaw_a, "B": args.yaw_b}

    intr = {}
    caps = {}
    for r in roles:
        ip = calib_dir / f"intrinsics_CAM_{r}.npz"
        if not ip.exists():
            print(f"missing {ip} — run stage3_intrinsics first.")
            return
        intr[r] = CameraIntrinsics.load(ip)
        caps[r] = open_camera(calibration.CAMERAS[r]["name"], width=1280, height=720)
        cv2.namedWindow(f"CAM_{r}", cv2.WINDOW_NORMAL)

    print("Live extrinsics. Hotkeys: a/b rotate yaw, s save both, r print centres, ESC quit.")
    last = {r: (None, None) for r in roles}  # (extr, resid)
    try:
        while True:
            for r in roles:
                disp, extr, resid = process(r, caps[r], intr[r], yaws[r], size)
                last[r] = (extr, resid)
                cv2.imshow(f"CAM_{r}", disp)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                print("quit (not saved).")
                break
            elif k == ord("a") and "A" in roles:
                yaws["A"] = (yaws["A"] + 90) % 360
                print(f"CAM_A tag-yaw -> {yaws['A']:g}")
            elif k == ord("b") and "B" in roles:
                yaws["B"] = (yaws["B"] + 90) % 360
                print(f"CAM_B tag-yaw -> {yaws['B']:g}")
            elif k == ord("r"):
                for r in roles:
                    e, _ = last[r]
                    if e is not None:
                        c = e.t_cam2base
                        print(f"CAM_{r} yaw {yaws[r]:g}: centre ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}) mm")
            elif k == ord("s"):
                for r in roles:
                    e, resid = last[r]
                    if e is None:
                        print(f"CAM_{r}: nothing to save (no solve).")
                        continue
                    out = calib_dir / f"extrinsics_CAM_{r}.npz"
                    e.save(out, tag_yaw_deg=float(yaws[r]),
                           resid_mm=float(resid) if resid is not None else -1.0)
                    print(f"SAVED {out}  (yaw {yaws[r]:g})")
    finally:
        for c in caps.values():
            c.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
