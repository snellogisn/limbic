"""Click -> table-coordinate viewer for both rig cameras (§A.5 / §0.3 #3).

Shows CAM_B (LEFT) and CAM_A (RIGHT) side by side, live. Click any point in
either feed and it converts that pixel -> a table coordinate (x, y) in mm in the
base frame (§0.3 #1: origin under the shoulder-pan axis, +x forward, +y left)
using THAT camera's calibrated intrinsics+extrinsics (pixel_to_table).

This is the human-click stand-in for detection (§0.3 #3): the same pixel a
detector's bbox-centre would give. It also applies the §8 camera-selection rule
(object on the RIGHT -> CAM_A, on the LEFT -> CAM_B) and flags whether the camera
you clicked is the "closer" one for that point — when you later detect an object,
prefer the closer camera's reading instead of averaging.

Click the SAME object in both panels to compare the two cameras' readings of one
physical point (a quick cross-check that both extrinsics agree).

Hotkeys:
    c       : clear all clicks
    ESC / q : quit

Usage (needs intrinsics_CAM_{A,B}.npz AND extrinsics_CAM_{A,B}.npz in --calib-dir):
    python scripts/click_localize.py --calib-dir calib

Safety: cameras only, no arm motion.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent))
sys.path.insert(0, str(_SCRIPTS))

from extrinsics_live import draw_axes  # noqa: E402  reuse the axis overlay
from limbic.control import calibration, localization  # noqa: E402

PANEL_W, PANEL_H = 640, 360          # display size per camera (half of 1280x720)
BAR_H = 150                          # status bar height
FULL_W, FULL_H = 1280, 720
SX, SY = FULL_W / PANEL_W, FULL_H / PANEL_H
ORDER = ["B", "A"]                   # left panel = LEFT cam (B), right = RIGHT cam (A)


def main() -> None:
    import cv2

    from limbic.platform_support import open_camera

    ap = argparse.ArgumentParser(description="Click->table coordinate viewer (both cameras).")
    ap.add_argument("--calib-dir", required=True,
                    help="dir with intrinsics_CAM_{A,B}.npz + extrinsics_CAM_{A,B}.npz")
    args = ap.parse_args()
    calib_dir = pathlib.Path(args.calib_dir)

    cams = {}            # role -> (intr, extr, cap)
    for r in ORDER:
        try:
            intr, extr = localization.load_camera(r, calib_dir)
        except FileNotFoundError as exc:
            print(f"CAM_{r}: missing calibration ({exc}). Run intrinsics+extrinsics first.")
            return
        cap = open_camera(calibration.CAMERAS[r]["name"], width=FULL_W, height=FULL_H)
        cams[r] = (intr, extr, cap)

    clicks: dict[str, tuple] = {r: None for r in ORDER}   # role -> (u, v, x, y)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or y >= PANEL_H:
            return
        panel = 0 if x < PANEL_W else 1
        role = ORDER[panel]
        lx = x - panel * PANEL_W
        u, v = lx * SX, y * SY
        intr, extr, _ = cams[role]
        tx, ty = localization.pixel_to_table(u, v, intr, extr)
        clicks[role] = (u, v, tx, ty)
        rec = calibration.camera_role_for_y(ty)
        tag = "closer-cam OK" if rec == role else f"closer cam is CAM_{rec}"
        print(f"CAM_{role} ({calibration.CAMERAS[role]['side']}) "
              f"pixel ({u:.0f},{v:.0f}) -> table ({tx:.1f}, {ty:.1f}) mm  [{tag}]")

    win = "click -> table coord  |  c=clear  ESC=quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("Click a point in either feed to get its base-frame (x, y) in mm.")

    try:
        while True:
            canvas = np.zeros((PANEL_H + BAR_H, PANEL_W * 2, 3), np.uint8)
            for i, role in enumerate(ORDER):
                intr, extr, cap = cams[role]
                ok, frame = cap.read()
                if not ok or frame is None:
                    frame = np.zeros((FULL_H, FULL_W, 3), np.uint8)
                    cv2.putText(frame, f"CAM_{role}: no frame", (40, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    draw_axes(frame, intr, extr, calibration.CAMERAS[role]["tag_xyz_mm"])
                    if clicks[role] is not None:
                        u, v, tx, ty = clicks[role]
                        cv2.circle(frame, (int(u), int(v)), 9, (0, 255, 255), 2)
                        cv2.drawMarker(frame, (int(u), int(v)), (0, 255, 255),
                                       cv2.MARKER_CROSS, 22, 2)
                        cv2.putText(frame, f"({tx:.0f},{ty:.0f})mm", (int(u) + 12, int(v) - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                disp = cv2.resize(frame, (PANEL_W, PANEL_H))
                canvas[0:PANEL_H, i * PANEL_W:(i + 1) * PANEL_W] = disp
                side = calibration.CAMERAS[role]["side"]
                cv2.putText(canvas, f"CAM_{role}  {side}", (i * PANEL_W + 12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(canvas, f"CAM_{role}  {side}", (i * PANEL_W + 12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.line(canvas, (PANEL_W, 0), (PANEL_W, PANEL_H), (60, 60, 60), 1)

            # ---- status bar ----
            y0 = PANEL_H + 26
            for role in ORDER:
                c = clicks[role]
                if c is None:
                    txt, col = f"CAM_{role}: (click a point)", (160, 160, 160)
                else:
                    _, _, tx, ty = c
                    rec = calibration.camera_role_for_y(ty)
                    flag = "closer-cam OK" if rec == role else f"-> use CAM_{rec} (closer)"
                    txt = f"CAM_{role}: ({tx:.0f}, {ty:.0f}) mm   {flag}"
                    col = (0, 255, 0) if rec == role else (0, 200, 255)
                cv2.putText(canvas, txt, (15, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
                y0 += 34
            if clicks["A"] and clicks["B"]:
                ax, ay = clicks["A"][2], clicks["A"][3]
                bx, by = clicks["B"][2], clicks["B"][3]
                d = float(np.hypot(ax - bx, ay - by))
                cv2.putText(canvas, f"same-point delta A vs B: {d:.0f} mm "
                            f"(dx {ax-bx:.0f}, dy {ay-by:.0f})", (15, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            y0 += 30
            cv2.putText(canvas, "rule: object on RIGHT -> CAM_A, LEFT -> CAM_B "
                        "(use closer cam, don't average)", (15, PANEL_H + BAR_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(win, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            elif k == ord("c"):
                clicks = {r: None for r in ORDER}
                print("cleared.")
    finally:
        for _, _, cap in cams.values():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
