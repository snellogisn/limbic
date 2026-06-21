"""Dual-camera open-vocab detection with cross-verification (Part B).

Runs Grounding DINO on BOTH calibrated cameras at once, side by side, with the
same workspace (gray-mat) filtering as the single-camera viewer. The payoff of
two cameras: every kept detection's box-centre is converted to a TABLE
coordinate (mm, base frame) via that camera's calibration — and because both
cameras share one frame, the same physical object lands at the same (x, y) in
both. So we can CROSS-VERIFY:

    * seen by BOTH cameras at the same (x, y)  -> "confirmed" (green)
    * seen by only ONE camera                  -> "single"    (orange)

This is the redundancy the LLM reasons over: trust confirmed objects, treat
singletons with suspicion (occlusion, glare, or a false positive). The §8 rule
still applies downstream — once confirmed, take the reading from the camera on
that object's side (it's most accurate there).

Usage (needs calib/ with intrinsics+extrinsics for A and B):
    python scripts/vision_detect_dual.py
    python scripts/vision_detect_dual.py --model IDEA-Research/grounding-dino-tiny  # faster

Detect-on-demand: the live feed stays smooth; detection (two full-res DINO passes)
runs only when you press SPACE, and the result persists on screen until the next
press. Good for a static tabletop and keeps full base-model quality.

Hotkeys:
    SPACE   : run a full-res detection pass on both cameras (cross-verify)
    ENTER   : save the current view + reprint the last report
    ESC / q : quit
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import math
import pathlib
import sys
import time

import cv2 as cv
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # for sibling scripts
from limbic.control import calibration
from limbic.control.localization import load_camera, pixel_to_table
from limbic.platform_support import open_camera
# The same Grounding DINO detector the live pipeline uses
# (inputs/library/object_detections.py -> limbic.vision.get_detector).
from limbic.vision.dino import DinoDetector
# pixel_ray + triangulate give a PARALLAX-FREE (x, y) for cross-confirmed objects
# (we use the x,y only; sizing/height estimation has been removed).
from limbic.vision.sizing import pixel_ray, triangulate
from limbic.vision.workspace import box_in_workspace, gray_mat_mask
# Live extrinsics: re-solve the AprilTag pose every detection pass (§A.5).
from stage3_extrinsics import detect_tag, solve_extrinsics

# ─── Settings ─────────────────────────────────────────────────────────────────
CLASSES     = ["red cube", "yellow cube", "card", "wooden plank", "wooden board", "wooden cube", "wooden block", "toy banana", "banana"]   # ← edit freely
BOX_THRESH  = 0.25
TEXT_THRESH = 0.20
NMS_IOU     = 0.7      # only merge heavily-overlapping duplicates; keep ADJACENT objects apart
MATCH_MM    = 35.0     # two detections within this table distance = the same object
WIDTH, HEIGHT = 1280, 720

ROLES = ["B", "A"]     # left panel = LEFT cam (B), right = RIGHT cam (A)
PANEL_W, PANEL_H = 640, 360


def detect_frame(frame, detector, tag=""):
    """Detect on one full-res frame -> list of dicts with box, label, conf, centre.

    Delegates to the shared Grounding DINO detector (the same one the live
    pipeline uses), then applies the strict gray-mat workspace filter (off-mat
    boxes dropped). NMS happens inside the detector."""
    h, w = frame.shape[:2]
    ws_mask, _ = gray_mat_mask(frame)

    raw = detector.detect_boxes(
        frame, CLASSES, conf=BOX_THRESH,
        text_threshold=TEXT_THRESH, nms_iou=NMS_IOU,
    )

    dets, n_offmat = [], 0
    for d in raw:
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        if not box_in_workspace(ws_mask, x1, y1, x2, y2):
            n_offmat += 1
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dets.append({"label": d.label, "conf": d.confidence,
                     "box": (x1, y1, x2, y2), "px": (cx, cy)})
    return dets

def add_table_xy(dets, intr, extr):
    """Attach the base-frame (x, y) mm centre of each box (pixel -> table, z=0),
    plus an estimated object HEIGHT in mm (``box_height_mm``).

    This is the single-camera read. ``fuse_3d`` later UPGRADES the position to a
    parallax-free stereo (x, y) for any object both cameras confirm. The height
    is independent of that — it's a per-box single-camera estimate.
    """
    for d in dets:
        d["xy"] = pixel_to_table(d["px"][0], d["px"][1], intr, extr)
    return dets


def load_tag_yaw(role, calib_dir, default=0.0):
    """Read the tag_yaw_deg baked into the saved extrinsics so the live re-solve
    reuses the human-verified yaw (§A.7) — no need to re-eyeball it each frame."""
    p = pathlib.Path(calib_dir) / f"extrinsics_CAM_{role}.npz"
    try:
        d = np.load(str(p))
        if "tag_yaw_deg" in d.files:
            return float(d["tag_yaw_deg"])
    except Exception:
        pass
    return default


def live_extrinsics(role, frame, intr, fallback, tag_yaw):
    """Re-solve this camera's pose from the AprilTag in THIS frame, so the xy math
    tracks a bumped camera. Falls back to the saved extrinsics if the tag isn't
    clean. Returns (extrinsics, corners_or_None)."""
    cam = calibration.CAMERAS[role]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners = detect_tag(gray, cam["tag_id"])
    if corners is None:
        return fallback, None
    extr = solve_extrinsics(corners, intr, cam["tag_xyz_mm"],
                            calibration.APRILTAG_SIZE_MM, tag_yaw)
    
    return extr, corners


def fuse_3d(dets, cal):
    """Cross-camera confirmation + parallax-free (x, y).

    For each object both cameras agree on (box centres within ``MATCH_MM`` on z=0),
    intersect the two box-centre rays in 3D (``triangulate``) to recover a
    PARALLAX-FREE centre ``(x, y)``. Both partners are flagged ``confirmed`` and
    given that shared ``xy`` (better than either single-camera z=0 read). Unmatched
    detections stay ``confirmed=False`` and keep their mono z=0 ``xy``.

    ``cal`` maps role -> (intr, extr) actually used this pass.
    """
    A, B = dets["A"], dets["B"]
    for d in A + B:
        d["confirmed"] = False
    used = set()
    for da in A:
        best, match = MATCH_MM, None
        for j, db in enumerate(B):
            if j in used:
                continue
            dist = math.hypot(da["xy"][0] - db["xy"][0], da["xy"][1] - db["xy"][1])
            if dist <= best:
                best, match = dist, j
        if match is None:
            continue
        db = B[match]
        used.add(match)
        da["confirmed"] = db["confirmed"] = True

        o_a, dir_a = pixel_ray(da["px"][0], da["px"][1], *cal["A"])
        o_b, dir_b = pixel_ray(db["px"][0], db["px"][1], *cal["B"])
        pt, resid = triangulate(o_a, dir_a, o_b, dir_b)
        if pt is None:
            continue
        centre = (float(pt[0]), float(pt[1]))
        da["xy"] = db["xy"] = centre          # parallax-free, shared by both cams
        da["resid_mm"] = db["resid_mm"] = resid


def draw(frame, dets):
    """Draw each detection's box, centre dot, and label + table (x, y) coord.

    Coords are shown on the image AND printed to the terminal (see print_report)."""
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        col = (0, 255, 0) if d.get("confirmed") else (0, 165, 255)   # green / orange
        cv.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cx, cy = int(d["px"][0]), int(d["px"][1])
        cv.circle(frame, (cx, cy), 4, col, -1)
        x, y = d["xy"]
        h = d.get("height_mm")
        htxt = f" h{h:.0f}" if h is not None else ""
        _label(frame, f"{d['label']} ({x:.0f}, {y:.0f})mm{htxt}",
               (x1, max(y1 - 8, 18)), col, 0.8)


def _label(img, text, org, color, scale=0.6):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, 2 if scale >= 0.6 else 1)


def build_canvas(frames, dets, fps, detected_once):
    canvas = np.zeros((PANEL_H, PANEL_W * 2, 3), np.uint8)
    for i, r in enumerate(ROLES):
        disp = cv.resize(frames[r], (PANEL_W, PANEL_H))
        canvas[0:PANEL_H, i * PANEL_W:(i + 1) * PANEL_W] = disp
        nconf = sum(d.get("confirmed", False) for d in dets[r])
        hdr = f"CAM_{r} {calibration.CAMERAS[r]['side']} | {len(dets[r])} det, {nconf} conf"
        _label(canvas, hdr, (i * PANEL_W + 12, 26), (255, 255, 0))
    cv.line(canvas, (PANEL_W, 0), (PANEL_W, PANEL_H), (60, 60, 60), 1)
    hint = "SPACE detect   ENTER save+report   ESC quit"
    if not detected_once:
        hint = "press SPACE to detect   |   " + hint
    _label(canvas, hint, (12, PANEL_H - 12), (255, 255, 255), 0.5)
    _label(canvas, f"{fps:4.1f} fps", (PANEL_W * 2 - 90, 20), (200, 200, 200), 0.5)
    return canvas


def print_report(dets):
    print("\n=== cross-verified detections ===")
    found = False
    for r in ROLES:
        for d in dets[r]:
            found = True
            x, y = d["xy"]
            res = d.get("resid_mm")
            res_s = f" ~{res:.0f}mm" if res is not None else ""
            flag = "CONFIRMED" if d.get("confirmed") else "single"
            h = d.get("height_mm")
            h_s = f"  h={h:5.0f}mm" if h is not None else "  h=  ?  "
            print(f"  CAM_{r} {d['label']:<14} {d['conf']:.2f} "
                  f"({x:7.1f}, {y:7.1f}) mm{h_s}  [{flag}{res_s}]")
    if not found:
        print("  (none)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-camera detection with cross-verification.")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--no-live-recalib", action="store_true",
                    help="use the saved extrinsics instead of re-solving the tag each pass")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    detector = DinoDetector(args.model)
    print(f"Model ready (on {detector.device.upper()}).")

    calib_dir = pathlib.Path(args.calib_dir)
    cams, calib, tag_yaw = {}, {}, {}
    for r in ROLES:
        cams[r] = open_camera(calibration.CAMERAS[r]["name"], width=WIDTH, height=HEIGHT)
        calib[r] = load_camera(r, calib_dir)   # (intr, extr) — extr is the fallback
        tag_yaw[r] = load_tag_yaw(r, calib_dir)
    live = not args.no_live_recalib
    print(f"Extrinsics: {'LIVE per-pass tag re-solve' if live else 'saved (static)'} "
          f"| tag_yaw {tag_yaw}")

    os.makedirs("Images", exist_ok=True)
    win = "Dual detection (green=confirmed, orange=single)  |  SPACE detect  ESC quit"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    fps = 0.0
    last = {"A": [], "B": []}   # most recent detection pass (persists on the live feed)
    detected_once = False

    try:
        while True:
            t0 = time.time()
            # --- smooth live feed: just grab + show, no inference ---
            frames = {}
            for r in ROLES:
                ok, f = cams[r].read()
                if not ok or f is None:
                    f = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                    cv.putText(f, f"CAM_{r}: no frame", (40, 80),
                               cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                frames[r] = f
            for r in ROLES:
                draw(frames[r], last[r])   # overlay the last detection pass

            fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - t0, 1e-3))
            canvas = build_canvas(frames, last, fps, detected_once)
            cv.imshow(win, canvas)

            key = cv.waitKey(1) & 0xFF
            if key == 32:   # SPACE — run one full-res detection pass on both cameras
                busy = canvas.copy()
                _label(busy, "DETECTING...", (PANEL_W - 110, PANEL_H // 2), (0, 255, 255), 1.0)
                cv.imshow(win, busy)
                cv.waitKey(1)
                cal_pass = {}   # the (intr, extr) actually used this pass, per camera
                for r in ROLES:
                    intr, saved_extr = calib[r]
                    extr = (live_extrinsics(r, frames[r], intr, saved_extr, tag_yaw[r])[0]
                            if live else saved_extr)
                    cal_pass[r] = (intr, extr)
                    last[r] = add_table_xy(
                        detect_frame(frames[r], detector, tag=f"CAM_{r}"),
                        intr, extr)
                fuse_3d(last, cal_pass)   # confirm matches + parallax-free (x, y)
                detected_once = True
                print_report(last)
            elif key == 13:   # ENTER — save the view + reprint the last report
                cv.imwrite(f"Images/dual_{time.strftime('%H%M%S')}.png", canvas)
                print(f"saved Images/dual_{time.strftime('%H%M%S')}.png")
                print_report(last)
            elif key in (27, ord("q")):
                break
    finally:
        for c in cams.values():
            c.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()