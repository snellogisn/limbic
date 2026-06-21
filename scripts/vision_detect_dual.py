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
# Object-footprint sizing now lives in the package, so the viewer and the live
# pipeline share ONE copy.
from limbic.vision.sizing import box_edge_size_mm as _box_edge_size_mm
from limbic.vision.sizing import mono_footprint, pixel_ray, triangulate
from limbic.vision.workspace import box_in_workspace, gray_mat_mask
# Live extrinsics: re-solve the AprilTag pose every detection pass (§A.5).
from stage3_extrinsics import detect_tag, solve_extrinsics

# ─── Settings ─────────────────────────────────────────────────────────────────
CLASSES     = ["red cube", "yellow cube", "block", "card", "cylinder",
               "toy banana", "toy pear", "tape roll", "chess piece", "soda can", "bottle"]   # ← edit freely
BOX_THRESH  = 0.25
TEXT_THRESH = 0.20
NMS_IOU     = 0.7      # only merge heavily-overlapping duplicates; keep ADJACENT objects apart
MATCH_MM    = 35.0     # two detections within this table distance = the same object
SIZE_MODE   = "auto"   # single-cam footprint/height correction: z0 | contour | height | auto
WIDTH, HEIGHT = 1280, 720

ROLES = ["B", "A"]     # left panel = LEFT cam (B), right = RIGHT cam (A)
PANEL_W, PANEL_H = 640, 360


DEBUG = os.environ.get("VISION_DEBUG", "1") == "0"


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

    if DEBUG:
        mat = "NONE (filter OFF)" if ws_mask is None else \
            f"{int((ws_mask > 0).sum())}px ({100*(ws_mask > 0).mean():.0f}% of frame)"
        print(f"[debug {tag}] frame {w}x{h} mean={frame.mean():.0f} | mat={mat} | "
              f"thr={BOX_THRESH} | raw={len(raw)} -> off-mat dropped={n_offmat} "
              f"-> kept={len(dets)}")
    return dets


def add_table_xy(dets, intr, extr, frame=None, mode=SIZE_MODE):
    """Attach the base-frame (x, y) mm centre, footprint, and (single-cam) height.

    This is the SINGLE-camera estimate: ``mono_footprint`` segments the object and
    corrects the z=0 elevation inflation per ``mode`` (z0 | contour | height |
    auto), returning a footprint, a height estimate, and a centre on the object's
    own plane. ``fuse_3d`` later UPGRADES both position and height to stereo for any
    object both cameras confirm. Without a frame, degrades to the box-edge size.
    """
    for d in dets:
        if frame is not None:
            long_, short_, height, centre = mono_footprint(frame, d["box"], intr, extr, mode)
            d["xy"] = centre
            d["size_mm"] = (long_, short_)
            d["height_mm"] = height
            d["height_src"] = None if height is None else "mono"
        else:
            d["xy"] = pixel_to_table(d["px"][0], d["px"][1], intr, extr)
            d["size_mm"] = _box_edge_size_mm(d["box"], intr, extr)
            d["height_mm"] = None
            d["height_src"] = None
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
        if DEBUG:
            print(f"[recalib CAM_{role}] tag id {cam['tag_id']} NOT seen "
                  "-> using SAVED extrinsics")
        return fallback, None
    extr = solve_extrinsics(corners, intr, cam["tag_xyz_mm"],
                            calibration.APRILTAG_SIZE_MM, tag_yaw)
    if extr is None:
        if DEBUG:
            print(f"[recalib CAM_{role}] solvePnP failed -> using SAVED extrinsics")
        return fallback, None
    if DEBUG:
        c = extr.t_cam2base
        print(f"[recalib CAM_{role}] tag OK -> LIVE cam centre "
              f"({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}) mm")
    return extr, corners


def fuse_3d(dets, frames, cal):
    """Cross-camera confirmation + 3D fusion (replaces flat 2D confirmation).

    For each object both cameras agree on (box centres within ``MATCH_MM`` on z=0),
    intersect the two box-centre rays in 3D (``triangulate``) to recover a
    PARALLAX-FREE centre ``(x, y)`` and the top-of-object HEIGHT ``z`` (table = 0),
    then re-measure each camera's footprint on that height plane. Both partners are
    flagged ``confirmed`` and given matching ``xy`` / ``height_mm`` / ``size_mm``
    with ``height_src="stereo"`` (stereo beats the single-cam estimate). Unmatched
    detections stay ``confirmed=False`` and keep their mono estimate.

    ``cal`` maps role -> (intr, extr) actually used this pass; ``frames`` maps role
    -> the BGR frame (to re-measure the footprint on the height plane).
    """
    from limbic.vision.sizing import object_size_mm

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
        height = max(0.0, float(pt[2]))
        centre = (float(pt[0]), float(pt[1]))
        for d, role in ((da, "A"), (db, "B")):
            intr, extr = cal[role]
            d["xy"] = centre                  # parallax-free, shared by both cams
            d["height_mm"] = height
            d["height_src"] = "stereo"
            d["resid_mm"] = resid
            d["size_mm"] = object_size_mm(frames[role], d["box"], intr, extr, plane_z=height)


def draw(frame, dets):
    """Draw only the detection boxes + centre dots. The object DATA (label, conf,
    xy, size) is printed to the terminal (see print_report), not the GUI."""
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        col = (0, 255, 0) if d.get("confirmed") else (0, 165, 255)   # green / orange
        cv.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv.circle(frame, (int(d["px"][0]), int(d["px"][1])), 4, col, -1)


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
    print("  (footprint = long x short; height: (stereo)=triangulated, "
          "(mono)=single-cam estimate, '?'=none)")
    found = False
    for r in ROLES:
        for d in dets[r]:
            found = True
            x, y = d["xy"]
            w, h = d["size_mm"]
            ht, src = d.get("height_mm"), d.get("height_src")
            ht_s = f"{ht:4.0f}({src})" if ht is not None else "      ?"
            res = d.get("resid_mm")
            res_s = f" ~{res:.0f}mm" if res is not None else ""
            flag = "CONFIRMED" if d.get("confirmed") else "single"
            print(f"  CAM_{r} {d['label']:<14} {d['conf']:.2f} "
                  f"({x:7.1f}, {y:7.1f}) mm  footprint {w:5.0f}x{h:5.0f}  "
                  f"height {ht_s} mm  [{flag}{res_s}]")
    if not found:
        print("  (none)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-camera detection with cross-verification.")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--no-live-recalib", action="store_true",
                    help="use the saved extrinsics instead of re-solving the tag each pass")
    ap.add_argument("--size-mode", choices=["z0", "contour", "height", "auto"],
                    default=SIZE_MODE,
                    help="single-cam footprint/height correction (default: %(default)s)")
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
                        intr, extr, frame=frames[r], mode=args.size_mode)
                fuse_3d(last, frames, cal_pass)   # confirm + triangulate height + fix footprint
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
