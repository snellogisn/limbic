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
import torch
from torchvision.ops import nms
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # for sibling scripts
from limbic.control import calibration
from limbic.control.localization import load_camera, pixel_to_table
from limbic.platform_support import open_camera
from limbic.vision.workspace import (
    DEFAULT_SAT_MAX, DEFAULT_VAL_MAX, DEFAULT_VAL_MIN,
    box_in_workspace, gray_mat_mask,
)
# Live extrinsics: re-solve the AprilTag pose every detection pass (§A.5).
from stage3_extrinsics import detect_tag, solve_extrinsics

# ─── Settings ─────────────────────────────────────────────────────────────────
CLASSES     = ["red cube", "yellow cube", "block", "card", "cylinder",
               "toy banana", "toy pear", "tape roll", "chess piece", "soda can", "bottle"]   # ← edit freely
BOX_THRESH  = 0.25
TEXT_THRESH = 0.20
INFER_WIDTH = None     # detect-on-demand -> run at FULL 1280 res (better on crowded scenes)
NMS_IOU     = 0.7      # only merge heavily-overlapping duplicates; keep ADJACENT objects apart
MATCH_MM    = 35.0     # two detections within this table distance = the same object
WIDTH, HEIGHT = 1280, 720

ROLES = ["B", "A"]     # left panel = LEFT cam (B), right = RIGHT cam (A)
PANEL_W, PANEL_H = 640, 360

TEXT_PROMPT = ". ".join(c.lower() for c in CLASSES) + "."


DEBUG = os.environ.get("VISION_DEBUG", "1") == "0"


def detect_frame(frame, processor, model, device, tag=""):
    """Detect on one full-res frame -> list of dicts with box, label, conf, centre.
    Applies NMS and the strict gray-mat workspace filter (off-mat boxes dropped)."""
    h, w = frame.shape[:2]
    ws_mask, _ = gray_mat_mask(frame)

    if INFER_WIDTH and w > INFER_WIDTH:
        scale = INFER_WIDTH / w
        infer = cv.resize(frame, (INFER_WIDTH, int(h * scale)))
    else:
        scale = 1.0
        infer = frame

    image = Image.fromarray(cv.cvtColor(infer, cv.COLOR_BGR2RGB))
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=BOX_THRESH,
        text_threshold=TEXT_THRESH, target_sizes=[image.size[::-1]],
    )[0]
    labels = results.get("text_labels", results.get("labels"))

    boxes_t, scores_t = results["boxes"], results["scores"]
    n_raw = len(boxes_t)
    if len(boxes_t) > 0:
        keep = nms(boxes_t, scores_t, NMS_IOU)
        boxes_t, scores_t = boxes_t[keep], scores_t[keep]
        labels = [labels[i] for i in keep.tolist()]
    n_nms = len(boxes_t)

    dets, n_offmat = [], 0
    for box, score, label in zip(boxes_t, scores_t, labels):
        x1, y1, x2, y2 = (box / scale).int().tolist()
        if not box_in_workspace(ws_mask, x1, y1, x2, y2):
            n_offmat += 1
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dets.append({"label": str(label), "conf": float(score),
                     "box": (x1, y1, x2, y2), "px": (cx, cy)})

    if DEBUG:
        mat = "NONE (filter OFF)" if ws_mask is None else \
            f"{int((ws_mask > 0).sum())}px ({100*(ws_mask > 0).mean():.0f}% of frame)"
        top = torch.sigmoid(outputs.logits).max(-1)[0].max().item()
        print(f"[debug {tag}] frame {w}x{h} mean={frame.mean():.0f} | mat={mat} | "
              f"top_score={top:.2f} thr={BOX_THRESH} | "
              f"raw={n_raw} -> nms={n_nms} -> off-mat dropped={n_offmat} -> kept={len(dets)}")
    return dets


def _box_edge_size_mm(box, intr, extr):
    """Fallback footprint: project the box's edge midpoints to z=0 and measure the
    two spans. Axis-aligned and includes box padding, so it over-reads — only used
    when the object can't be segmented out of the mat."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    lx, ly = pixel_to_table(x1, cy, intr, extr)
    rx, ry = pixel_to_table(x2, cy, intr, extr)
    tx, ty = pixel_to_table(cx, y1, intr, extr)
    bx, by = pixel_to_table(cx, y2, intr, extr)
    return (math.hypot(rx - lx, ry - ly), math.hypot(bx - tx, by - ty))


def object_size_mm(frame, box, intr, extr):
    """Real object footprint (long_mm, short_mm), orientation-independent.

    Segments the object inside the box (it's coloured/dark; the mat is gray =
    low-saturation), fits an ORIENTED min-area rectangle to the largest blob, and
    projects that rect's corners to the table plane. This beats the axis-aligned
    box because (a) it drops the box padding and (b) a diagonal object (e.g. a
    banana) gets its true width, not its bounding-box diagonal. Falls back to the
    box-edge projection when the object can't be cleanly segmented."""
    x1, y1, x2, y2 = box
    rx1, ry1 = max(x1, 0), max(y1, 0)
    roi = frame[ry1:y2, rx1:x2]
    if roi.size == 0:
        return _box_edge_size_mm(box, intr, extr)

    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    # object = anything that is NOT the gray mat: coloured, dark, or blown out.
    obj = ((s > DEFAULT_SAT_MAX) | (v < DEFAULT_VAL_MIN) | (v > DEFAULT_VAL_MAX))
    obj = obj.astype("uint8") * 255
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    obj = cv.morphologyEx(obj, cv.MORPH_OPEN, k)

    cnts, _ = cv.findContours(obj, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return _box_edge_size_mm(box, intr, extr)
    c = max(cnts, key=cv.contourArea)
    if cv.contourArea(c) < 0.10 * roi.shape[0] * roi.shape[1]:
        return _box_edge_size_mm(box, intr, extr)  # too small/sparse to trust

    pts = cv.boxPoints(cv.minAreaRect(c))            # 4 corners, ROI px
    pts = pts + np.array([rx1, ry1], dtype=np.float32)  # -> full-frame px
    table = [pixel_to_table(px, py, intr, extr) for px, py in pts]
    side1 = math.hypot(table[1][0] - table[0][0], table[1][1] - table[0][1])
    side2 = math.hypot(table[2][0] - table[1][0], table[2][1] - table[1][1])
    return (max(side1, side2), min(side1, side2))


def add_table_xy(dets, intr, extr, frame=None):
    """Attach the base-frame (x, y) mm centre and the real-world (long, short) size.

    Size comes from an oriented segmentation of the object (``object_size_mm``)
    when ``frame`` is given; otherwise it degrades to the box-edge projection.
    """
    for d in dets:
        d["xy"] = pixel_to_table(d["px"][0], d["px"][1], intr, extr)
        d["size_mm"] = (object_size_mm(frame, d["box"], intr, extr)
                        if frame is not None else _box_edge_size_mm(d["box"], intr, extr))
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


def mark_confirmed(dets_a, dets_b):
    """Flag each detection 'confirmed' if the OTHER camera has one within MATCH_MM."""
    def near(d, others):
        return any(math.hypot(d["xy"][0] - o["xy"][0], d["xy"][1] - o["xy"][1]) <= MATCH_MM
                   for o in others)
    for d in dets_a:
        d["confirmed"] = near(d, dets_b)
    for d in dets_b:
        d["confirmed"] = near(d, dets_a)


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
    found = False
    for r in ROLES:
        for d in dets[r]:
            found = True
            x, y = d["xy"]
            w, h = d["size_mm"]
            flag = "CONFIRMED" if d.get("confirmed") else "single"
            print(f"  CAM_{r} {d['label']:<14} {d['conf']:.2f} "
                  f"({x:7.1f}, {y:7.1f}) mm  size {w:5.0f}x{h:5.0f} mm  [{flag}]")
    if not found:
        print("  (none)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-camera detection with cross-verification.")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--no-live-recalib", action="store_true",
                    help="use the saved extrinsics instead of re-solving the tag each pass")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device.upper()} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device)
    model.eval()
    print("Model ready.")

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
                for r in ROLES:
                    intr, saved_extr = calib[r]
                    extr = (live_extrinsics(r, frames[r], intr, saved_extr, tag_yaw[r])[0]
                            if live else saved_extr)
                    last[r] = add_table_xy(
                        detect_frame(frames[r], processor, model, device, tag=f"CAM_{r}"),
                        intr, extr, frame=frames[r])
                mark_confirmed(last["A"], last["B"])
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
