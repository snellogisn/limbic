# Vision (Part B) — how it works & where everything is

Orientation map for the vision/world-model component. For the team-wide spec see
`CLAUDE.md` Part 0 + Part B; this is the concrete "what file does what" guide.

Vision's one job (the seam, CLAUDE.md §0.3 #3): **given a camera frame + an object
name, return where each object is in the image** as `(label, (u, v))`, where `(u, v)`
is the bounding-box centre pixel. That pixel is identical to a human mouse-click, so
it drops straight into Part A's pixel→table localization. **Vision produces pixels;
Part A turns pixels into table coordinates; Part C chains them.** The viewers below
*also* chain in Part A localization to print table coords / sizes — that's validation
tooling, not the seam.

---

## Where everything is

### Library (the importable seam) — `limbic/vision/`
| File | What it is |
|------|------------|
| `detector.py` | **YOLO-World** detector (ultralytics). The packaged §0.3 #3 API: `Detector` / `Detection`, `detect(frame, prompt) -> [(label,(u,v))]`. `set_classes` re-encodes the prompt (cached). Loads bundled offline weights. |
| `workspace.py` | **Gray-mat workspace filter.** `gray_mat_mask()` finds the mat (largest low-saturation blob), `box_in_workspace()` rejects any box not fully on it. Black is excluded (the arm is black). |
| `__init__.py` | Re-exports `Detector`, `Detection`, `detect`, defaults. Lazy torch import so the base package still imports on a torch-less machine. |

### Live viewers / tools — `scripts/`
| File | What it does |
|------|--------------|
| `vision_detect_demo.py` | **Single-camera Grounding DINO** live viewer. Draws boxes on the gray mat. SPACE-free continuous feed. Good for eyeballing detection + tuning thresholds. |
| `vision_detect_dual.py` | **Dual-camera Grounding DINO** with cross-verification + table coords + object sizing. The main vision tool. (Details below.) |
| `workspace_view.py` | Visualize the gray-mat mask alone (tune the HSV thresholds). |
| `click_localize.py` | Click a pixel → table coord, in both feeds. The human-click stand-in for detection; also checks the two cameras agree. |
| `stage3_intrinsics.py` / `stage3_extrinsics.py` / `_robust.py` | Camera calibration (Part A §A.5/§A.6). `vision_detect_dual.py` imports `detect_tag` / `solve_extrinsics` from the extrinsics script for its live re-solve. |

### Calibration data — `calib/` (⚠ gitignored)
`intrinsics_CAM_{A,B}.npz` + `extrinsics_CAM_{A,B}.npz`. **Not in git** — the values
and a rebuild script are in **`docs/CALIBRATION_VALUES.md`**. Camera registry
(names/sides/tag ids/positions) lives in `limbic/control/calibration.py` (§8).

### Weights — `weights/` (Git LFS)
`yolov8s-world.pt` + `clip/ViT-B-32.pt` for YOLO-World. Grounding DINO weights load
from the local HuggingFace cache (offline). See "Offline weights" below.

---

## Two detectors (and why)

Both are **open-vocabulary** (any object name at runtime, no retraining) and both need
**PyTorch** → they run on the x64 / Mac / Linux box, never ARM64-Windows (CLAUDE.md §0.4).

1. **Grounding DINO** — HuggingFace `transformers`
   (`AutoModelForZeroShotObjectDetection`, `IDEA-Research/grounding-dino-base`, or
   `-tiny` for speed). **This is the working live detector** used in the viewer
   scripts and the active detection work. Prompt = the class list lowercased,
   period-separated: `"red cube. block. soda can."`.
   ⚠ **Needs `transformers` installed** — it is *not yet* in the `vision` extra in
   `pyproject.toml` (which only lists `torch` + `ultralytics`). Install it alongside
   torch: `pip install transformers`.
2. **YOLO-World** — `ultralytics` + `torch`, in `limbic/vision/detector.py`. The
   lighter/faster packaged seam implementation (`detect()` API). Kept as the
   importable module.

---

## The dual-camera pipeline (`scripts/vision_detect_dual.py`)

The main tool. Per **SPACE** press it runs one full-res DINO pass on **both** cameras
and produces cross-verified, measured detections. Flow per pass:

1. **Detect** (`detect_frame`): DINO on each full-res frame → boxes.
2. **Workspace filter**: drop boxes not entirely on the gray mat (`workspace.py`).
3. **Class-agnostic NMS** (`NMS_IOU`): DINO emits several overlapping boxes per
   object (worse with synonym prompts); keep the best per region, but keep *adjacent*
   objects separate.
4. **Per-camera localize + size** (`add_table_xy` → `mono_footprint`): box centre →
   table `(x, y)`; segment the object and measure its footprint. (Sizing details below.)
5. **Live extrinsics** (`live_extrinsics`): re-solve each camera's pose from its
   AprilTag *this frame*, so a bumped camera still tracks; falls back to the saved
   extrinsics if the tag isn't clean.
6. **Cross-verify + 3D fuse** (`fuse_3d`): match detections across cameras by table
   distance (`MATCH_MM`). Seen by **both** → "confirmed" (green); by **one** →
   "single" (orange). For confirmed objects it **triangulates** the two box-centre
   rays for parallax-free position + height (below).
7. **Report** (`print_report`): label, confidence, `(x, y)` mm, footprint, height, flag.

Hotkeys: **SPACE** detect · **ENTER** save image + reprint · **ESC/q** quit.

---

## Object sizing & height — the parallax problem

The footprint (`object_size_mm`) segments the object (coloured/dark vs. the gray
mat), fits an **oriented min-area rectangle**, and projects its corners to a
table-parallel plane → `(long_mm, short_mm)`, orientation-independent.

**The error:** the segmented silhouette is the object's *elevated top*, and the
cameras look **obliquely** (an AprilTag must be seen off-nadir for a clean solve,
§A.7). Ray-casting an elevated outline down to **z = 0** spreads it outward → the
footprint over-reads (historically ~±10 mm) and the position shifts away from the
camera. Elevation parallax displaces points **radially** from the camera's nadir.

**Two fixes, in the code:**

- **Stereo (best, for confirmed objects)** — `fuse_3d` + `_triangulate`: intersect
  the two cameras' box-centre rays in 3D → a parallax-free `(x, y)` **and** the
  top-of-object height `z`. Re-measure each footprint on that height plane instead of
  z = 0. Output `(long, short, height)` with a triangulation **residual** as a
  quality flag. This is the authoritative height.
- **Single-camera (for "single" objects)** — `mono_footprint`: corrects without a
  second camera, selectable by mode. *In progress* — see below. Single-cam height is
  geometrically under-constrained (the far base edge is self-occluded), so it uses one
  documented closure assumption and is an *estimate*; stereo wins when available.

Report tags each height with its source: `mono` (single-cam estimate) vs `stereo`
(triangulated) vs `?` (no height).

---

## Offline weights (this network blocks model downloads)

An SSL-intercepting proxy breaks HuggingFace **and** ultralytics auto-download, so
weights are bundled and loaded offline:

- **Grounding DINO**: set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` **before
  importing transformers** (the scripts do this at the top) → loads from the local HF
  cache. The checkpoint must already be in that cache.
- **YOLO-World**: `weights/yolov8s-world.pt` + the CLIP text encoder under
  `weights/clip/` (`detector.py` points ultralytics at the bundled dir). `.pt` files
  are tracked with **Git LFS** — run `git lfs pull` after cloning.

---

## Tuning knobs (all at the rig, not assumed — CLAUDE.md §B.4)

| Knob | Where | Effect |
|------|-------|--------|
| `BOX_THRESH` | viewers | DINO box confidence. Start permissive (NMS dedupes), raise until false positives stop. |
| `TEXT_THRESH` | viewers | DINO per-class text match. Lower → better recall on synonyms. |
| `NMS_IOU` | viewers | Merge duplicate boxes; low enough to keep adjacent objects apart. |
| `MATCH_MM` | dual | Cross-camera "same object" distance. |
| `INFER_WIDTH` | viewers | Inference resolution (full res for crowded scenes; downscale for speed). |
| `sat_max`/`val_min`/`val_max`, margin, min-area | `workspace.py` | Gray-mat detection. Raise `sat_max` if the mat is missed. |
| `CLASSES` / `classes.txt` | viewers / library | The object prompt list. |

---

## Quickstart

```bash
pip install transformers            # + torch/ultralytics (the `vision` extra)
git lfs pull                        # fetch bundled .pt weights
python - < docs/CALIBRATION_VALUES.md  # (run the rebuild snippet to recreate calib/)

python scripts/vision_detect_demo.py            # single-cam, eyeball detection
python scripts/vision_detect_dual.py            # dual-cam + cross-verify + sizing
python scripts/vision_detect_dual.py --model IDEA-Research/grounding-dino-tiny  # faster
```

Needs `calib/intrinsics_CAM_{A,B}.npz` + `extrinsics_CAM_{A,B}.npz` (see
`docs/CALIBRATION_VALUES.md`). Cameras are resolved **by name** (indices shuffle
between machines), so confirm the device names in `calibration.py` on each box.
