import os

# This network's proxy blocks HuggingFace downloads, so force transformers to use
# the LOCAL cache only (both grounding-dino checkpoints are already cached). Must
# be set before importing transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
import pathlib

import cv2 as cv
import numpy as np
import torch
from torchvision.ops import nms
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# limbic compatibility: open the CALIBRATED rig camera BY NAME (§8).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from limbic.control import calibration
from limbic.platform_support import open_camera

# ─── Settings ─────────────────────────────────────────────────────────────────
CLASSES      = ["chess piece", "chess", "red cube", "yellow cube", "block", "card", "wooden plank", "wood", "cylinder", "warthog", "toy banana", "toy pear", "tape", "tape roll", "measuring tape"]                       # ← edit for whatever you want
MODEL_ID = "IDEA-Research/grounding-dino-base"  # or "...-base" (slower, better)
BOX_THRESH   = 0.25       # min confidence to keep a box (lowered: NMS dedupes now)
TEXT_THRESH  = 0.20       # min text-match score per class (lower = better recall on synonyms)
INFER_WIDTH  = 960        # downscale width for inference (speed). None = full res.
NMS_IOU      = 0.5        # merge boxes overlapping more than this IoU (class-agnostic)

# Workspace = the gray mat, auto-detected by colour each frame. Detections must
# sit ENTIRELY on it; anything off the mat or touching its edge is dropped.
# Gray = low saturation. Tune these if the mat isn't found / the wrong region is
# picked (press 'm' in the viewer to toggle the filter off and see everything).
GRAY_SAT_MAX  = 80        # max HSV saturation to count as "gray" (raise if mat missed)
GRAY_VAL_MIN  = 25        # ignore near-black pixels below this brightness
GRAY_VAL_MAX  = 205       # ignore blown-out highlights above this brightness
MAT_MARGIN_PX = 8         # shrink the mat inward this many px (enforces "not touching the edge")
MAT_MIN_AREA  = 0.05      # ignore gray blobs smaller than this fraction of the frame

# Calibrated rig camera: role A = right-side C930e (§8), captured at the 1280x720
# calibration resolution so a detected box centre is a valid pixel for
# pixel->table localization. Switch to "B" for the left C920.
CAMERA       = calibration.CAMERAS["B"]["name"]
WIDTH, HEIGHT = 1280, 720

# Grounding DINO wants lowercase classes, each ending in a period.
TEXT_PROMPT = ". ".join(c.lower() for c in CLASSES) + "."

# ─── Load model ───────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Grounding DINO on {device.upper()} … (first run downloads weights)")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
model.eval()
print("Model ready.")


def workspace_mask(frame):
    """Find the gray mat by colour and return (eroded_mask, outline_contour).

    Gray = low saturation. We take the largest gray blob as the mat, FILL it
    (so objects sitting on it count as inside), then erode by MAT_MARGIN_PX so a
    box has to clear the edge. Returns (None, None) if no plausible mat is found.
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    gray = ((s <= GRAY_SAT_MAX) & (v >= GRAY_VAL_MIN) & (v <= GRAY_VAL_MAX))
    mask = (gray.astype("uint8")) * 255
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k)

    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv.contourArea)
    if cv.contourArea(c) < MAT_MIN_AREA * frame.shape[0] * frame.shape[1]:
        return None, None

    filled = np.zeros(frame.shape[:2], "uint8")
    cv.drawContours(filled, [c], -1, 255, -1)        # solid mat (holes for objects filled)
    if MAT_MARGIN_PX > 0:
        ek = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * MAT_MARGIN_PX + 1,) * 2)
        filled = cv.erode(filled, ek)
    return filled, c


def liveDetect():
    cap = open_camera(CAMERA, width=WIDTH, height=HEIGHT)

    if not cap.isOpened():
        print(f"Error: could not open camera {CAMERA!r}.")
        return

    os.makedirs("Images", exist_ok=True)

    while True:
        success, frame = cap.read()

        if not success:
            break

        h, w = frame.shape[:2]

        # Gray-mat workspace (on the FULL frame, where the rescaled boxes live).
        # Always on, behind the scenes — used only to filter detections, never drawn.
        ws_mask, _ = workspace_mask(frame)

        # Downscale for inference, remember the scale to map boxes back.
        if INFER_WIDTH and w > INFER_WIDTH:
            scale = INFER_WIDTH / w
            infer = cv.resize(frame, (INFER_WIDTH, int(h * scale)))
        else:
            scale = 1.0
            infer = frame

        # BGR (OpenCV) → RGB (PIL) for the processor.
        image = Image.fromarray(cv.cvtColor(infer, cv.COLOR_BGR2RGB))

        inputs = processor(images=image, text=TEXT_PROMPT,
                           return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESH,
            text_threshold=TEXT_THRESH,
            target_sizes=[image.size[::-1]],   # (height, width) of infer image
        )[0]

        # Newer transformers use "text_labels"; older use "labels".
        labels = results.get("text_labels", results.get("labels"))

        # Class-agnostic NMS: Grounding DINO emits several overlapping boxes for the
        # same object (especially when the prompt has synonyms). Keep the highest-
        # scoring box per region, dropping the duplicates regardless of label.
        boxes_t, scores_t = results["boxes"], results["scores"]
        if len(boxes_t) > 0:
            keep = nms(boxes_t, scores_t, NMS_IOU)
            boxes_t = boxes_t[keep]
            scores_t = scores_t[keep]
            labels = [labels[i] for i in keep.tolist()]

        for box, score, label in zip(boxes_t, scores_t, labels):
            x1, y1, x2, y2 = (box / scale).int().tolist()   # back to full frame

            # Strict workspace filter: skip boxes that leave the frame, leave the
            # mat, or touch its (margin-shrunk) edge.
            if ws_mask is not None:
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    continue
                sub = ws_mask[y1:y2, x1:x2]
                if sub.size == 0 or (sub == 0).any():
                    continue

            conf = float(score)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.imshow("Grounding DINO  |  ENTER save  ESC quit", frame)
        key = cv.waitKey(1) & 0xFF

        if key == 13:   # ENTER — save annotated frame
            cv.imwrite("Images/picture.png", frame)
            print("\nFrame saved to Images/picture.png\n")
        elif key == 27: # ESC — quit
            print("Terminated.")
            break

    cap.release()
    cv.destroyAllWindows()

liveDetect()