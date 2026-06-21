"""Grounding DINO open-vocabulary detector (Part B) — same seam as YOLO-World.

Wraps IDEA-Research **Grounding DINO** behind the SAME interface as
``limbic.vision.detector.Detector`` (``detect_boxes() -> list[Detection]`` and
``detect() -> [(label, (u, v)), ...]``), so it is a DROP-IN for the integrated
``object_detections`` pipeline — swapping the model never touches the brain loop.

The detection logic mirrors the proven ``scripts/vision_detect_dual.py`` dual-
camera viewer: full-resolution inference, a box threshold + a text threshold, and
class-agnostic NMS to merge the several overlapping boxes Grounding DINO emits per
object. Those defaults (``DEFAULT_*``) are the script's, tuned at the rig (§B.4).

``torch`` + ``transformers`` are the optional 'vision' extra (§0.4); they are
imported lazily inside the methods so importing this module never requires them —
only actually constructing a :class:`DinoDetector` does.
"""

from __future__ import annotations

import os

from .detector import Detection

# Use the LOCAL Hugging Face cache only. Like the bundled YOLO weights, this is
# because the rig network has an SSL-intercepting proxy that breaks auto-download
# (see scripts/vision_detect_dual.py, which sets the same flags). setdefault so a
# user can still opt back into online mode by exporting these to "0".
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 'base' is the accurate default the dual-camera script ships with; export
# LIMBIC_DINO_MODEL=IDEA-Research/grounding-dino-tiny for a faster, lighter pass.
DEFAULT_DINO_MODEL = os.environ.get("LIMBIC_DINO_MODEL", "IDEA-Research/grounding-dino-base")

# Grounding DINO scores differently from a closed-set detector; these mirror the
# dual-camera viewer's rig-tuned thresholds (§B.4).
DEFAULT_BOX_THRESHOLD = 0.25   # min box confidence to keep a detection
DEFAULT_TEXT_THRESHOLD = 0.20  # min text-grounding score for the matched phrase
DEFAULT_NMS_IOU = 0.7          # merge only heavily-overlapping dups; keep neighbours apart


def _as_prompt_text(prompt: "str | list[str]") -> str:
    """Build the Grounding DINO text query: lowercase phrases joined by '. '.

    Grounding DINO is prompted with a single caption string where each candidate
    object is a lowercase phrase ending in a period (e.g. ``"red cube. block."``).
    Accepts the same inputs as the YOLO detector — a comma-separated string or a
    list of names — so the two backends are interchangeable behind the seam.
    """
    if isinstance(prompt, str):
        names = [p.strip() for p in prompt.split(",") if p.strip()]
    else:
        names = [str(p).strip() for p in prompt if str(p).strip()]
    if not names:
        raise ValueError("empty prompt — give at least one object name")
    return ". ".join(n.lower() for n in names) + "."


class DinoDetector:
    """A Grounding DINO detector. Construct once, reuse across frames.

    Unlike YOLO-World there is no persistent class state to set: Grounding DINO
    takes the object names as a text caption on every call, so ``detect_boxes``
    always needs a ``prompt`` (open-vocabulary, no retraining).
    """

    def __init__(self, model: str = DEFAULT_DINO_MODEL, device: "str | None" = None):
        # Lazy: the optional 'vision' extra (torch + transformers), §0.4.
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_id = model

        # When offline (the rig default — the proxy blocks HF downloads), force
        # local-files-only so a MISSING cached model fails FAST with a clear error
        # instead of hanging on a proxied network call. Pre-cache the weights on
        # the demo box (like the bundled YOLO weights) or export HF_HUB_OFFLINE=0.
        offline = os.environ.get("HF_HUB_OFFLINE", "0").lower() in ("1", "true", "yes")
        try:
            self.processor = AutoProcessor.from_pretrained(model, local_files_only=offline)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model, local_files_only=offline
            ).to(device)
        except Exception as exc:
            if offline:
                raise RuntimeError(
                    f"Grounding DINO model {model!r} is not in the local Hugging Face "
                    "cache and HF_HUB_OFFLINE is set (the rig proxy blocks downloads). "
                    "Pre-cache it on a machine with internet "
                    f"(python -c \"from transformers import AutoModelForZeroShotObjectDetection as M; "
                    f"M.from_pretrained('{model}')\"), or export HF_HUB_OFFLINE=0 to fetch it now."
                ) from exc
            raise
        self.model.eval()

    def detect_boxes(
        self,
        frame,
        prompt: "str | list[str] | None" = None,
        conf: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        nms_iou: float = DEFAULT_NMS_IOU,
        max_det: int = 100,
    ) -> list[Detection]:
        """Detect objects, returning full ``Detection`` records (with boxes).

        ``frame`` is a BGR image (numpy array, as OpenCV gives). ``conf`` is the
        box-confidence threshold (Grounding DINO's ``threshold``); ``max_det``
        caps how many boxes are returned. Mirrors the dual-camera script's
        pipeline: inference -> threshold -> class-agnostic NMS -> ``Detection``s.
        """
        import cv2 as cv
        import torch
        from PIL import Image
        from torchvision.ops import nms

        if prompt is None:
            raise ValueError(
                "Grounding DINO needs the object name(s) on every call — pass prompt="
            )
        text = _as_prompt_text(prompt)

        image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=float(conf),
            text_threshold=float(text_threshold),
            target_sizes=[image.size[::-1]],
        )[0]

        # transformers renamed this key across versions ('text_labels' is newer).
        labels = results.get("text_labels", results.get("labels"))
        boxes_t, scores_t = results["boxes"], results["scores"]

        # Class-agnostic NMS: Grounding DINO emits several overlapping boxes for
        # the same object — collapse them, but keep ADJACENT objects apart.
        if len(boxes_t) > 0:
            keep = nms(boxes_t, scores_t, nms_iou)
            boxes_t, scores_t = boxes_t[keep], scores_t[keep]
            labels = [labels[i] for i in keep.tolist()]

        dets: list[Detection] = []
        for box, score, label in zip(boxes_t, scores_t, labels):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            dets.append(
                Detection(label=str(label), confidence=float(score), box=(x1, y1, x2, y2))
            )
            if len(dets) >= max_det:
                break
        return dets

    def detect(
        self,
        frame,
        prompt: "str | list[str] | None" = None,
        conf: float = DEFAULT_BOX_THRESHOLD,
    ) -> list[tuple[str, tuple[float, float]]]:
        """The §0.3 #3 seam: ``[(label, (u, v)), ...]`` (bbox-centre pixels)."""
        return [(d.label, d.center) for d in self.detect_boxes(frame, prompt, conf)]
