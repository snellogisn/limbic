"""limbic.vision — open-vocabulary object detection (Part B, the world model).

The Senses' eyes: given a camera frame + an object name/prompt, find where each
named object is in the image. Exposes the §0.3 #3 seam

    detect(frame, prompt) -> [(label, (u, v)), ...]

where (u, v) is the bounding-box CENTRE pixel — exactly the "click" the arm side
(Part A localization) turns into a table coordinate, so vision is a drop-in
replacement for a human click.

Two interchangeable backends sit behind that seam:

  * **Grounding DINO** (``transformers`` + ``torch``) — the dual-camera viewer's
    detector; the default when ``transformers`` is installed.
  * **YOLO-World** (``ultralytics`` + ``torch``) — the lighter fallback.

Both are OPTIONAL 'vision' extras (§0.4) with lazy imports, so the base limbic
package still imports on machines without torch (e.g. ARM64). Pick one with
:func:`get_detector` (honours ``$LIMBIC_VISION_BACKEND``); the integrated
``object_detections`` pipeline calls it, so a backend swap never touches the
brain loop.
"""

from __future__ import annotations

import os

from .detector import DEFAULT_CONF, DEFAULT_MODEL, Detection, Detector, detect

# Built detectors are expensive (model load); cache one per backend per process.
_DETECTORS: dict[str, object] = {}


def _default_backend() -> str:
    """Pick the detection backend: explicit env override, else best available.

    ``$LIMBIC_VISION_BACKEND`` ("dino" | "yolo") wins when set. Otherwise we
    prefer Grounding DINO when ``transformers`` is importable (it's the rig's
    tuned detector), and fall back to YOLO-World when it isn't.
    """
    override = os.environ.get("LIMBIC_VISION_BACKEND")
    if override:
        return override.strip().lower()
    import importlib.util

    return "dino" if importlib.util.find_spec("transformers") is not None else "yolo"


def get_detector(backend: str | None = None):
    """Return a shared detector for ``backend`` (built once, reused per process).

    The returned object exposes the same interface regardless of backend —
    ``detect_boxes(frame, prompt, conf=...) -> list[Detection]`` and
    ``detect(frame, prompt) -> [(label, (u, v)), ...]`` — so callers (notably the
    ``object_detections`` input) never branch on which model is running.
    """
    name = (backend or _default_backend()).lower()
    if name in ("dino", "grounding-dino", "groundingdino"):
        key = "dino"
    elif name in ("yolo", "yolo-world", "yoloworld"):
        key = "yolo"
    else:
        raise ValueError(f"unknown vision backend {name!r} (use 'dino' or 'yolo')")

    if key not in _DETECTORS:
        if key == "dino":
            from .dino import DinoDetector

            _DETECTORS[key] = DinoDetector()
        else:
            _DETECTORS[key] = Detector()
    return _DETECTORS[key]


__all__ = [
    "Detector",
    "Detection",
    "detect",
    "get_detector",
    "DEFAULT_MODEL",
    "DEFAULT_CONF",
]
