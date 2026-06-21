"""limbic.vision — open-vocabulary object detection (Part B, the world model).

The Senses' eyes: given a camera frame + an object name/prompt, find where each
named object is in the image. Exposes the §0.3 #3 seam

    detect(frame, prompt) -> [(label, (u, v)), ...]

where (u, v) is the bounding-box CENTRE pixel — exactly the "click" the arm side
(Part A localization) turns into a table coordinate, so vision is a drop-in
replacement for a human click.

Backed by YOLO-World (ultralytics + torch), an OPTIONAL 'vision' extra (§0.4):
the imports are lazy so the base limbic package still imports on machines without
torch (e.g. ARM64). Import this subpackage explicitly to use it.
"""

from .detector import DEFAULT_CONF, DEFAULT_MODEL, Detection, Detector, detect

__all__ = ["Detector", "Detection", "detect", "DEFAULT_MODEL", "DEFAULT_CONF"]
