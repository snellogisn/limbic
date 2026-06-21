"""Open-vocabulary object detection with YOLO-World (Part B, §B.1-B.3).

The seam (§0.3 #3): ``detect(frame, prompt) -> [(label, (u, v)), ...]`` where
``(u, v)`` is each object's bounding-box CENTRE pixel — the same input a human
click gives, so vision drops straight into Part A's pixel->table path.

YOLO-World is OPEN-VOCABULARY: you set the wanted class names at runtime
(``set_classes``) and it detects them with no retraining — so the Brain can ask
for arbitrary object names. ``torch`` + ``ultralytics`` are the optional
``vision`` extra (§0.4); they're imported lazily inside the methods so importing
this module never fails on a torch-less machine — the failure (with an install
hint) only happens if you actually build a Detector.

Tune at the rig (§B.4): the confidence threshold and the exact prompt wording
matter a lot for YOLO-World; defaults here are a starting point, not gospel.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

# Bundled offline weights (this network has an SSL-intercepting proxy, so
# ultralytics' auto-download fails — see weights/README). The repo ships the
# YOLO-World checkpoint and the CLIP text encoder under <repo>/weights/.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BUNDLED_DIR = _REPO_ROOT / "weights"
_BUNDLED_MODEL = _BUNDLED_DIR / "yolov8s-world.pt"

# YOLO-World checkpoint. 's' = small (fast, good default for a live feed); bump to
# 'yolov8x-worldv2.pt' for accuracy. Falls back to the bare name (auto-download)
# only if the bundled file is absent.
DEFAULT_MODEL = str(_BUNDLED_MODEL) if _BUNDLED_MODEL.exists() else "yolov8s-world.pt"
# YOLO-World tends to score lower than closed-set YOLO; start permissive and
# raise at the rig until false positives disappear (§B.4).
DEFAULT_CONF = 0.10


@dataclass(frozen=True)
class Detection:
    """One detected object in image space."""

    label: str
    confidence: float
    box: tuple[float, float, float, float]   # (x1, y1, x2, y2) in pixels

    @property
    def center(self) -> tuple[float, float]:
        """The bounding-box centre pixel — the §0.3 #3 seam point."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _use_bundled_clip() -> None:
    """Point ultralytics' CLIP loader at the bundled ``weights/clip`` dir.

    ``set_classes`` loads the CLIP text encoder from ultralytics' ``WEIGHTS_DIR /
    'clip'`` (a CWD-relative ``weights`` by default). We override that module
    global to our absolute bundled dir so detection works regardless of the
    working directory and without hitting the (proxy-blocked) download. No-op if
    the bundled CLIP file isn't present.
    """
    if not (_BUNDLED_DIR / "clip" / "ViT-B-32.pt").exists():
        return
    try:
        import ultralytics.nn.text_model as _tm

        _tm.WEIGHTS_DIR = _BUNDLED_DIR
    except Exception:
        pass


def _as_class_list(prompt: "str | list[str]") -> list[str]:
    """Normalise a prompt to a list of class names.

    A string is split on commas (``"red cup, blue block"`` -> two classes); a
    list/tuple is taken as-is. Open-vocabulary, so any phrase is valid.
    """
    if isinstance(prompt, str):
        return [p.strip() for p in prompt.split(",") if p.strip()]
    return [str(p).strip() for p in prompt if str(p).strip()]


class Detector:
    """A YOLO-World detector. Construct once, reuse across frames.

    ``set_classes`` re-encodes the prompt through the model's text encoder, which
    is not free — so we cache the current class list and only re-encode when it
    actually changes (important for a live feed calling ``detect`` every frame).
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: "str | int | None" = None):
        from ultralytics import YOLO  # lazy: optional 'vision' extra (§0.4)

        _use_bundled_clip()
        self.model = YOLO(model)
        self._classes: list[str] | None = None
        if device is None:
            try:
                import torch

                device = 0 if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

    def set_classes(self, prompt: "str | list[str]") -> list[str]:
        """Set the open-vocabulary class list (no-op if unchanged)."""
        names = _as_class_list(prompt)
        if not names:
            raise ValueError("empty prompt — give at least one object name")
        if names != self._classes:
            # Re-encoding the prompt must happen with the model on CPU: after an
            # inference the model lives on the GPU, but the new text tokens are
            # built on CPU -> a CUDA device-mismatch. Move to CPU for the (cheap)
            # re-encode; the next predict(device=...) re-homes the model. Without
            # this, changing classes at runtime crashes (only the first set works).
            try:
                self.model.to("cpu")
            except Exception:
                pass
            self.model.set_classes(names)
            self._classes = names
        return names

    def detect_boxes(
        self,
        frame,
        prompt: "str | list[str] | None" = None,
        conf: float = DEFAULT_CONF,
        max_det: int = 50,
    ) -> list[Detection]:
        """Detect objects, returning full ``Detection`` records (with boxes).

        ``frame`` is a BGR image (numpy array, as OpenCV gives). If ``prompt`` is
        given it (re)sets the classes; otherwise the last-set classes are used.
        """
        classes = self.set_classes(prompt) if prompt is not None else self._classes
        if classes is None:
            raise ValueError("no classes set — pass prompt= or call set_classes() first")

        results = self.model.predict(
            frame, conf=conf, max_det=max_det, device=self.device, verbose=False
        )
        dets: list[Detection] = []
        if not results:
            return dets
        for b in results[0].boxes:
            idx = int(b.cls[0])
            label = classes[idx] if 0 <= idx < len(classes) else str(idx)
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            dets.append(Detection(label=label, confidence=float(b.conf[0]),
                                  box=(x1, y1, x2, y2)))
        return dets

    def detect(
        self,
        frame,
        prompt: "str | list[str] | None" = None,
        conf: float = DEFAULT_CONF,
    ) -> list[tuple[str, tuple[float, float]]]:
        """The §0.3 #3 seam: ``[(label, (u, v)), ...]`` (bbox-centre pixels)."""
        return [(d.label, d.center) for d in self.detect_boxes(frame, prompt, conf)]


# --------------------------------------------------------------------------- #
# Module-level convenience: a lazily-built shared detector.
# --------------------------------------------------------------------------- #
_default_detector: Detector | None = None


def detect(
    frame,
    prompt: "str | list[str]",
    conf: float = DEFAULT_CONF,
) -> list[tuple[str, tuple[float, float]]]:
    """One-shot detection with a shared default Detector (§0.3 #3 seam).

    Builds the model on first call and reuses it after. For tight loops prefer
    constructing your own :class:`Detector` so you control its lifetime.
    """
    global _default_detector
    if _default_detector is None:
        _default_detector = Detector()
    return _default_detector.detect(frame, prompt, conf)
