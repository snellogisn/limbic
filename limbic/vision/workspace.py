"""Workspace (gray-mat) segmentation for the world model (Part B).

The robot only acts within the gray mat. This finds that mat in a camera frame by
colour (gray = low saturation), so detections can be restricted to it — anything
off the mat or straddling its edge is not a valid target.

Design notes:
  * The mat is the largest low-saturation blob. We FILL it, so objects sitting on
    it (and the black arm occluding it — black is excluded from "gray") become
    holes that fill in as "inside" rather than carving up the workspace.
  * We then erode by a margin so a box must clear the edge (the "not touching the
    border" rule).
  * Black is intentionally excluded (val_min): the arm is black, so nothing black
    is treated as workspace — and per the rig rule, no object is black either.

cv2/numpy are imported lazily so importing limbic.vision stays cheap.
"""

from __future__ import annotations

DEFAULT_SAT_MAX = 80          # max HSV saturation counted as "gray"
DEFAULT_VAL_MIN = 25          # ignore near-black pixels (the arm, shadows)
DEFAULT_VAL_MAX = 205         # ignore blown-out highlights
DEFAULT_MARGIN_PX = 8         # shrink the mat inward (enforces "not touching the edge")
DEFAULT_MIN_AREA_FRAC = 0.05  # ignore gray blobs smaller than this fraction of the frame


def gray_mat_mask(
    frame,
    sat_max: int = DEFAULT_SAT_MAX,
    val_min: int = DEFAULT_VAL_MIN,
    val_max: int = DEFAULT_VAL_MAX,
    margin_px: int = DEFAULT_MARGIN_PX,
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
):
    """Return ``(mask, contour)`` for the gray mat, or ``(None, None)``.

    ``mask`` is a uint8 (H, W) image: 255 = inside the (margin-shrunk) workspace,
    0 = outside. ``contour`` is the mat's outline (pre-erosion) for drawing.
    """
    import cv2 as cv
    import numpy as np

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    gray = (s <= sat_max) & (v >= val_min) & (v <= val_max)
    mask = gray.astype("uint8") * 255

    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k)

    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv.contourArea)
    if cv.contourArea(c) < min_area_frac * frame.shape[0] * frame.shape[1]:
        return None, None

    filled = np.zeros(frame.shape[:2], "uint8")
    cv.drawContours(filled, [c], -1, 255, -1)        # solid mat (object/arm holes filled)
    if margin_px > 0:
        ek = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * margin_px + 1,) * 2)
        filled = cv.erode(filled, ek)
    return filled, c


def box_in_workspace(mask, x1: int, y1: int, x2: int, y2: int) -> bool:
    """True iff the whole box lies inside the workspace mask (strict — any pixel
    off the mat, or a box that leaves the frame, fails). If ``mask`` is None
    (no mat found) everything passes, so callers degrade to "no filter"."""
    if mask is None:
        return True
    h, w = mask.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
        return False
    sub = mask[y1:y2, x1:x2]
    return sub.size > 0 and not (sub == 0).any()


def highlight(frame, mask, contour):
    """Return a copy of ``frame`` with the workspace tinted/outlined and the
    off-mat area dimmed — for the visual workspace check."""
    import cv2 as cv
    import numpy as np

    out = frame.copy()
    if mask is None:
        cv.putText(out, "mat NOT found", (20, 44), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return out
    out[mask == 0] = (out[mask == 0] * 0.35).astype("uint8")          # dim outside
    green = np.full_like(out, (0, 180, 0))
    inside = mask > 0
    out[inside] = cv.addWeighted(out, 0.85, green, 0.15, 0)[inside]   # green tint inside
    cv.drawContours(out, [contour], -1, (0, 255, 255), 2)            # yellow outline
    return out
