"""Real-world object footprint from a detection box (Part B).

Turns a pixel bounding box into the object's true table-plane size in millimetres
``(long_mm, short_mm)`` — the information a grasp planner needs to know whether the
gripper can span an object and which way to approach it. Extracted from the
dual-camera viewer (``scripts/vision_detect_dual.py``) so the live pipeline
(``inputs/library/object_detections.py``) and the viewer share ONE implementation.

Two strategies, best-first:
  * :func:`object_size_mm` — segment the object out of the gray mat, fit an
    ORIENTED min-area rectangle, project its corners to the z=0 table plane. This
    drops the box padding and gives a diagonal object (e.g. a banana) its true
    width rather than its bounding-box diagonal.
  * :func:`box_edge_size_mm` — the fallback when the object can't be segmented:
    project the box's edge midpoints to z=0. Axis-aligned and includes padding,
    so it over-reads, but always works.

cv2/numpy are imported lazily so importing ``limbic.vision`` stays cheap.
"""

from __future__ import annotations

import math

from .workspace import DEFAULT_SAT_MAX, DEFAULT_VAL_MAX, DEFAULT_VAL_MIN


def box_edge_size_mm(box, intr, extr) -> tuple[float, float]:
    """Fallback footprint: project the box's edge midpoints to z=0 and measure.

    Axis-aligned and includes the box padding, so it over-reads — used only when
    the object can't be cleanly segmented out of the mat.
    """
    from limbic.control.localization import pixel_to_table

    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    lx, ly = pixel_to_table(x1, cy, intr, extr)
    rx, ry = pixel_to_table(x2, cy, intr, extr)
    tx, ty = pixel_to_table(cx, y1, intr, extr)
    bx, by = pixel_to_table(cx, y2, intr, extr)
    return (math.hypot(rx - lx, ry - ly), math.hypot(bx - tx, by - ty))


def object_size_mm(frame, box, intr, extr) -> tuple[float, float]:
    """Real object footprint ``(long_mm, short_mm)``, orientation-independent.

    Segments the object inside ``box`` (it's coloured/dark; the mat is gray =
    low-saturation), fits an oriented min-area rectangle to the largest blob, and
    projects that rect's corners to the table plane. Falls back to
    :func:`box_edge_size_mm` when the object can't be cleanly segmented.
    """
    import cv2 as cv
    import numpy as np

    from limbic.control.localization import pixel_to_table

    x1, y1, x2, y2 = box
    rx1, ry1 = max(x1, 0), max(y1, 0)
    roi = frame[ry1:y2, rx1:x2]
    if roi.size == 0:
        return box_edge_size_mm(box, intr, extr)

    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    # object = anything that is NOT the gray mat: coloured, dark, or blown out.
    obj = ((s > DEFAULT_SAT_MAX) | (v < DEFAULT_VAL_MIN) | (v > DEFAULT_VAL_MAX))
    obj = obj.astype("uint8") * 255
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    obj = cv.morphologyEx(obj, cv.MORPH_OPEN, k)

    cnts, _ = cv.findContours(obj, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return box_edge_size_mm(box, intr, extr)
    c = max(cnts, key=cv.contourArea)
    if cv.contourArea(c) < 0.10 * roi.shape[0] * roi.shape[1]:
        return box_edge_size_mm(box, intr, extr)  # too small/sparse to trust

    pts = cv.boxPoints(cv.minAreaRect(c))               # 4 corners, ROI px
    pts = pts + np.array([rx1, ry1], dtype=np.float32)  # -> full-frame px
    table = [pixel_to_table(px, py, intr, extr) for px, py in pts]
    side1 = math.hypot(table[1][0] - table[0][0], table[1][1] - table[0][1])
    side2 = math.hypot(table[2][0] - table[1][0], table[2][1] - table[1][1])
    return (max(side1, side2), min(side1, side2))
