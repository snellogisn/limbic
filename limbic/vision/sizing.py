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


def box_edge_size_mm(box, intr, extr, plane_z: float = 0.0) -> tuple[float, float]:
    """Fallback footprint: project the box's edge midpoints to ``plane_z`` and measure.

    Axis-aligned and includes the box padding, so it over-reads — used only when
    the object can't be cleanly segmented out of the mat. ``plane_z`` is the height
    plane to project onto (the object's measured top, when known; see
    :func:`mono_footprint`); defaults to the z=0 table plane.
    """
    from limbic.control.localization import pixel_to_table

    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    lx, ly = pixel_to_table(x1, cy, intr, extr, table_z_mm=plane_z)
    rx, ry = pixel_to_table(x2, cy, intr, extr, table_z_mm=plane_z)
    tx, ty = pixel_to_table(cx, y1, intr, extr, table_z_mm=plane_z)
    bx, by = pixel_to_table(cx, y2, intr, extr, table_z_mm=plane_z)
    return (math.hypot(rx - lx, ry - ly), math.hypot(bx - tx, by - ty))


def object_size_mm(frame, box, intr, extr, plane_z: float = 0.0) -> tuple[float, float]:
    """Real object footprint ``(long_mm, short_mm)``, orientation-independent.

    Segments the object inside ``box`` (it's coloured/dark; the mat is gray =
    low-saturation), fits an oriented min-area rectangle to the largest blob, and
    projects that rect's corners to the plane ``z = plane_z``. Falls back to
    :func:`box_edge_size_mm` when the object can't be cleanly segmented.

    ``plane_z`` matters for accuracy: the silhouette is the object's elevated TOP
    and the cameras look obliquely (§A.7), so projecting to ``z=0`` ray-casts the
    outline outward and over-reads. Pass the object's measured top height (from
    :func:`mono_footprint` or stereo :func:`triangulate`) to project onto its own
    plane — for a vertical-sided object the top face ≈ the base footprint.
    """
    import cv2 as cv
    import numpy as np

    from limbic.control.localization import pixel_to_table

    # Detection boxes are floats (Grounding DINO); numpy slicing needs ints.
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    rx1, ry1 = max(x1, 0), max(y1, 0)
    roi = frame[ry1:y2, rx1:x2]
    if roi.size == 0:
        return box_edge_size_mm(box, intr, extr, plane_z)

    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    # object = anything that is NOT the gray mat: coloured, dark, or blown out.
    obj = ((s > DEFAULT_SAT_MAX) | (v < DEFAULT_VAL_MIN) | (v > DEFAULT_VAL_MAX))
    obj = obj.astype("uint8") * 255
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    obj = cv.morphologyEx(obj, cv.MORPH_OPEN, k)

    cnts, _ = cv.findContours(obj, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return box_edge_size_mm(box, intr, extr, plane_z)
    c = max(cnts, key=cv.contourArea)
    if cv.contourArea(c) < 0.10 * roi.shape[0] * roi.shape[1]:
        return box_edge_size_mm(box, intr, extr, plane_z)  # too small/sparse to trust

    pts = cv.boxPoints(cv.minAreaRect(c))               # 4 corners, ROI px
    pts = pts + np.array([rx1, ry1], dtype=np.float32)  # -> full-frame px
    table = [pixel_to_table(px, py, intr, extr, table_z_mm=plane_z) for px, py in pts]
    side1 = math.hypot(table[1][0] - table[0][0], table[1][1] - table[0][1])
    side2 = math.hypot(table[2][0] - table[1][0], table[2][1] - table[1][1])
    return (max(side1, side2), min(side1, side2))


# --------------------------------------------------------------------------- #
# Elevation correction — fixing the z=0 over-read, with object HEIGHT.
#
# object_size_mm projects the object's silhouette to a fixed table plane, but the
# silhouette is the object's ELEVATED top and the cameras look obliquely (§A.7),
# so projecting to z=0 ray-casts the outline OUTWARD: the footprint over-reads and
# the position shifts away from the camera (~±10 mm). The fix needs the object's
# height, then we project the footprint onto that plane instead of z=0.
#
# Two ways to get height:
#   * STEREO (best): triangulate() the two cameras' box-centre rays -> a
#     parallax-free (x, y, z). Used by the dual viewer's fuse_3d.
#   * MONO (one camera): mono_footprint() estimates height from a single view.
#     Parallax acts RADIALLY from the camera nadir, so the TANGENTIAL width is
#     ~unaffected while the RADIAL depth inflates with height; comparing the true
#     base contact (z=0) to the apparent top centre recovers height. The far base
#     edge is self-occluded, so this is UNDER-CONSTRAINED without a prior — closed
#     with "radial depth ≈ tangential width" and refined iteratively. It's an
#     ESTIMATE; prefer stereo when both cameras see the object.
# --------------------------------------------------------------------------- #

def camera_nadir(extr):
    """``(nadir_xy, height_mm)``: the camera centre's table (x, y) and its height."""
    import numpy as np

    t = np.asarray(extr.t_cam2base, dtype=np.float64)
    return t[:2], float(t[2])


def pixel_ray(u, v, intr, extr):
    """Back-project a pixel to a 3D ray in the BASE frame: ``(origin, direction)``.

    Origin = camera centre; direction = the (un-normalised) ray through the
    undistorted pixel. The same geometry :func:`pixel_to_table` uses, kept as a ray
    so two cameras' rays can be triangulated instead of hitting a fixed plane.
    """
    import cv2 as cv
    import numpy as np

    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    norm = cv.undistortPoints(pts, intr.camera_matrix, intr.dist_coeffs)
    ray_cam = np.array([norm[0, 0, 0], norm[0, 0, 1], 1.0], dtype=np.float64)
    origin = np.asarray(extr.t_cam2base, dtype=np.float64)
    direction = np.asarray(extr.R_cam2base, dtype=np.float64) @ ray_cam
    return origin, direction


def triangulate(o_a, d_a, o_b, d_b):
    """Closest approach of two 3D rays -> ``(midpoint, residual_mm)``.

    The midpoint is the best 3D estimate of the point both rays pass through; the
    residual (their gap at closest approach) is a quality flag — large means the
    rays don't really meet, so the cross-camera match or a calibration is off.
    Returns ``(None, inf)`` for parallel rays.
    """
    import numpy as np

    r = o_a - o_b
    a = float(d_a @ d_a); b = float(d_a @ d_b); c = float(d_b @ d_b)
    d = float(d_a @ r);   e = float(d_b @ r)
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        return None, float("inf")
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    p_a = o_a + s * d_a
    p_b = o_b + t * d_b
    return (p_a + p_b) / 2.0, float(np.linalg.norm(p_a - p_b))


def _segment_contour(frame, box):
    """Largest object contour inside ``box`` in FULL-FRAME px (or None).

    Object = not the gray mat (coloured / dark / blown out) — the same segmentation
    :func:`object_size_mm` uses, shared so the size and the height estimate agree on
    one silhouette.
    """
    import cv2 as cv
    import numpy as np

    x1, y1, x2, y2 = box
    rx1, ry1 = max(int(x1), 0), max(int(y1), 0)
    roi = frame[ry1:int(y2), rx1:int(x2)]
    if roi.size == 0:
        return None
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    obj = ((s > DEFAULT_SAT_MAX) | (v < DEFAULT_VAL_MIN) | (v > DEFAULT_VAL_MAX))
    obj = obj.astype("uint8") * 255
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    obj = cv.morphologyEx(obj, cv.MORPH_OPEN, k)
    cnts, _ = cv.findContours(obj, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv.contourArea)
    if cv.contourArea(c) < 0.10 * roi.shape[0] * roi.shape[1]:
        return None
    return c.reshape(-1, 2).astype(np.float64) + np.array([rx1, ry1], dtype=np.float64)


def _project_contour(contour_px, intr, extr, plane_z):
    import numpy as np

    from limbic.control.localization import pixel_to_table

    return np.array([pixel_to_table(px, py, intr, extr, table_z_mm=plane_z)
                     for px, py in contour_px], dtype=np.float64)


def mono_footprint(frame, box, intr, extr, mode: str = "auto", auto_min_mm: float = 8.0):
    """SINGLE-camera footprint + height, correcting the z=0 elevation inflation.

    Returns ``(long_mm, short_mm, height_mm_or_None, (cx, cy))``. Modes:
      ``"z0"``      no correction (baseline; over-reads tall objects), height None.
      ``"contour"`` trust the segmented silhouette on z=0, anchor the centre on the
                    base contour. Accurate when the base perimeter is visible
                    (short objects); height still reported for reference.
      ``"height"``  estimate height from the image (radial-parallax vs. tangential
                    width) and re-project footprint + centre onto that plane;
                    iterates the depth closure to refine.
      ``"auto"``    "height" when the estimate is non-trivial (>= ``auto_min_mm``),
                    else "contour" — i.e. correct only when elevation matters.

    Height is an ESTIMATE (the far base is self-occluded; see the module note).
    Stereo :func:`triangulate` is more reliable when both cameras see the object.
    """
    import numpy as np

    from limbic.control.localization import pixel_to_table

    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    centre0 = pixel_to_table(cx, cy, intr, extr)
    if mode == "z0":
        return (*object_size_mm(frame, box, intr, extr, 0.0), None, centre0)

    contour = _segment_contour(frame, box)
    if contour is None:                       # can't segment -> no correction possible
        return (*box_edge_size_mm(box, intr, extr, 0.0), None, centre0)

    N, H = camera_nadir(extr)
    T0 = _project_contour(contour, intr, extr, 0.0)
    d = T0 - N
    r = np.hypot(d[:, 0], d[:, 1])
    if r.max() < 1e-6:                         # object directly under the camera
        return (*object_size_mm(frame, box, intr, extr, 0.0), 0.0, centre0)
    u = d[np.argmax(r)] / r.max()             # radial unit, outward from nadir
    rr0 = d @ u                               # radial coord of each contour pt
    tt = d @ np.array([-u[1], u[0]])          # tangential coord (parallax-safe)
    contact = T0[rr0 <= np.percentile(rr0, 25)].mean(axis=0)   # true near base, z=0
    r_contact = float((contact - N) @ u)
    top = np.asarray(pixel_to_table(cx, cy, intr, extr, table_z_mm=0.0))  # apparent top
    r_app = float((top - N) @ u)

    def solve_h(depth_mm: float) -> float:
        # true object centre radius = near contact + half the radial depth; the
        # apparent (top) radius is inflated by H/(H-h). Invert for h.
        r_centre = r_contact + 0.5 * depth_mm
        if r_centre <= 1e-6 or r_app <= r_centre:
            return 0.0
        k = r_app / r_centre                  # = H / (H - h)
        return float(max(0.0, min(H * (1.0 - 1.0 / k), 0.8 * H)))

    depth = float(tt.max() - tt.min())        # closure #0: tangential width
    h = solve_h(depth)
    for _ in range(2):                        # refine: depth = corrected radial extent
        rr_h = (_project_contour(contour, intr, extr, h) - N) @ u
        h = solve_h(float(rr_h.max() - rr_h.min()))

    if mode == "contour" or (mode == "auto" and h < auto_min_mm):
        base_c = T0.mean(axis=0)
        return (*object_size_mm(frame, box, intr, extr, 0.0), h,
                (float(base_c[0]), float(base_c[1])))

    long_, short_ = object_size_mm(frame, box, intr, extr, h)
    return (long_, short_, h, pixel_to_table(cx, cy, intr, extr, table_z_mm=h))
