"""Pixel -> table-frame localization (Part A, §A.5).

Turns a camera pixel (a human click or a detected bounding-box centre, §0.3 #3)
into a table coordinate in the SAME frame the arm uses (§0.3 #1: origin under
the shoulder-pan axis on the table surface, +x forward, +y left, +z up, mm).

The geometry, given a calibrated camera:
    1. undistort the pixel -> a normalized ray in the camera frame,
    2. rotate that ray into the base/table frame and place it at the camera
       centre (the extrinsics),
    3. intersect it with the table plane z = 0 -> (x, y) mm.

Calibration data lives in .npz files (per §8), one pair per camera:
    * intrinsics: camera_matrix (3x3) + dist_coeffs  — the camera's optics;
      UNAFFECTED by where the camera is mounted, so they transfer between rigs.
    * extrinsics: the camera's pose in the base frame — RE-MEASURED whenever the
      camera is physically moved (``scripts/stage3_extrinsics.py``).

This module is pure geometry: it loads those files and does the ray math. It
does NOT detect tags or open cameras (that's the extrinsics script / the camera
input). cv2 + numpy are imported lazily so the package still imports without
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# --------------------------------------------------------------------------- #
# Calibration containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for one camera (pixels)."""

    camera_matrix: "object"  # np.ndarray (3, 3)
    dist_coeffs: "object"    # np.ndarray (N,)

    @staticmethod
    def load(path: str | Path) -> "CameraIntrinsics":
        """Load intrinsics from a .npz, tolerant of common key spellings.

        Accepts ``camera_matrix``/``mtx``/``K`` for the matrix and
        ``dist_coeffs``/``dist``/``D`` for the distortion vector — different
        OpenCV calibration scripts name these differently.
        """
        import numpy as np

        data = np.load(str(path))
        keys = set(data.files)

        def pick(names: tuple[str, ...], what: str):
            for n in names:
                if n in keys:
                    return np.asarray(data[n], dtype=np.float64)
            raise KeyError(
                f"{path}: no {what} found (looked for {names}; file has {sorted(keys)})"
            )

        K = pick(("camera_matrix", "mtx", "K", "intrinsics"), "camera matrix")
        D = pick(("dist_coeffs", "dist", "D", "distortion"), "distortion coeffs")
        return CameraIntrinsics(camera_matrix=K.reshape(3, 3), dist_coeffs=D.reshape(-1))


@dataclass(frozen=True)
class CameraExtrinsics:
    """Camera pose in the base/table frame: a point in the camera frame maps to
    base as ``p_base = R_cam2base @ p_cam + t_cam2base`` (t in mm)."""

    R_cam2base: "object"  # np.ndarray (3, 3)
    t_cam2base: "object"  # np.ndarray (3,) — camera centre in base frame, mm

    @staticmethod
    def load(path: str | Path) -> "CameraExtrinsics":
        import numpy as np

        data = np.load(str(path))
        R = np.asarray(data["R_cam2base"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(data["t_cam2base"], dtype=np.float64).reshape(3)
        return CameraExtrinsics(R_cam2base=R, t_cam2base=t)

    @staticmethod
    def from_solvepnp(rvec, tvec) -> "CameraExtrinsics":
        """Build from a solvePnP result whose OBJECT points were given in the
        BASE frame (so rvec/tvec map base -> camera). Inverts that to camera ->
        base. ``tvec`` is whatever unit the object points used (mm here)."""
        import cv2
        import numpy as np

        R_b2c, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        t_b2c = np.asarray(tvec, dtype=np.float64).reshape(3)
        R_c2b = R_b2c.T
        t_c2b = -R_c2b @ t_b2c  # camera centre expressed in the base frame
        return CameraExtrinsics(R_cam2base=R_c2b, t_cam2base=t_c2b)

    def save(self, path: str | Path, **extra) -> None:
        import numpy as np

        np.savez(
            str(path),
            R_cam2base=self.R_cam2base,
            t_cam2base=self.t_cam2base,
            **extra,
        )


# --------------------------------------------------------------------------- #
# The core: pixel -> table (x, y)
# --------------------------------------------------------------------------- #
def pixel_to_table(
    u: float,
    v: float,
    intr: CameraIntrinsics,
    extr: CameraExtrinsics,
    table_z_mm: float = 0.0,
) -> tuple[float, float]:
    """Map pixel ``(u, v)`` to a table coordinate ``(x_mm, y_mm)``.

    Casts the camera ray for the (undistorted) pixel and intersects it with the
    horizontal plane ``z = table_z_mm`` in the base frame. Returns the (x, y) of
    that intersection in table-frame mm. Raises if the ray is parallel to the
    table (camera looking along the horizon — should never happen for an
    overhead cam).
    """
    import cv2
    import numpy as np

    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    norm = cv2.undistortPoints(pts, intr.camera_matrix, intr.dist_coeffs)
    ray_cam = np.array([norm[0, 0, 0], norm[0, 0, 1], 1.0], dtype=np.float64)

    centre = extr.t_cam2base                  # camera centre in base (mm)
    direction = extr.R_cam2base @ ray_cam     # ray direction in base
    if abs(direction[2]) < 1e-9:
        raise ValueError("camera ray is parallel to the table plane; check extrinsics")

    lam = (table_z_mm - centre[2]) / direction[2]
    hit = centre + lam * direction
    return float(hit[0]), float(hit[1])


def load_camera(role: str, calib_dir: str | Path) -> tuple[CameraIntrinsics, CameraExtrinsics]:
    """Load the intrinsics+extrinsics pair for a camera role from ``calib_dir``.

    Expects ``intrinsics_CAM_<role>.npz`` and ``extrinsics_CAM_<role>.npz``
    (e.g. role ``"A"`` / ``"B"`` for the RIGHT / LEFT cameras, §8).
    """
    d = Path(calib_dir)
    intr = CameraIntrinsics.load(d / f"intrinsics_CAM_{role}.npz")
    extr = CameraExtrinsics.load(d / f"extrinsics_CAM_{role}.npz")
    return intr, extr
