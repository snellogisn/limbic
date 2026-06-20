"""Sensory input: capture one frame from a webcam and return a summary dict.

This replaces the original Windows-only DirectShow / pygrabber camera access
(which used ``cv2.CAP_DSHOW`` and ``pygrabber.dshow_graph``) with the
cross-platform ``open_camera`` helper from ``limbic.platform_support``. That
helper selects the right OpenCV backend automatically — AVFoundation on macOS,
DirectShow on Windows, V4L2 on Linux — so the same code runs everywhere.

What this input returns
-----------------------
Raw pixel arrays are not JSON-serialisable, so we return a summary dict:
    * ``ok``              -- True on success, False on any failure
    * ``camera_index``    -- which camera was used
    * ``width``, ``height`` -- actual frame dimensions (after driver negotiation)
    * ``mean_brightness`` -- mean pixel value across all channels (0.0..255.0),
                            useful for detecting occlusion or lighting problems
    * ``saved_to``        -- path where the PNG was written, or None

On failure the dict is ``{"ok": False, "error": "<helpful message>"}``.

Extensibility
-------------
This is intentionally thin — it proves the camera works and returns a scalar
summary the LLM can reason about. Future inputs can add richer computer-vision
outputs (object detection bounding boxes, AprilTag poses for workspace
localisation, colour histograms, depth maps, …) by dropping new files in this
``library/`` directory. Each new input is discovered automatically by the
registry; no existing file needs to change.

Graceful degradation
--------------------
``opencv-python`` is an optional dependency. The import is deferred to
``read()`` so that the limbic package imports cleanly on machines that don't
have cv2 installed. If cv2 is missing, or if the camera cannot be opened, or
if no frame is returned, a structured error dict is returned — this input
never raises.
"""

from __future__ import annotations

from typing import Any

from limbic.inputs.base import Input


class Camera(Input):
    """Capture one frame from a webcam and return a summary (not raw pixels)."""

    name = "camera"
    summary = (
        "Capture a single webcam frame. Returns brightness and optionally "
        "saves a PNG. Use to verify the camera works or check lighting."
    )
    parameters: dict[str, dict[str, Any]] = {
        "camera_index": {
            "type": "integer",
            "description": "Camera device index (0 = default/built-in webcam).",
            "default": 0,
        },
        "width": {
            "type": "integer",
            "description": (
                "Requested capture width in pixels. The driver picks the "
                "nearest supported resolution; read ``width`` in the response "
                "for the actual value. Omit to use the camera default."
            ),
            "default": None,
        },
        "height": {
            "type": "integer",
            "description": (
                "Requested capture height in pixels. Same negotiation as "
                "``width``. Omit to use the camera default."
            ),
            "default": None,
        },
        "save_path": {
            "type": "string",
            "description": (
                "Filesystem path where the captured frame should be written as "
                "a PNG (e.g. ``/tmp/frame.png``). If omitted or null the frame "
                "is not saved to disk."
            ),
            "default": None,
        },
    }

    def read(
        self,
        *,
        camera_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Capture one frame and return a JSON-serialisable summary.

        Imports ``cv2`` lazily so that a missing ``opencv-python`` install is
        caught here rather than at module load time. Opens the camera via
        ``limbic.platform_support.open_camera`` (which picks the right backend
        per OS) and releases it in a ``finally`` block regardless of outcome.

        Args:
            camera_index: Index of the camera to open (0 = first/default).
            width:        Requested frame width in pixels (driver may differ).
            height:       Requested frame height in pixels (driver may differ).
            save_path:    If given, write the captured frame as a PNG here.

        Returns:
            On success::

                {
                    "ok": True,
                    "camera_index": <int>,
                    "width": <int>,          # actual width after driver negotiation
                    "height": <int>,         # actual height after driver negotiation
                    "mean_brightness": <float>,  # mean pixel value 0.0..255.0
                    "saved_to": <str | None>,    # path written, or None
                }

            On failure::

                {"ok": False, "error": "<human-readable message>"}
        """
        # Lazily import cv2 so a missing install never breaks the module load.
        try:
            import cv2  # type: ignore
        except ImportError:
            return {
                "ok": False,
                "error": (
                    "opencv-python is not installed. "
                    "Install it with: pip install opencv-python"
                ),
            }

        # Lazily import open_camera here too — avoids any top-level side-effects.
        from limbic.platform_support import open_camera

        cap = None
        try:
            # open_camera selects the right OS backend and applies width/height.
            try:
                cap = open_camera(camera_index, width=width, height=height)
            except (ImportError, RuntimeError) as exc:
                return {"ok": False, "error": str(exc)}

            # Read a single frame from the capture device.
            frame_ok, frame = cap.read()
            if not frame_ok or frame is None:
                return {
                    "ok": False,
                    "error": (
                        f"Camera index {camera_index} opened but returned no "
                        "frame. The camera may be in use by another application."
                    ),
                }

            # Read back the actual negotiated resolution from the capture object
            # (the driver may have chosen a different size than requested).
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Compute mean brightness: average pixel value across all channels.
            # This is a cheap scalar summary the LLM can use to detect a covered
            # lens (near 0) or blown-out exposure (near 255).
            mean_brightness: float = float(frame.mean())

            # Optionally persist the frame as a PNG for downstream inspection.
            written_path: str | None = None
            if save_path is not None:
                success = cv2.imwrite(save_path, frame)
                if success:
                    written_path = save_path
                else:
                    # Non-fatal: report the failure in the path field so the
                    # caller knows the save didn't work without raising.
                    written_path = None

            return {
                "ok": True,
                "camera_index": camera_index,
                "width": actual_width,
                "height": actual_height,
                "mean_brightness": mean_brightness,
                "saved_to": written_path,
            }

        finally:
            # Always release the capture so the OS gives the camera back to
            # other applications.  Done unconditionally, even on exceptions.
            if cap is not None:
                cap.release()
