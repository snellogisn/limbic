"""Side-by-side workspace check for both rig cameras (Part B).

Shows CAM_B (LEFT) and CAM_A (RIGHT) live, each with the gray-mat WORKSPACE
highlighted: the mat is tinted green + outlined yellow, everything off it is
dimmed. This is the visual confirmation that what we consider the workspace
matches the real mat in both views before we start filtering detections to it.

The black arm occluding the mat is fine — black isn't "gray", and the mat fill
covers such holes, so the workspace stays whole.

Usage:
    python scripts/workspace_view.py

Hotkeys:
    ESC / q : quit
"""

from __future__ import annotations

import pathlib
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from limbic.control import calibration
from limbic.platform_support import open_camera
from limbic.vision.workspace import gray_mat_mask, highlight

PANEL_W, PANEL_H = 640, 360
FULL_W, FULL_H = 1280, 720
ORDER = ["B", "A"]   # left panel = LEFT cam (B), right = RIGHT cam (A)


def main() -> None:
    caps = {}
    for r in ORDER:
        caps[r] = open_camera(calibration.CAMERAS[r]["name"], width=FULL_W, height=FULL_H)

    win = "Workspace (gray mat)  |  ESC quit"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    print("Showing the detected workspace on both cameras. ESC to quit.")

    try:
        while True:
            canvas = np.zeros((PANEL_H, PANEL_W * 2, 3), np.uint8)
            for i, role in enumerate(ORDER):
                ok, frame = caps[role].read()
                if not ok or frame is None:
                    frame = np.zeros((FULL_H, FULL_W, 3), np.uint8)
                    cv.putText(frame, f"CAM_{role}: no frame", (40, 80),
                               cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    mask, contour = gray_mat_mask(frame)
                    frame = highlight(frame, mask, contour)
                disp = cv.resize(frame, (PANEL_W, PANEL_H))
                canvas[0:PANEL_H, i * PANEL_W:(i + 1) * PANEL_W] = disp
                side = calibration.CAMERAS[role]["side"]
                label = f"CAM_{role}  {side}"
                cv.putText(canvas, label, (i * PANEL_W + 12, 28),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv.putText(canvas, label, (i * PANEL_W + 12, 28),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv.line(canvas, (PANEL_W, 0), (PANEL_W, PANEL_H), (60, 60, 60), 1)

            cv.imshow(win, canvas)
            if (cv.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        for c in caps.values():
            c.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
