"""Throw a held object forwards (+x): wind back, swing forward-up fast, and release."""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class ThrowForward(Primitive):
    name = "throw_forward"
    summary = "Fling a currently-held object forwards by swinging forward-up and releasing the gripper."
    parameters = {
        "release_x_mm": {"type": "float", "description": "Forward x position to release the object at (mm).", "default": 230.0},
        "release_y_mm": {"type": "float", "description": "Lateral y position of the throw (mm).", "default": 0.0},
        "windup_x_mm": {"type": "float", "description": "Forward x of the wind-up (back) position (mm).", "default": 120.0},
        "windup_z_mm": {"type": "float", "description": "Height of the wind-up position (mm).", "default": 80.0},
        "release_z_mm": {"type": "float", "description": "Height to release at, after the upward swing (mm).", "default": 170.0},
    }

    def run(self, arm: RobotArm, **kwargs: Any) -> Any:
        release_x = float(kwargs.get("release_x_mm", 230.0))
        release_y = float(kwargs.get("release_y_mm", 0.0))
        windup_x = float(kwargs.get("windup_x_mm", 120.0))
        windup_z = float(kwargs.get("windup_z_mm", 80.0))
        release_z = float(kwargs.get("release_z_mm", 170.0))

        # Wind up: pull back and low, tool pointing down/forward.
        arm.move_to_xyz(windup_x, release_y, windup_z)
        # Fast forward-and-up swing to build momentum in +x.
        arm.move_to_xyz(release_x, release_y, release_z)
        # Release at the top of the forward swing so the object flies forward.
        arm.open_gripper()
