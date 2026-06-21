"""Return the arm to its centred, neutral home pose by zeroing all joints explicitly."""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class Home(Primitive):
    name = "home"
    summary = "Return the arm to its centred, neutral home pose (all joints to zero)."
    parameters = {}

    def run(self, arm: RobotArm, **kwargs: Any) -> Any:
        arm.set_joint("shoulder_pan", 0.0)
        arm.set_joint("shoulder_lift", 0.0)
        arm.set_joint("elbow_flex", 0.0)
        arm.set_joint("wrist_flex", 0.0)
        arm.set_joint("wrist_roll", 0.0)
