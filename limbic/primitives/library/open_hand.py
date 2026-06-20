"""``open_hand`` — open the gripper fully.

Releasing or preparing to grasp both start the same way: get the claw out of the
way by opening it. Kept as its own primitive (rather than folding it into
pick/place) so the planner can sequence gripper actions explicitly when a task
doesn't fit the standard composites.
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class OpenHand(Primitive):
    """Open the gripper fully."""

    name = "open_hand"
    summary = "Open the gripper fully."
    parameters: dict[str, dict[str, Any]] = {}

    def run(self, arm: RobotArm, **kwargs: Any) -> None:
        arm.open_gripper()
