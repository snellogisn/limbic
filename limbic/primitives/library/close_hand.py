"""``close_hand`` — close the gripper to its grip position.

The grasp action: close the claw onto an object. The control layer drives to a
calibrated "closed enough to grip" position (not crushing) and waits for the
claw to fully actuate before returning, so the caller can rely on the grip being
established once this primitive completes.
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class CloseHand(Primitive):
    """Close the gripper to its calibrated grip position."""

    name = "close_hand"
    summary = "Close the gripper to grip an object."
    parameters: dict[str, dict[str, Any]] = {}

    def run(self, arm: RobotArm, **kwargs: Any) -> None:
        arm.close_gripper()
