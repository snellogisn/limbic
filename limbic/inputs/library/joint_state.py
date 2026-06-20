"""Sensory input: current angle of every joint/motor on the arm.

The LLM needs joint state whenever it reasons about the arm's posture — for
example, confirming a move completed, checking whether a joint is near its
limit, or diagnosing why an expected pose wasn't reached. This is the
lowest-level motor snapshot: raw joint angles as reported by the hardware
(or mock) backend.

The gripper is included in the dict, expressed on the same 0..100 scale used
everywhere else in limbic (0 = fully closed, 100 = fully open). The LLM can
read this input without any parameters; it simply asks and gets the current
snapshot.

Runtime context
---------------
``arm`` is injected by the registry (see ``registry.set_context``); the LLM
never passes it explicitly, which is why it does not appear in ``parameters``.
"""

from __future__ import annotations

from typing import Any

from limbic.inputs.base import Input


class JointState(Input):
    """Current angle of every motor/joint (degrees; gripper 0..100)."""

    name = "joint_state"
    summary = "Current angle of every motor/joint (degrees; gripper 0..100)."
    # No LLM-facing parameters — the arm is injected by the registry.
    parameters: dict[str, dict[str, Any]] = {}

    def read(self, *, arm: Any, **kwargs: Any) -> dict[str, float]:
        """Return ``{joint_name: degrees}`` for all joints including the gripper.

        Args:
            arm: Injected by the registry; a connected ``RobotArm`` instance.

        Returns:
            A dict mapping each joint name to its current angle in degrees
            (gripper on a 0..100 scale where 0 is closed and 100 is open).
        """
        return arm.read_joints()
