"""Sensory input: gripper opening percentage and a human-readable state label.

The LLM uses this input when it needs to reason about whether the hand is
holding something. Raw joint angles alone are not intuitive for that purpose;
this input translates the gripper's 0..100 position into a structured dict
with both the numeric value and a categorical label so the planner can write
conditions like ``gripper_state["state"] == "closed"`` without magic numbers.

Threshold rationale
-------------------
The gripper servo is driven on a 0..100 scale where 0 means fully closed and
100 means fully open. In practice, "closed enough to hold an object" is not
exactly 0 — mechanical compliance and object thickness mean the servo often
reads ~10-30 when gripping. The thresholds below are deliberately generous:

    * ``<= 20``  -> "closed"  (gripper is gripping or fully shut)
    * ``>= 80``  -> "open"    (gripper is clear of any object)
    * in-between -> "partial" (transition / mid-open / grasping large object)

These can be tuned later without changing the interface the planner uses.

Runtime context
---------------
``arm`` is injected by the registry (see ``registry.set_context``); the LLM
never passes it explicitly, which is why it does not appear in ``parameters``.
"""

from __future__ import annotations

from typing import Any

from limbic.inputs.base import Input

# Thresholds for the categorical state labels (see module docstring).
_CLOSED_THRESHOLD = 20.0   # percent_open <= this -> "closed"
_OPEN_THRESHOLD = 80.0     # percent_open >= this -> "open"


class GripperState(Input):
    """Gripper opening percentage and a closed/partial/open state label."""

    name = "gripper_state"
    summary = (
        "Gripper opening as a percent (0=closed, 100=open) and a "
        "categorical state: 'closed', 'partial', or 'open'. "
        "Use this to infer whether the hand is holding an object."
    )
    # No LLM-facing parameters — the arm is injected by the registry.
    parameters: dict[str, dict[str, Any]] = {}

    def read(self, *, arm: Any, **kwargs: Any) -> dict[str, Any]:
        """Return the gripper opening percentage and a categorical state label.

        Args:
            arm: Injected by the registry; a connected ``RobotArm`` instance.

        Returns:
            A dict with:
                ``percent_open`` (float): Current gripper reading on 0..100
                    (0 = fully closed, 100 = fully open).
                ``state`` (str): One of ``"closed"``, ``"partial"``, or
                    ``"open"`` based on the thresholds defined in this module.
        """
        joints = arm.read_joints()
        percent_open: float = float(joints["gripper"])

        if percent_open <= _CLOSED_THRESHOLD:
            state = "closed"
        elif percent_open >= _OPEN_THRESHOLD:
            state = "open"
        else:
            state = "partial"

        return {"percent_open": percent_open, "state": state}
