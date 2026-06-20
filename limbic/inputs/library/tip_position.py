"""Sensory input: current tool-tip position in the table frame (mm).

The LLM needs Cartesian position to reason about where the hand actually is
in space — for example, to decide whether the tip is above the right spot
before descending, or to verify that a move landed close enough to the
intended target. This is a higher-level view than ``joint_state``: rather than
individual joint angles, it gives a single (x, y, z) point the planner can
compare directly to a Cartesian target.

Coordinates are in the table frame: origin under the pan-axis base plate,
+x forward (away from the robot), +y leftward, +z upward. Units are mm.
The position is computed via the forward-kinematics model from the current
joint readings — no external sensor is required.

Runtime context
---------------
``arm`` is injected by the registry (see ``registry.set_context``); the LLM
never passes it explicitly, which is why it does not appear in ``parameters``.
"""

from __future__ import annotations

from typing import Any

from limbic.inputs.base import Input


class TipPosition(Input):
    """Where the tool-tip currently is in the table frame (x_mm, y_mm, z_mm)."""

    name = "tip_position"
    summary = "Current tool-tip position in the table frame: {x_mm, y_mm, z_mm} in mm."
    # No LLM-facing parameters — the arm is injected by the registry.
    parameters: dict[str, dict[str, Any]] = {}

    def read(self, *, arm: Any, **kwargs: Any) -> dict[str, float]:
        """Return the tool-tip Cartesian position in table-frame mm.

        Args:
            arm: Injected by the registry; a connected ``RobotArm`` instance.

        Returns:
            A dict with keys ``x_mm``, ``y_mm``, and ``z_mm`` representing the
            tip's position in mm relative to the base origin.  The values are
            derived from the current joint readings via forward kinematics.
        """
        x_mm, y_mm, z_mm = arm.current_xyz()
        return {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm}
