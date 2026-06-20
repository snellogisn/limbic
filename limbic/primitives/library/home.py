"""``home`` — return the arm to its centred, neutral pose.

The home pose is the safe anchor between tasks: a known, collision-free
configuration the planner can always fall back to. Most plans open with a
``home`` (so motion starts from a predictable place) and close with one (so the
arm is parked clear of the workspace). It takes no parameters because there is a
single canonical home defined by the control layer.
"""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class Home(Primitive):
    """Move every arm joint back to the centred home configuration."""

    name = "home"
    summary = "Return the arm to its centred, neutral home pose."
    parameters: dict[str, dict[str, Any]] = {}

    def run(self, arm: RobotArm, **kwargs: Any) -> dict[str, float]:
        return arm.go_home()
