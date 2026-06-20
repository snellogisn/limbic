"""A worked example of an LLM-authored plan, runnable end to end on the mock arm.

This is what the LLM brain produces for a task like *"pick up the block at
(160, 40) and place it at (160, -40)"*: a flat list of primitive calls, in the
same ``{"primitive": ..., "args": {...}}`` shape the sequence runner executes.
It doubles as a smoke test of the whole stack — run it with::

    python -m limbic.primitives.example_plan

and it drives the auto-selected mock :class:`RobotArm`, printing every primitive
and the resulting motion so you can watch the full plan flow without hardware.
"""

from __future__ import annotations

from .run_sequence import run_plan
from ..control import RobotArm


# "Pick up the block at (160, 40) and place it at (160, -40)", as primitive steps.
# Start home for a predictable origin, open the hand to be sure it's clear, pick
# (which hovers, descends into the block, closes, and lifts), place at the target
# (carry, lower, release, retreat), then park home clear of the workspace.
EXAMPLE_PLAN: list[dict] = [
    {"primitive": "home", "args": {}},
    {"primitive": "open_hand", "args": {}},
    {"primitive": "pick", "args": {"x_mm": 160, "y_mm": 40, "object_height_mm": 25}},
    {"primitive": "place", "args": {"x_mm": 160, "y_mm": -40}},
    {"primitive": "home", "args": {}},
]


def main() -> int:
    """Run the example plan against a mock arm (auto-selected with no hardware)."""
    with RobotArm() as arm:
        run_plan(arm, EXAMPLE_PLAN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
