"""The sequence runner: execute an LLM-authored plan of primitive calls.

This is the bridge between "thinking" and "moving". The LLM brain browses the
primitive catalog (``registry.catalog()``), decides how to accomplish a task,
and **compiles a plan** — an ordered list of primitive calls, each a step of the
form ``{"primitive": <name>, "args": {...}}``. The plan is just data: the LLM can
emit it as an in-memory Python list, or write it to a JSON file. This module
takes such a plan and executes it against a :class:`RobotArm` in one go,
resolving each primitive by name through the registry and feeding it the step's
arguments.

Keeping execution separate from planning means the same plan can be reviewed,
saved, replayed, or hand-edited before it ever drives a motor, and a plan
produced as JSON by the LLM runs identically to one written in Python.

    from limbic.control import RobotArm
    from limbic.primitives.run_sequence import run_plan

    plan = [
        {"primitive": "home", "args": {}},
        {"primitive": "pick", "args": {"x_mm": 160, "y_mm": 40}},
        {"primitive": "place", "args": {"x_mm": 160, "y_mm": -40}},
    ]
    with RobotArm() as arm:
        run_plan(arm, plan)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from . import registry
from ..control import RobotArm


@dataclass
class Step:
    """One entry in a plan: a primitive to invoke plus its keyword arguments."""

    primitive: str
    args: dict[str, Any] = field(default_factory=dict)


def _as_step(raw: Step | dict[str, Any], index: int) -> Step:
    """Normalise a plan entry (``Step`` or plain dict) into a ``Step``.

    Raises a clear, step-indexed error if the entry is malformed so a bad plan
    fails loudly before any motion happens.
    """
    if isinstance(raw, Step):
        return raw
    if not isinstance(raw, dict) or "primitive" not in raw:
        raise ValueError(
            f"plan step {index} is malformed: expected a Step or a dict with a "
            f"'primitive' key, got {raw!r}"
        )
    return Step(primitive=raw["primitive"], args=dict(raw.get("args", {})))


def run_plan(
    arm: RobotArm,
    plan: list[Step | dict[str, Any]],
    verbose: bool = True,
) -> list[Any]:
    """Execute ``plan`` step by step against ``arm`` and return each step's result.

    Each step is looked up in the registry by name and called with its args. A
    numbered progress line is printed per step when ``verbose``. If a step names
    an unknown primitive or omits a required argument, execution stops with a
    message identifying the offending step index.
    """
    steps = [_as_step(raw, i) for i, raw in enumerate(plan)]
    total = len(steps)
    results: list[Any] = []

    for index, step in enumerate(steps):
        if verbose:
            arg_text = ", ".join(f"{k}={v}" for k, v in step.args.items())
            print(f"[{index + 1}/{total}] {step.primitive}({arg_text})")

        try:
            primitive = registry.get(step.primitive)
        except KeyError as exc:
            raise ValueError(f"plan step {index} ({step.primitive!r}): {exc}") from exc

        try:
            result = primitive(arm, **step.args)
        except TypeError as exc:
            # Surfaces the base Primitive's "missing required argument(s)" error,
            # tagged with the step index so the bad plan entry is obvious.
            raise ValueError(f"plan step {index} ({step.primitive!r}): {exc}") from exc

        results.append(result)

    return results


def run_plan_file(arm: RobotArm, path: str, verbose: bool = True) -> list[Any]:
    """Load a JSON plan (a list of steps) from ``path`` and run it against ``arm``.

    This is how an LLM-authored plan saved to disk gets executed: the JSON is a
    list of ``{"primitive": ..., "args": {...}}`` objects, exactly the shape
    :func:`run_plan` consumes.
    """
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    if not isinstance(plan, list):
        raise ValueError(
            f"plan file {path!r} must contain a JSON list of steps, got {type(plan).__name__}"
        )
    return run_plan(arm, plan, verbose=verbose)


def main(argv: list[str] | None = None) -> int:
    """Run a JSON plan file passed on the command line against a fresh arm.

    With no hardware connected the control layer auto-selects the mock backend,
    so this works on a bare laptop as well as on a real arm.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m limbic.primitives.run_sequence <plan.json>")
        return 2

    with RobotArm() as arm:
        run_plan_file(arm, args[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
