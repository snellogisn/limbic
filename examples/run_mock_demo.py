"""End-to-end demo of the limbic brain on the MOCK arm — always runnable offline.

Run it two ways:

    * With ANTHROPIC_API_KEY set, it asks the brain to plan a real instruction
      ("pick up the block at (160, 40) and place it at (160, -40)") and prints the
      model chosen, the plan, and the execution results.
    * Without a key, it prints a clear note and falls back to executing a fixed
      plan on the mock arm so the WHOLE pipeline (registry -> run_plan -> RobotArm
      -> mock backend, with the safety layer) still runs.

Either way it builds the arm with ``RobotArm(verbose=True)`` — with no hardware
attached the control layer auto-selects the software mock, so this is safe and
deterministic on any laptop.

    python3 examples/run_mock_demo.py
"""

from __future__ import annotations

import os
import sys

# Make ``limbic`` importable when run as a bare script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic import RobotArm, runlog  # noqa: E402

INSTRUCTION = "pick up the block at (160, 40) and place it at (160, -40)"


def _offline_fallback_plan() -> list[dict]:
    """A small, sensible plan to run when there is no API key.

    Prefer the project's shared ``EXAMPLE_PLAN`` if the primitives package
    exposes one; otherwise use a minimal hardcoded plan built from whatever
    primitives happen to be registered, and finally a pure-arm motion if the
    primitive library is still empty (other agents may be populating it).
    """
    try:
        from limbic.primitives.example_plan import EXAMPLE_PLAN

        return list(EXAMPLE_PLAN)
    except Exception:
        pass

    # Build a tiny plan from registered primitives if any look like home/move.
    from limbic.primitives import registry as primitives

    available = set(primitives.all_primitives())
    plan: list[dict] = []
    if "home" in available:
        plan.append({"primitive": "home", "args": {}})
    if "move_to" in available:
        plan.append({"primitive": "move_to", "args": {"x_mm": 160, "y_mm": 0, "z_mm": 70}})
    return plan


def _run_plan_offline(arm: RobotArm) -> None:
    """Execute the offline fallback plan through the sequence runner if available."""
    plan = _offline_fallback_plan()

    try:
        from limbic.primitives import run_sequence
    except Exception:
        run_sequence = None

    if run_sequence is not None and plan:
        print(f"[demo] running fallback plan ({len(plan)} step(s)) via run_plan:")
        for step in plan:
            print(f"        - {step['primitive']} {step.get('args', {})}")
        results = run_sequence.run_plan(arm, plan, verbose=True)
        print(f"[demo] results: {results}")
        return

    # Last resort: the primitive library / runner is not ready yet. Drive the arm
    # directly so the demo still exercises the control + mock layers cleanly.
    print(
        "[demo] no plan/runner available yet (primitive library still loading) — "
        "exercising the arm directly instead."
    )
    arm.go_home()
    arm.reach_above(160, 0, 70)
    arm.descend_to(160, 0, 25)
    arm.lift_by(50)
    arm.go_home()
    print(f"[demo] final tip position: {arm.current_xyz()}")


def main() -> None:
    # Open a run so this demo records its movements, data, and (with a key) its
    # thinking into logs/<timestamp>-mock-demo/. The brain's plan_and_run sees the
    # active run and logs into it rather than opening a second one.
    with runlog.run("mock-demo") as log:
        print(f"[demo] logging this run to: {log.run_dir}\n")
        # No hardware -> the control layer auto-selects the mock backend.
        with RobotArm(verbose=True) as arm:
            _run_demo(arm)
    print(f"\n[demo] run log written to: {log.run_dir}")


def _run_demo(arm: RobotArm) -> None:
    """Plan with the brain if a key is present, else run the offline fallback."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[demo] ANTHROPIC_API_KEY found — planning with the brain.\n")
        from limbic.brain import plan_and_run

        outcome = plan_and_run(INSTRUCTION, arm)
        print("\n[demo] ----- outcome -----")
        print(f"[demo] model:     {outcome['model']}")
        print(f"[demo] rationale: {outcome['rationale']}")
        print("[demo] plan:")
        for step in outcome["plan"]:
            print(f"        - {step['primitive']} {step['args']}")
        print(f"[demo] executed:  {outcome['executed']}")
        print(f"[demo] results:   {outcome['results']}")
    else:
        print(
            "[demo] ANTHROPIC_API_KEY is not set — running the OFFLINE demo.\n"
            "       (Set the key to plan '" + INSTRUCTION + "' from natural "
            "language.)\n"
        )
        _run_plan_offline(arm)


if __name__ == "__main__":
    main()
