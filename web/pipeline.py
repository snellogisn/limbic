"""Bridge between the web server and the limbic pipeline.

The website hands a natural-language task to :func:`run_task`, which runs the
full limbic pipeline on the (mock) arm inside a logged run, then returns a
structured result the UI can render. It also exposes :func:`list_runs` and
:func:`get_run` so the "logs" page can browse every run ever made.

Two planning modes, chosen automatically:

    * **claude**  — used when ``ANTHROPIC_API_KEY`` is set. The real brain
      (``limbic.brain.plan_and_run``) asks Claude to perceive and compile a plan.
    * **offline** — a small, deterministic rule-based planner used when no key is
      present. It keeps the whole site (and the test buttons) working with zero
      configuration, which is what makes this Claude-Code-friendly: you can drive
      the arm and exercise every page without an API key.

THE "CANNOT COMPLETE" SCENARIO is first-class. Both modes can fail to produce a
runnable plan — the offline planner raises :class:`CannotComplete` for tasks
outside the arm's tabletop abilities or outside its reach; the claude path
surfaces a refusal or a no-plan outcome the same way. Either becomes a
``status="cannot_complete"`` result that the UI shows distinctly from a hard
error, with the reason the arm couldn't do it.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

# --- Make limbic importable and point logs at the repo root, BEFORE importing it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("LIMBIC_LOG_DIR", str(_REPO_ROOT / "logs"))
# Run the mock arm fast so web requests return promptly (real hardware is
# unaffected — these only shrink the simulated inter-step delays).
os.environ.setdefault("LIMBIC_SMOOTH_DT", "0.001")
os.environ.setdefault("LIMBIC_SLOW_DT", "0.001")
os.environ.setdefault("LIMBIC_GRIPPER_SETTLE", "0.02")

from limbic import RobotArm, runlog  # noqa: E402
from limbic.control import safety  # noqa: E402


class CannotComplete(Exception):
    """Raised when the arm genuinely cannot do what was asked (the failure case)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# A capability blurb shown when we decline a task, so the user learns what works.
_CAPABILITIES = (
    "I'm a tabletop arm. I can home, open/close the hand, move to a point, "
    "pick, place, and push objects — all within reach of the base. Try e.g. "
    "\"pick up the block at (160, 40) and place it at (160, -40)\"."
)

_COORD_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(?:\s*,\s*(-?\d+(?:\.\d+)?))?\s*\)"
)


def _coords(task: str) -> list[tuple[float, ...]]:
    """Pull every ``(x, y)`` or ``(x, y, z)`` tuple out of the task text (mm)."""
    out: list[tuple[float, ...]] = []
    for match in _COORD_RE.finditer(task):
        out.append(tuple(float(g) for g in match.groups() if g is not None))
    return out


def _require_reachable(x_mm: float, y_mm: float) -> None:
    """Raise CannotComplete if a point is outside the arm's workspace dome."""
    radius = math.hypot(x_mm, y_mm)
    ws = safety.WORKSPACE
    if radius > ws.reach_max_mm:
        raise CannotComplete(
            f"the point ({x_mm:.0f}, {y_mm:.0f}) is {radius:.0f} mm from the base, "
            f"beyond the arm's {ws.reach_max_mm:.0f} mm reach. Pick a point closer in."
        )


def _offline_plan(task: str) -> tuple[list[dict[str, Any]], str]:
    """Rule-based planner: map a task to a primitive plan, or decline it.

    Recognises home / open / close / move / pick / place / push (the standard
    primitive set). Returns ``(plan, rationale)``; raises :class:`CannotComplete`
    for anything outside those abilities or outside the arm's reach — that's the
    deliberate failure scenario the UI demonstrates.
    """
    low = task.lower()
    coords = _coords(task)

    # --- gripper-only ----------------------------------------------------
    if re.search(r"\bopen\b", low) and "hand" in low or low.strip() in {"open", "open hand"}:
        return [{"primitive": "open_hand", "args": {}}], "Open the gripper."
    if re.search(r"\b(close|grip|grab)\b", low) and "hand" in low or low.strip() in {"close", "close hand"}:
        return [{"primitive": "close_hand", "args": {}}], "Close the gripper."

    # --- home ------------------------------------------------------------
    if re.search(r"\b(home|rest|park|reset)\b", low) and not coords:
        return [{"primitive": "home", "args": {}}], "Return the arm to its home pose."

    # --- pick AND place (two coordinates) --------------------------------
    wants_pick = bool(re.search(r"\b(pick|grab|grasp)\b", low))
    wants_place = bool(re.search(r"\b(place|put|drop|set)\b", low))
    if wants_pick and wants_place and len(coords) >= 2:
        (px, py), (qx, qy) = coords[0][:2], coords[1][:2]
        _require_reachable(px, py)
        _require_reachable(qx, qy)
        plan = [
            {"primitive": "home", "args": {}},
            {"primitive": "open_hand", "args": {}},
            {"primitive": "pick", "args": {"x_mm": px, "y_mm": py}},
            {"primitive": "place", "args": {"x_mm": qx, "y_mm": qy}},
            {"primitive": "home", "args": {}},
        ]
        return plan, f"Pick the object at ({px:.0f},{py:.0f}) and place it at ({qx:.0f},{qy:.0f})."

    # --- pick only -------------------------------------------------------
    if wants_pick and len(coords) >= 1:
        px, py = coords[0][:2]
        _require_reachable(px, py)
        plan = [
            {"primitive": "home", "args": {}},
            {"primitive": "open_hand", "args": {}},
            {"primitive": "pick", "args": {"x_mm": px, "y_mm": py}},
        ]
        return plan, f"Pick the object at ({px:.0f},{py:.0f})."

    # --- push ------------------------------------------------------------
    if re.search(r"\bpush\b", low) and len(coords) >= 2:
        (px, py), (qx, qy) = coords[0][:2], coords[1][:2]
        _require_reachable(px, py)
        _require_reachable(qx, qy)
        plan = [{"primitive": "push", "args": {"x_mm": px, "y_mm": py, "x2_mm": qx, "y2_mm": qy}}]
        return plan, f"Push from ({px:.0f},{py:.0f}) to ({qx:.0f},{qy:.0f})."

    # --- move / go to ----------------------------------------------------
    if re.search(r"\b(move|go|reach)\b", low) and len(coords) >= 1:
        first = coords[0]
        x, y = first[0], first[1]
        z = first[2] if len(first) >= 3 else 60.0
        _require_reachable(x, y)
        return (
            [{"primitive": "move_to", "args": {"x_mm": x, "y_mm": y, "z_mm": z}}],
            f"Move the tool tip to ({x:.0f},{y:.0f},{z:.0f}).",
        )

    # --- nothing matched: decline ----------------------------------------
    raise CannotComplete(
        f"I can't work out how to do \"{task}\" with the arm. {_CAPABILITIES}"
    )


def _count_streams(run_dir: Path) -> dict[str, int]:
    """Count records in each log stream of a finished run."""
    counts = {}
    for channel, filename in (
        ("movements", "movements.jsonl"),
        ("data", "data.jsonl"),
        ("thinking", "thinking.jsonl"),
    ):
        path = run_dir / filename
        counts[channel] = sum(1 for _ in path.open()) if path.exists() else 0
    return counts


def run_task(task: str, mode: str = "auto") -> dict[str, Any]:
    """Run ``task`` through the pipeline on the mock arm; return a structured result.

    ``mode`` is ``"auto"`` (claude if a key is set, else offline), ``"claude"``,
    or ``"offline"``. Everything — the planning, the motion, and (in claude mode)
    the reasoning — is recorded into a fresh run folder under ``logs/``.

    Returns a dict with at least: ``run_id``, ``task``, ``mode``, ``status``
    (``completed`` | ``cannot_complete`` | ``error``), ``model``, ``plan``,
    ``rationale``, ``error``, and ``counts``.
    """
    task = (task or "").strip()
    if not task:
        return {"status": "error", "task": task, "error": "empty task", "plan": [], "mode": mode}

    use_claude = mode == "claude" or (mode == "auto" and bool(os.environ.get("ANTHROPIC_API_KEY")))
    result: dict[str, Any] = {
        "task": task,
        "mode": "claude" if use_claude else "offline",
        "status": "error",
        "model": None,
        "plan": [],
        "rationale": "",
        "error": None,
    }

    run_dir: Path | None = None
    with runlog.run(task, metadata={"source": "web", "mode": result["mode"]}) as log:
        run_dir = log.run_dir
        arm = RobotArm(verbose=False).connect()
        try:
            if use_claude:
                from limbic.brain import plan_and_run

                try:
                    # plan_and_run logs into our already-active run.
                    outcome = plan_and_run(task, arm)
                    result.update(
                        status="completed",
                        model=outcome["model"],
                        plan=outcome["plan"],
                        rationale=outcome["rationale"],
                    )
                except (ValueError, RuntimeError) as exc:
                    # No usable plan / a refusal: the model could not complete it.
                    log.thought("cannot_complete", str(exc))
                    result.update(status="cannot_complete", model="claude", error=str(exc))
            else:
                log.thought(
                    "model_choice",
                    "offline rule-based planner (no ANTHROPIC_API_KEY set)",
                    model="offline-planner",
                )
                try:
                    plan, rationale = _offline_plan(task)
                except CannotComplete as exc:
                    log.thought("cannot_complete", exc.reason)
                    result.update(status="cannot_complete", model="offline-planner", error=exc.reason)
                else:
                    log.thought("plan", rationale, plan=plan)
                    from limbic.primitives.run_sequence import run_plan

                    run_plan(arm, plan, verbose=False)
                    result.update(
                        status="completed", model="offline-planner", plan=plan, rationale=rationale
                    )
        except Exception as exc:  # any unexpected failure -> hard error (not the model's fault)
            log.thought("error", f"{type(exc).__name__}: {exc}")
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            arm.disconnect()

    result["run_id"] = run_dir.name
    result["log_dir"] = str(run_dir)
    result["counts"] = _count_streams(run_dir)
    # Persist the rich web result alongside the raw log streams.
    (run_dir / "web_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Browsing past runs (for the logs page)
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def list_runs() -> list[dict[str, Any]]:
    """Summarise every run in the logs folder, newest first (for the list page)."""
    base = runlog.base_log_dir()
    if not base.exists():
        return []

    runs: list[dict[str, Any]] = []
    for entry in sorted(base.iterdir(), reverse=True):  # name starts with timestamp
        if not entry.is_dir():
            continue
        web = _read_json(entry / "web_result.json")
        meta = _read_json(entry / "run.json")
        runs.append(
            {
                "run_id": entry.name,
                "task": web.get("task") or meta.get("label") or entry.name,
                "status": web.get("status", meta.get("status", "unknown")),
                "mode": web.get("mode"),
                "model": web.get("model"),
                "started_at": meta.get("started_at"),
                "steps": len(web.get("plan", []) or []),
                "error": web.get("error"),
                "counts": web.get("counts") or meta.get("record_counts", {}),
            }
        )
    return runs


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return the full detail of one run (metadata + all three log streams).

    ``run_id`` is validated against the actual folders to avoid path traversal —
    only a real direct child of the logs directory is accepted.
    """
    base = runlog.base_log_dir()
    safe = {p.name for p in base.iterdir() if p.is_dir()} if base.exists() else set()
    if run_id not in safe:
        return None

    run_dir = base / run_id
    return {
        "run_id": run_id,
        "result": _read_json(run_dir / "web_result.json"),
        "meta": _read_json(run_dir / "run.json"),
        "movements": _read_jsonl(run_dir / "movements.jsonl"),
        "data": _read_jsonl(run_dir / "data.jsonl"),
        "thinking": _read_jsonl(run_dir / "thinking.jsonl"),
    }
