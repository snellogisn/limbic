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

# Default serial port the website looks for the arm on (the demo box's COM port).
# setdefault, so an explicit LIMBIC_PORT in the environment still wins; and on a
# machine with no such port (or no lerobot) the auto backend falls back to mock.
os.environ.setdefault("LIMBIC_PORT", "COM5")

# No MID-RUN camera correction: the plan executes start-to-finish without the
# closed-loop visual grasp nudge (aligned_pick -> align_to_object), which used to
# over-adjust and miss. The arm only re-checks the camera AFTER the task — it homes
# and re-detects for a single, lenient verification pass (see the plan_and_run call
# below). setdefault, so an explicit env still wins.
os.environ.setdefault("LIMBIC_VISUAL_ALIGN", "0")

# Motion timing for web-driven runs is set HERE, before limbic imports (config.py
# reads these env vars once at import). Safety rule: the website must NEVER drive
# the physical arm at speed. So only an explicit mock backend runs fast (for a
# snappy offline demo); anything that could be real hardware (backend "real", or
# "auto", which may resolve to the real arm) moves DELIBERATELY VERY SLOWLY.
# To get the fast offline demo, run with LIMBIC_BACKEND=mock.
if os.environ.get("LIMBIC_BACKEND", "auto").lower() == "mock":
    os.environ.setdefault("LIMBIC_SMOOTH_DT", "0.001")     # snappy simulation only
    os.environ.setdefault("LIMBIC_SLOW_DT", "0.001")
    os.environ.setdefault("LIMBIC_GRIPPER_SETTLE", "0.02")
else:
    # Real (or auto) hardware: SLOW *and* SMOOTH. Smoothness here is a SERIAL-BUS
    # problem, not just a math one. Tiny steps at ~50 Hz flood the Feetech bus
    # (6 motor writes per step => ~300 writes/s), and the irregular timing of a
    # saturated bus is itself the jerk. Direct hand-driven picks on THIS arm were
    # smooth at ~1.0 deg steps (~12-25 Hz, comfortable for the bus) with no extra
    # tuning — so we use bus-friendly steps and get the slowness from dt, not from
    # shrinking the step. Speed ≈ step/dt.
    os.environ.setdefault("LIMBIC_SMOOTH_STEP", "1.0")     # bus-friendly step (~25 Hz stream)
    os.environ.setdefault("LIMBIC_SMOOTH_DT", "0.04")      # transit ≈ 25 deg/s — slow + smooth
    os.environ.setdefault("LIMBIC_SLOW_STEP", "1.0")       # precision: same proven-smooth step ...
    os.environ.setdefault("LIMBIC_SLOW_DT", "0.06")        # ... slower (≈ 17 deg/s) for contact moves
    os.environ.setdefault("LIMBIC_GRIPPER_SETTLE", "0.7")
    # If any residual per-setpoint jerk remains, the next lever is the servo's own
    # Acceleration register (backends.py _apply_servo_acceleration) via
    # LIMBIC_SERVO_ACCEL — a hardware value to tune live, left UNSET here on purpose.

import threading  # noqa: E402

from limbic import RobotArm, runlog  # noqa: E402
from limbic.control import MotionStopped, safety  # noqa: E402


# A single shared stop signal for the in-flight run. The arm checks it during
# every move (RobotArm.bind_stop), so a separate "Stop" request can freeze the
# arm mid-motion. One arm => one run at a time, so one Event is enough; it's
# cleared at the start of each run so a stale stop never carries over.
_STOP = threading.Event()

# In-memory registry of runs started through the non-blocking streaming path
# (POST /api/run/start). Maps run_id -> {"status": "running"|"done", "result"}.
# The live thought/movement events themselves are read straight off disk
# (thinking.jsonl / movements.jsonl), which the brain appends to AS IT RUNS — so
# the browser sees reasoning arrive in real time by polling /api/run/live.
_RUNS: dict[str, dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()
_ACTIVE = threading.Event()  # one arm => at most one streaming run at a time


def request_stop() -> dict[str, Any]:
    """Signal the in-flight run to stop. Safe to call when nothing is running."""
    _STOP.set()
    return {"ok": True, "stopping": True}


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


def _collapse_thinking(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge live-streamed thought deltas back into one record per reasoning block.

    The brain streams its reasoning to the live feed as many small delta records
    that share a ``stream_id`` (see orchestrator._stream_response). For the logs
    page and the thought count we want the WHOLE thought, so consecutive records
    with the same ``stream_id`` are concatenated into a single clean record (the
    ``stream_id`` / ``partial`` bookkeeping dropped). Records without a
    ``stream_id`` (one-shot thoughts) pass through untouched, order preserved.
    """
    out: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        sid = rec.get("stream_id")
        if not sid:
            out.append(rec)
            continue
        if sid in by_id:
            by_id[sid]["message"] = (by_id[sid].get("message", "") or "") + (rec.get("message", "") or "")
        else:
            merged = {k: v for k, v in rec.items() if k not in ("stream_id", "partial")}
            by_id[sid] = merged
            out.append(merged)  # same reference kept in by_id, so appends update it
    return out


def _count_streams(run_dir: Path) -> dict[str, int]:
    """Count records in each log stream of a finished run."""
    counts = {}
    for channel, filename in (
        ("movements", "movements.jsonl"),
        ("data", "data.jsonl"),
    ):
        path = run_dir / filename
        counts[channel] = sum(1 for _ in path.open()) if path.exists() else 0
    # Thinking is counted AFTER collapsing streamed deltas, so "N thoughts" reflects
    # real reasoning blocks, not the live-streaming chunk count.
    counts["thinking"] = len(_collapse_thinking(_read_jsonl(run_dir / "thinking.jsonl")))
    return counts


def run_task(task: str, mode: str = "auto", on_start=None) -> dict[str, Any]:
    """Run ``task`` through the pipeline on the mock arm; return a structured result.

    ``mode`` is ``"auto"`` (claude if a key is set, else offline), ``"claude"``,
    or ``"offline"``. Everything — the planning, the motion, and (in claude mode)
    the reasoning — is recorded into a fresh run folder under ``logs/``.

    ``on_start``, if given, is called with the run_id the instant the run folder
    exists (before any planning/motion), so a caller streaming the live thought
    trail knows which folder to tail.

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

    _STOP.clear()  # fresh run — never inherit a stale stop request
    run_dir: Path | None = None
    with runlog.run(task, metadata={"source": "web", "mode": result["mode"]}) as log:
        run_dir = log.run_dir
        if on_start is not None:
            try:
                on_start(run_dir.name)
            except Exception:
                pass  # a streaming hook must never break the run
        arm = RobotArm(verbose=False, stop_event=_STOP).connect()
        try:
            if use_claude:
                from limbic.brain import plan_and_run

                try:
                    # plan_and_run logs into our already-active run; it returns a
                    # status (completed / incomplete / cannot_complete) rather than
                    # raising for a failed task. verify=True: after executing, the arm
                    # homes and the cameras re-detect for a LENIENT success check (the
                    # verifier accepts "close enough" — it won't re-do a grasp over a
                    # small positional miss). max_attempts=2 caps it at one retry so a
                    # genuine miss gets a single redo, not endless over-correction.
                    # (Auto-falls back to single-shot if no camera is available.)
                    outcome = plan_and_run(task, arm, verify=True, max_attempts=2)
                except (ValueError, RuntimeError) as exc:
                    # A refusal or a hard planner error: surface as cannot_complete.
                    log.thought("cannot_complete", str(exc))
                    result.update(status="cannot_complete", model="claude", error=str(exc))
                else:
                    st = outcome.get("status", "completed")
                    verdict = outcome.get("verification") or {}
                    # "incomplete" (ran out of retries) and "cannot_complete" both
                    # map to the UI's cannot_complete, with the verifier's reason.
                    result.update(
                        status="completed" if st == "completed" else "cannot_complete",
                        model=outcome["model"],
                        plan=outcome["plan"],
                        rationale=outcome["rationale"],
                        error=None if st == "completed" else (verdict.get("reason") or f"task {st} after retries"),
                    )
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

                    # Record the plan BEFORE executing, so if the run is stopped
                    # mid-motion the result still shows what it was doing.
                    result.update(model="offline-planner", plan=plan, rationale=rationale)
                    run_plan(arm, plan, verbose=False)
                    result.update(status="completed")
        except MotionStopped:
            # User hit Stop. The arm froze in place (torque held); report it
            # distinctly from a failure, and keep whatever model/plan we had.
            log.thought("stopped", "run stopped by user")
            result.update(status="stopped", error="stopped by user")
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
# Live streaming path (POST /api/run/start + GET /api/run/live)
# --------------------------------------------------------------------------- #
def start_run_async(task: str, mode: str = "auto") -> dict[str, Any]:
    """Kick off a run on a background thread; return its ``run_id`` immediately.

    The browser then polls :func:`live` to watch the brain's reasoning, tool use,
    and motion arrive in real time, instead of staring at a spinner until the
    whole run finishes. Only one run at a time (one physical arm).
    """
    task = (task or "").strip()
    if not task:
        return {"error": "missing 'task'"}
    if _ACTIVE.is_set():
        return {"error": "a run is already in progress — wait for it to finish or hit Stop", "busy": True}

    started = threading.Event()
    holder: dict[str, str] = {}

    def _on_start(run_id: str) -> None:
        holder["run_id"] = run_id
        with _RUNS_LOCK:
            _RUNS[run_id] = {"status": "running", "result": None}
        started.set()

    def _worker() -> None:
        _ACTIVE.set()
        try:
            result = run_task(task, mode=mode, on_start=_on_start)
        except Exception as exc:  # never let the worker thread die silently
            result = {"status": "error", "task": task, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            rid = holder.get("run_id") or (result.get("run_id") if isinstance(result, dict) else None)
            if rid:
                with _RUNS_LOCK:
                    _RUNS[rid] = {"status": "done", "result": result}
            started.set()  # unblock the waiter even if on_start never fired
            _ACTIVE.clear()

    threading.Thread(target=_worker, daemon=True).start()
    started.wait(timeout=15)  # the run folder is created near-instantly
    rid = holder.get("run_id")
    if not rid:
        return {"error": "the run failed to start"}
    return {"run_id": rid, "status": "running"}


def _live_events(run_id: str, since: int) -> list[dict[str, Any]]:
    """Read new thinking + movement records (seq > ``since``) for ``run_id``.

    Both streams share one monotonically increasing ``seq`` (see runlog._emit),
    so merging and sorting by ``seq`` yields the true chronological order of
    'thought, then thought, then the move it triggered, then the next thought'.
    """
    base = runlog.base_log_dir()
    # Validate run_id is a real direct child of the logs dir (no path traversal).
    run_dir = (base / run_id).resolve()
    if base.resolve() not in run_dir.parents or not run_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for channel, fname in (("thinking", "thinking.jsonl"), ("movements", "movements.jsonl")):
        for rec in _read_jsonl(run_dir / fname):
            if int(rec.get("seq", 0)) > since:
                rec = dict(rec)
                rec["channel"] = channel
                out.append(rec)
    out.sort(key=lambda r: int(r.get("seq", 0)))
    return out


def live(run_id: str, since: int = 0) -> dict[str, Any]:
    """Return new live events for ``run_id`` plus run status (for polling).

    ``since`` is the highest ``seq`` the client already has; only newer events
    are returned. When the run is finished, ``done`` is True and the full
    structured ``result`` is included so the UI can render the final outcome.
    """
    with _RUNS_LOCK:
        entry = _RUNS.get(run_id)
    status = entry["status"] if entry else "unknown"
    events = _live_events(run_id, since)
    last_seq = int(events[-1]["seq"]) if events else since
    out: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "events": events,
        "last_seq": last_seq,
        "done": status == "done",
    }
    if status == "done" and entry:
        out["result"] = entry["result"]
    return out


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
        "thinking": _collapse_thinking(_read_jsonl(run_dir / "thinking.jsonl")),
    }
