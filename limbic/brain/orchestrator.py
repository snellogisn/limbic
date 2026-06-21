"""The orchestrator: a plan -> execute -> verify -> retry cycle driven by Claude.

This is the heart of the brain. A task is accomplished as a CYCLE, not a single
shot:

    1. PLAN    Claude perceives (sense_* tools) if needed, may CREATE or EDIT a
               motion primitive when no existing skill fits (the library is
               dynamic), then commits one ordered list of primitive steps
               (`submit_plan`).
    2. EXECUTE we run that list on the arm — every step through `RobotArm`, so the
               safety clamps always apply (our code drives the motors, never the
               model directly).
    3. VERIFY  a check decides whether the task is actually SATISFIED or still
               INCOMPLETE, using a fresh perception snapshot (`inputs.snapshot()`).
    4. RETRY   if incomplete, the failure + snapshot are fed back and the cycle
               repeats (Claude can revise the plan, or author a better primitive),
               up to `max_attempts`.

The verification step is pluggable (`verifier=`). The default asks Claude to judge
from the snapshot + execution results; when a live object-detection input (e.g. a
streaming YOLO feed) is later added to ``inputs/library/``, it is automatically
part of every snapshot, so verification becomes detection-grounded with no change
here — that integration is intentionally left to a separate effort.

`choose_model` routes between a fast model (urgent / trivial) and the most capable
one (complex). The Anthropic SDK is imported lazily, and a `client` can be
injected (so the whole cycle is testable offline without a key).
"""

from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any, Callable

from .. import runlog
from ..inputs import registry as inputs
from ..primitives import authoring  # noqa: F401  (kept for discoverability/tools)
from ..primitives import registry as primitives
from .system_prompt import build_system_prompt
from .tools import SENSE_PREFIX, authoring_tools, input_tools, submit_plan_tool

# Literal model IDs — never date-suffixed. The brain defaults to the most capable
# model and steps down to the fast one only when routing says so.
MODEL_FAST = "claude-sonnet-4-6"
MODEL_CAPABLE = "claude-opus-4-8"

# Single-word action verbs that are unambiguous one-shot commands; if an
# instruction is essentially just one of these, the fast model is plenty.
_SIMPLE_VERBS = {"home", "open", "close", "stop", "rest", "park", "release", "grip"}

# Above this many words an instruction is almost certainly multi-step / spatial
# enough to want the capable model.
_SHORT_WORD_LIMIT = 4

# Guard rails so a misbehaving model can't loop forever within one planning turn.
_MAX_PLAN_TURNS = 8


def choose_model(instruction: str, urgent: bool = False) -> str:
    """Pick a model for ``instruction`` — fast for urgency/triviality, else capable.

    Heuristic ("a quick model for urgency, a large model for complex problems"):

      * ``urgent=True``           -> always the fast model. The caller is telling
                                     us latency matters more than depth (e.g. an
                                     emergency stop), so don't pay for reasoning.
      * trivial single actions    -> fast model. If the instruction is just an
                                     obvious one-shot verb ("home", "open",
                                     "close", "stop", ...) or is very short
                                     (<= 4 words) with no sign of multi-step or
                                     spatial structure, the fast model handles it.
      * everything else           -> the capable model (default). Multi-step,
                                     spatial, or anything we are unsure about
                                     gets the strong planner. Defaulting to
                                     capable is the safe bias: a wrong plan is far
                                     more costly than a few extra tokens.
    """
    if urgent:
        return MODEL_FAST

    text = instruction.strip().lower()
    words = re.findall(r"[a-z']+", text)

    complex_signals = (
        " then ", " and ", ",", "stack", "place", "left of", "right of",
        "on top", "next to", "between", "(",
    )
    if any(signal in f" {text} " for signal in complex_signals):
        return MODEL_CAPABLE

    if words and words[0] in _SIMPLE_VERBS and len(words) <= _SHORT_WORD_LIMIT:
        return MODEL_FAST
    if len(words) <= _SHORT_WORD_LIMIT:
        return MODEL_FAST
    return MODEL_CAPABLE


def _validate_plan(plan: Any) -> list[dict[str, Any]]:
    """Validate a submitted plan against the registry, returning clean steps.

    Checks shape (a list of ``{"primitive", "args"}``), that every primitive name
    exists in the (freshly re-scanned) registry, and that each step supplies all
    required arguments (those whose schema lacks a ``"default"``). Raises
    ``ValueError`` with a clear, step-indexed message on the first problem — which
    the planning turn feeds back to the model so it can fix it before anything
    moves.
    """
    if not isinstance(plan, list) or not plan:
        raise ValueError("submitted plan must be a non-empty list of steps")

    available = primitives.all_primitives()
    cleaned: list[dict[str, Any]] = []

    for index, step in enumerate(plan):
        if not isinstance(step, dict):
            raise ValueError(f"plan step {index} is not an object: {step!r}")
        name = step.get("primitive")
        args = step.get("args", {}) or {}
        if name not in available:
            raise ValueError(
                f"plan step {index} names unknown primitive '{name}'. Available: "
                f"{', '.join(sorted(available)) or '(none)'}. Create it with "
                "create_primitive, or use an existing one."
            )
        if not isinstance(args, dict):
            raise ValueError(f"plan step {index} args must be an object, got {args!r}")

        spec = available[name].parameters or {}
        missing = [
            arg for arg, meta in spec.items()
            if "default" not in (meta or {}) and arg not in args
        ]
        if missing:
            raise ValueError(
                f"plan step {index} ('{name}') is missing required argument(s): "
                f"{', '.join(missing)}"
            )
        cleaned.append({"primitive": name, "args": args})

    return cleaned


# --------------------------------------------------------------------------- #
# One planning turn: perceive / author / submit a validated plan
# --------------------------------------------------------------------------- #
def _dispatch_tool(block: Any, trail, verbose: bool, perceptions: dict | None = None) -> tuple[dict[str, Any], dict | None]:
    """Run one tool_use block. Returns (tool_result, captured_plan_or_None).

    ``captured_plan`` is non-None only for a valid ``submit_plan`` (a dict with
    ``plan`` + ``rationale``). Senses dispatch to ``inputs.read``; create/edit
    dispatch to the authoring helpers (which hot-reload the registry).

    ``perceptions``, if given, accumulates the planner's object detections (the
    "before" world state) so the outcome of the attempt can be scored against it:
    its ``prompts`` set records what was looked for (to re-detect later) and its
    ``objects`` list collects the detected objects.
    """
    name = block.name
    tool_id = block.id

    def result(payload: Any, is_error: bool = False) -> dict[str, Any]:
        out = {"type": "tool_result", "tool_use_id": tool_id, "content": json.dumps(payload, default=str)}
        if is_error:
            out["is_error"] = True
        return out

    if name == "submit_plan":
        raw = dict(block.input or {})
        try:
            plan = _validate_plan(raw.get("plan"))
        except ValueError as exc:
            # Don't fail the run — tell the model what's wrong so it can fix it.
            trail.thought("plan_rejected", message=str(exc))
            return result({"status": "rejected", "error": str(exc)}, is_error=True), None
        captured = {"plan": plan, "rationale": raw.get("rationale", "")}
        return result({"status": "plan accepted"}), captured

    if name in ("create_primitive", "edit_primitive"):
        fn = authoring.create_primitive if name == "create_primitive" else authoring.edit_primitive
        args = block.input or {}
        outcome = fn(args.get("name", ""), args.get("code", ""))
        trail.thought(
            "authoring",
            message=f"{name}({args.get('name','')}) -> {'ok' if outcome.get('ok') else 'failed'}",
            outcome=outcome,
        )
        if verbose:
            print(f"[brain] {name} {args.get('name','')!r}: {'ok' if outcome.get('ok') else outcome.get('error')}")
        return result(outcome, is_error=not outcome.get("ok", False)), None

    if name.startswith(SENSE_PREFIX):
        sense = name[len(SENSE_PREFIX):]
        try:
            reading = inputs.read(sense, **(block.input or {}))
            payload: Any = {"reading": reading}
            # Capture object detections as the "before" world state for outcome
            # scoring: remember the prompt (to re-detect after the action) and the
            # objects it located.
            if (perceptions is not None and sense == "object_detections"
                    and isinstance(reading, dict) and reading.get("objects")):
                prompt = (block.input or {}).get("prompt")
                if prompt:
                    perceptions.setdefault("prompts", set()).add(prompt)
                    perceptions.setdefault("objects", []).extend(reading["objects"])
        except Exception as exc:
            payload = {"error": str(exc)}
        trail.thought("perceive", message=f"queried sense '{sense}'", sense=sense, result=payload)
        if verbose:
            print(f"[brain] perceived {sense}: {payload}")
        return result(payload), None

    return result({"error": f"unknown tool {name}"}, is_error=True), None


# How many characters of new reasoning/text to accumulate before flushing a live
# chunk to the thinking log. Small enough that the website feels like it's watching
# the model think in real time; large enough not to spam the log with one record
# per token.
_LIVE_FLUSH_CHARS = 48


def _emit_whole_blocks(resp: Any, trail) -> None:
    """Fallback emitter: log each finished thinking/text block in one shot.

    Used when the client can't stream (e.g. an injected mock client in offline
    runs) — the reasoning still reaches the thinking log, just not incrementally.
    """
    for block in resp.content:
        kind = getattr(block, "type", None)
        if kind == "thinking" and getattr(block, "thinking", ""):
            trail.thought("reasoning", block.thinking)
        elif kind == "text" and getattr(block, "text", ""):
            trail.thought("message", block.text)


def _stream_response(client, create_kwargs, messages, trail):
    """Run one planning call, streaming the model's reasoning to the live feed.

    As Claude generates, its **thinking** and **assistant text** arrive as deltas;
    we accumulate them and flush a thought record every ``_LIVE_FLUSH_CHARS`` (and
    at each block's end), tagged with a per-block ``stream_id`` so the website can
    coalesce the deltas into one growing, "typing" bubble — the user watches the
    decision form instead of waiting for the whole turn. Returns the final
    ``Message`` (identical shape to ``messages.create``), so the rest of the
    planning loop — tool-use extraction, the thinking signatures in the assistant
    turn — is unchanged.

    Degrades gracefully: a client without ``messages.stream`` (a test/offline mock)
    falls back to a single non-streaming call + :func:`_emit_whole_blocks`.
    """
    stream_ctx = getattr(getattr(client, "messages", None), "stream", None)
    if stream_ctx is None:
        resp = client.messages.create(messages=messages, **create_kwargs)
        _emit_whole_blocks(resp, trail)
        return resp

    state = {"id": None, "phase": None, "buf": "", "pending": 0}

    def flush(partial: bool) -> None:
        if state["buf"]:
            # message carries only the NEW text since the last flush (a delta); the
            # website appends deltas sharing a stream_id. partial=False marks the
            # block complete (so the UI can drop its typing caret).
            trail.thought(state["phase"], state["buf"], stream_id=state["id"], partial=partial)
            state["buf"] = ""
            state["pending"] = 0

    with stream_ctx(messages=messages, **create_kwargs) as stream:
        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "content_block_start":
                block = getattr(event, "content_block", None)
                btype = getattr(block, "type", None)
                if btype in ("thinking", "text"):
                    state["id"] = uuid.uuid4().hex[:8]
                    state["phase"] = "reasoning" if btype == "thinking" else "message"
                    state["buf"] = ""
                    state["pending"] = 0
                else:
                    state["phase"] = None  # tool_use etc. — not narrated live
            elif etype == "content_block_delta" and state["phase"]:
                delta = getattr(event, "delta", None)
                dtype = getattr(delta, "type", None)
                piece = ""
                if dtype == "thinking_delta":
                    piece = getattr(delta, "thinking", "") or ""
                elif dtype == "text_delta":
                    piece = getattr(delta, "text", "") or ""
                if piece:
                    state["buf"] += piece
                    state["pending"] += len(piece)
                    if state["pending"] >= _LIVE_FLUSH_CHARS:
                        flush(partial=True)
            elif etype == "content_block_stop" and state["phase"]:
                flush(partial=False)
                state["phase"] = None
        return stream.get_final_message()


def _plan_turn(client, model, tools, system, messages, trail, verbose, perceptions=None) -> dict | None:
    """Drive Claude until it commits a valid plan (or gives up). Mutates ``messages``.

    Returns ``{"plan", "rationale"}`` on success, or ``None`` if the model ended
    its turn without a usable plan. Perception, authoring, and plan-rejection
    feedback all happen here, so a single "attempt" can self-heal (e.g. create a
    missing primitive, or fix a bad argument) before it ever moves the arm.

    ``perceptions`` (if given) collects the planner's object detections so the
    attempt's outcome can later be measured against where things were.
    """
    create_kwargs: dict[str, Any] = {
        "model": model, "max_tokens": 16000, "tools": tools, "system": system,
    }
    if model == MODEL_CAPABLE:
        create_kwargs["thinking"] = {"type": "adaptive"}

    for _ in range(_MAX_PLAN_TURNS):
        # Stream the call so the model's reasoning reaches the live feed AS it's
        # produced (deltas), not in one chunk after the turn finishes.
        resp = _stream_response(client, create_kwargs, messages, trail)

        if resp.stop_reason == "refusal":
            trail.thought("refused", message="model declined to plan this instruction")
            raise RuntimeError("the model refused to plan this instruction; rephrase and retry")

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            # No tools called and the turn ended -> no plan produced.
            return None

        messages.append({"role": "assistant", "content": resp.content})

        tool_results: list[dict[str, Any]] = []
        captured: dict | None = None
        for block in tool_uses:
            tool_result, plan = _dispatch_tool(block, trail, verbose, perceptions)
            tool_results.append(tool_result)
            if plan is not None:
                captured = plan

        messages.append({"role": "user", "content": tool_results})
        if captured is not None:
            trail.thought(
                "plan_validated",
                message=f"{len(captured['plan'])} step(s); {captured['rationale']}",
                plan=captured["plan"],
            )
            return captured

    return None  # ran out of planning iterations without a committed plan


# --------------------------------------------------------------------------- #
# Execution + verification
# --------------------------------------------------------------------------- #
def _execute(arm, plan, verbose) -> tuple[list[Any], str | None]:
    """Run the plan on the arm. Returns (results, error_message_or_None)."""
    from ..control import MotionStopped
    from ..primitives import run_sequence

    try:
        results = run_sequence.run_plan(arm, plan, verbose=verbose)
        return results, None
    except MotionStopped:
        raise  # a user stop is not a retryable error — abort the whole cycle
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Before/after outcome measurement
# --------------------------------------------------------------------------- #
# So a retry knows HOW it missed, not just THAT it did: we re-detect the objects
# the planner located, compare their positions before vs after the action, and
# quantify each aimed step against where it ended up. That measured error is what
# lets attempt 2 actually differ from attempt 1.
_AIMED_PRIMS = ("pick", "aligned_pick", "place", "push", "move_to", "descend_to", "reach_above")


def _obj_xy(o: Any) -> tuple[float, float] | None:
    """Pull a detection's table (x, y) in mm, or None if it has no coordinate."""
    tm = o.get("table_mm") if isinstance(o, dict) else None
    if isinstance(tm, (list, tuple)) and len(tm) >= 2 and tm[0] is not None:
        return float(tm[0]), float(tm[1])
    return None


def _nearest(objs: list[dict], x: float, y: float, label: str | None = None):
    """Nearest detection to (x, y) (optionally same label). Returns (obj, dist_mm)."""
    best, best_d = None, float("inf")
    for o in objs:
        if label is not None and o.get("label") != label:
            continue
        p = _obj_xy(o)
        if p is None:
            continue
        d = math.hypot(p[0] - x, p[1] - y)
        if d < best_d:
            best, best_d = o, d
    return best, best_d


def _redetect(prompts) -> list[dict]:
    """Re-run each detection prompt; return a flat, de-duplicated object list."""
    found: list[dict] = []
    for prompt in prompts:
        try:
            reading = inputs.read("object_detections", prompt=prompt)
        except Exception:
            continue
        if not isinstance(reading, dict):
            continue
        for o in reading.get("objects") or []:
            p = _obj_xy(o)
            if p is None:
                continue
            dup = any(o.get("label") == s.get("label")
                      and (_obj_xy(s) or (1e9, 1e9))
                      and math.hypot(p[0] - _obj_xy(s)[0], p[1] - _obj_xy(s)[1]) < 12
                      for s in found if _obj_xy(s))
            if not dup:
                found.append(o)
    return found


def _measure_outcome(before: list[dict], after: list[dict], plan: list[dict]) -> dict | None:
    """Quantify the attempt: per aimed step + per object before->after. None if nothing.

    For each aimed step we report the nearest object detected AFTER the action and
    its offset from the aim (so a missed grasp / off place is a concrete mm vector).
    For each object the planner saw we report how far it moved (or that it's gone —
    e.g. lifted away). Factual numbers; the verifier/planner interpret them.
    """
    if not before and not after:
        return None

    steps: list[dict[str, Any]] = []
    for step in plan:
        prim = step.get("primitive")
        if prim not in _AIMED_PRIMS:
            continue
        args = step.get("args", {}) or {}
        if prim == "push" and args.get("x2_mm") is not None:
            ax, ay = float(args["x2_mm"]), float(args["y2_mm"])  # push aims at its destination
        elif args.get("x_mm") is not None and args.get("y_mm") is not None:
            ax, ay = float(args["x_mm"]), float(args["y_mm"])
        else:
            continue
        entry: dict[str, Any] = {"step": prim, "aim_mm": [round(ax, 1), round(ay, 1)]}
        near, dist = _nearest(after, ax, ay)
        if near is not None and dist < 60.0:
            nx, ny = _obj_xy(near)
            entry["nearest_after"] = {
                "label": near.get("label"),
                "xy_mm": [round(nx, 1), round(ny, 1)],
                "offset_from_aim_mm": [round(nx - ax, 1), round(ny - ay, 1)],
            }
        else:
            entry["nearest_after"] = None  # nothing landed near the aim
        steps.append(entry)

    objects: list[dict[str, Any]] = []
    used = [False] * len(after)
    for b in before:
        bp = _obj_xy(b)
        if bp is None:
            continue
        mi, md = None, float("inf")
        for i, a in enumerate(after):
            if used[i] or a.get("label") != b.get("label"):
                continue
            ap = _obj_xy(a)
            if ap is None:
                continue
            d = math.hypot(bp[0] - ap[0], bp[1] - ap[1])
            if d < md:
                mi, md = i, d
        rec: dict[str, Any] = {"label": b.get("label"), "before_mm": [round(bp[0], 1), round(bp[1], 1)]}
        if mi is not None:
            used[mi] = True
            ap = _obj_xy(after[mi])
            rec["after_mm"] = [round(ap[0], 1), round(ap[1], 1)]
            rec["moved_mm"] = round(md, 1)
        else:
            rec["after_mm"] = None  # not seen afterwards (lifted away, or occluded)
        objects.append(rec)

    if not steps and not objects:
        return None
    return {"aimed_steps": steps, "objects": objects,
            "objects_before": len(before), "objects_after": len(after)}


def _outcome_hint(outcome: dict | None) -> str:
    """A short natural-language read of the measured outcome for the retry feedback."""
    if not outcome:
        return ""
    lines: list[str] = []
    for s in outcome.get("aimed_steps", []):
        na = s.get("nearest_after")
        if na:
            lines.append(
                f"- {s['step']} aimed at {s['aim_mm']} mm: nearest object after is "
                f"'{na['label']}' at {na['xy_mm']} mm — offset {na['offset_from_aim_mm']} mm "
                "from your aim (to land on it, shift your aim by that offset)."
            )
        else:
            lines.append(f"- {s['step']} aimed at {s['aim_mm']} mm: NOTHING detected near the aim afterwards.")
    for o in outcome.get("objects", []):
        if o.get("after_mm") is None:
            lines.append(f"- '{o['label']}' was at {o['before_mm']} mm and is NOT detected now (likely lifted/removed).")
        elif o.get("moved_mm", 0) >= 5:
            lines.append(f"- '{o['label']}' moved {o['moved_mm']} mm: {o['before_mm']} -> {o['after_mm']} mm.")
    return "\n".join(lines)


def _llm_verifier(client, model: str) -> Callable[..., dict[str, Any]]:
    """Default verifier: ask Claude whether the task is satisfied, from the snapshot.

    Pragmatic by design: with only joint/tip/gripper senses (no object detector
    yet), object-level success often can't be *proven*, so the prompt says to
    accept a cleanly-executed plan unless the snapshot actively contradicts it.
    Once a detection feed is added to the snapshot, the same call becomes
    detection-grounded and can catch genuine failures (missed grasp, wrong place).
    """
    report_tool = {
        "name": "report_verification",
        "description": "Report whether the robot task is now complete.",
        "input_schema": {
            "type": "object",
            "properties": {
                "satisfied": {"type": "boolean", "description": "True if the task is complete."},
                "reason": {"type": "string", "description": "Why you concluded that."},
                "suggestions": {"type": "string", "description": "If not satisfied, what to change next attempt."},
            },
            "required": ["satisfied", "reason"],
        },
    }
    system = (
        "You verify whether a tabletop robot-arm task has been accomplished. The arm "
        "has been moved HOME and the cameras re-detected the scene, so judge from the "
        "executed plan, the execution result, and the current sensor snapshot. The "
        "snapshot may include a `measured_outcome` field: objects re-detected AFTER the "
        "action, with each step's aim and how far the nearest object ended up from it, "
        "plus how far each object moved. Use those real numbers as your primary evidence "
        "of success/failure (e.g. a pick step whose object is still right at the aim = "
        "missed; a place whose object is >1 cm from the aim = off). "
        "Aim for REASONABLE precision: accept minor imperfection, but the goal must "
        "ACTUALLY be achieved. A few millimetres of offset or a slight lean is fine; "
        "being off by a centimetre or more, or ending in the wrong STATE, is NOT. "
        "Concretely: for a STACK, the upper object must be resting ON TOP of the lower "
        "one and clearly overlapping it — objects are only ~2.5 cm wide, so a top cube "
        "more than ~1 cm off-centre is hanging off, and one sitting a few cm away "
        "(e.g. 3-4 cm, on the table beside the stack) has clearly fallen off: "
        "INCOMPLETE. "
        "For a place/move, the object must be within about 1 cm of the intended spot "
        "and in the intended orientation (not knocked over). Mark INCOMPLETE when "
        "execution errored, the object was never moved/grasped, it fell off or over, or "
        "it is clearly displaced from where it should be; otherwise pass. Don't force a "
        "re-do over a few-millimetre cosmetic offset — but DO catch a real miss like an "
        "object that fell off a stack. If incomplete, say briefly what to change."
    )

    def verify(instruction, plan, results, snapshot, exec_error=None) -> dict[str, Any]:
        content = (
            f"Task: {instruction}\n\n"
            f"Plan executed: {json.dumps(plan, default=str)}\n\n"
            f"Execution error: {exec_error or 'none'}\n\n"
            f"Step results: {json.dumps(results, default=str)}\n\n"
            f"Current sensor snapshot: {json.dumps(snapshot, default=str)}\n\n"
            "Call report_verification."
        )
        resp = client.messages.create(
            model=model, max_tokens=1024, system=system,
            tools=[report_tool], tool_choice={"type": "tool", "name": "report_verification"},
            messages=[{"role": "user", "content": content}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "report_verification":
                data = dict(block.input or {})
                return {
                    "satisfied": bool(data.get("satisfied", False)),
                    "reason": data.get("reason", ""),
                    "suggestions": data.get("suggestions", ""),
                }
        # Defensive: if the model didn't call the tool, treat as unverified.
        return {"satisfied": False, "reason": "verifier returned no verdict", "suggestions": ""}

    return verify


# Inputs that yield task-relevant WORLD STATE — an object detector fed by the
# camera (e.g. the YOLO feed reporting what/where/size). Verification + retries
# only make sense when one of these can supply NEW information about where things
# are or whether the task actually succeeded. A raw camera FRAME does NOT count:
# the verifier reasons over a snapshot of structured readings, not an image, so it
# needs a detector's output — and a raw read would also false-positive on a
# laptop's built-in webcam, which isn't watching the workspace.
_VISION_SENSES = ("object_detections", "detections", "objects", "yolo")


def _vision_available() -> bool:
    """True if an object-detection / world-perception input is registered and usable.

    Without one, the only "senses" are the arm's own joint encoders, which can't
    tell whether a task succeeded or where an object is — so re-reasoning after a
    move would just be guessing from the same proprioception. In that case the
    brain runs the plan once, with no verification or retries. Once the detection
    feed is added to ``inputs/library/``, this flips on automatically.
    """
    names = {c["name"] for c in inputs.catalog()}
    for name in _VISION_SENSES:
        if name in names:
            try:
                reading = inputs.read(name)
            except Exception:
                continue
            # A sense reporting {"ok": False} (not actually producing detections) doesn't count.
            if not isinstance(reading, dict) or reading.get("ok", True):
                return True
    return False


def plan_and_run(
    instruction: str,
    arm: Any,
    model: str | None = None,
    urgent: bool = False,
    execute: bool = True,
    verbose: bool = True,
    log: bool = True,
    max_attempts: int = 3,
    verify: bool = True,
    client: Any = None,
    verifier: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Accomplish ``instruction`` via the plan -> execute -> verify -> retry cycle.

    Args:
        instruction: the natural-language task.
        arm: a connected :class:`~limbic.control.RobotArm`.
        model: force a model id, else routed by :func:`choose_model`.
        urgent: hint the router toward the fast model.
        execute: actually drive the arm (False = plan + return without moving).
        verbose: print progress.
        log: open a run folder for the whole cycle (movements + data + thinking).
        max_attempts: how many plan/execute/verify cycles before giving up.
        verify: run the post-execution verification + retry cycle. AUTOMATICALLY
            forced off when no camera/vision input is available — proprioception
            alone can't confirm a task, so the brain runs the plan once. False =
            always single shot.
        client: inject an Anthropic-style client (for tests / custom config). If
            None, a real ``anthropic.Anthropic()`` is created and an API key is
            required.
        verifier: a callable ``(instruction, plan, results, snapshot, exec_error)``
            -> ``{"satisfied", "reason", "suggestions"}``. Defaults to the LLM
            verifier; this is the seam where detection-based verification plugs in.

    Returns a dict: ``{"model", "instruction", "status", "plan", "rationale",
    "results", "attempts", "verification", "executed"}`` where ``status`` is
    ``"completed"``, ``"incomplete"`` (ran out of attempts), or ``"cannot_complete"``
    (no usable plan / refusal).

    Raises ``RuntimeError`` if no ``client`` is given and ``ANTHROPIC_API_KEY`` is
    unset.
    """
    import os

    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set, so the planner cannot reach Claude. "
                "Export your key to plan from natural language, or run "
                "`examples/run_mock_demo.py` to exercise the pipeline offline."
            )
        import anthropic  # lazy: keep the package importable without the dependency

        client = anthropic.Anthropic()

    inputs.set_context(arm=arm)
    chosen_model = model or choose_model(instruction, urgent=urgent)
    verify_fn = verifier or _llm_verifier(client, chosen_model)

    owns_run = log and not runlog.is_running()
    if owns_run:
        runlog.start_run(instruction, metadata={"component": "brain"})
    trail = runlog.current()

    tools = input_tools() + authoring_tools() + [submit_plan_tool()]
    system_prompt = build_system_prompt()
    messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]

    status = "cannot_complete"
    attempts: list[dict[str, Any]] = []
    final_plan: list[dict[str, Any]] = []
    final_rationale = ""
    final_results: list[Any] = []
    last_verdict: dict[str, Any] | None = None
    summary: dict[str, Any] = {"ok": False}

    try:
        trail.thought("instruction", instruction)
        trail.thought("model_choice", message=f"routed to {chosen_model}", model=chosen_model)
        if verbose:
            print(f"[brain] instruction: {instruction!r}\n[brain] model: {chosen_model}")

        # Verification + retries require EXTERNAL perception. With only the arm's
        # own encoders there is no new information to confirm where things are or
        # whether the task succeeded, so re-reasoning would be guessing — run once.
        if verify and not _vision_available():
            verify = False
            trail.thought(
                "verify_disabled",
                message="No camera/vision input — running single-shot (no verification, no retries).",
            )
            if verbose:
                print("[brain] no camera input -> verification + retries OFF (nothing new to perceive)")

        for attempt in range(1, max_attempts + 1):
            trail.thought("attempt", message=f"attempt {attempt}/{max_attempts}")
            if verbose:
                print(f"[brain] --- attempt {attempt}/{max_attempts} ---")

            # Collect what the planner detects this attempt (the "before" world
            # state), so the outcome can be measured against it after the action.
            perceptions: dict[str, Any] = {"prompts": set(), "objects": []}
            captured = _plan_turn(client, chosen_model, tools, system_prompt, messages, trail, verbose, perceptions)
            if captured is None:
                status = "cannot_complete"
                last_verdict = {"satisfied": False, "reason": "model produced no usable plan", "suggestions": ""}
                break

            final_plan = captured["plan"]
            final_rationale = captured["rationale"]

            if not execute:
                status = "completed"
                break

            results, exec_error = _execute(arm, final_plan, verbose)
            final_results = results
            attempts.append({"plan": final_plan, "results": results, "error": exec_error})

            if not verify:
                status = "completed"
                break

            # The plan ran — now CHECK it. Move the arm HOME first so it isn't
            # occluding the cameras (and home is the right resting pose after a
            # task), THEN re-detect to perceive the result. A user stop during the
            # homing still aborts; any other homing hiccup is non-fatal — we verify
            # from wherever the arm ended up.
            trail.thought("verify", message="task executed — homing, then re-checking with the camera")
            try:
                arm.go_home()
            except BaseException as exc:  # noqa: BLE001
                from ..control import MotionStopped

                if isinstance(exc, MotionStopped):
                    raise
                # otherwise ignore: homing is best-effort before the visual check

            # Re-detect the objects the planner located (now the arm is clear) and
            # MEASURE the outcome against where each step aimed — so verify + retry
            # have a quantified error, not just pass/fail. Folded into the snapshot
            # the verifier already reads.
            snapshot = inputs.snapshot()
            outcome = None
            try:
                if perceptions.get("prompts"):
                    after_objs = _redetect(perceptions["prompts"])
                    outcome = _measure_outcome(perceptions.get("objects", []), after_objs, final_plan)
            except Exception:
                outcome = None  # measurement must never break the run
            if outcome:
                snapshot["measured_outcome"] = outcome
                trail.thought("verify", message="measured outcome vs aim", outcome=outcome)

            verdict = verify_fn(instruction, final_plan, results, snapshot, exec_error)
            last_verdict = verdict
            trail.thought(
                "verify",
                message=("satisfied" if verdict["satisfied"] else "incomplete") + f": {verdict['reason']}",
                verdict=verdict,
            )
            if verbose:
                print(f"[brain] verify: {'SATISFIED' if verdict['satisfied'] else 'INCOMPLETE'} — {verdict['reason']}")

            if verdict["satisfied"]:
                status = "completed"
                break

            status = "incomplete"
            if attempt < max_attempts:
                hint = _outcome_hint(outcome)
                measured = (
                    "MEASURED OUTCOME (objects re-detected after the action — use these "
                    "real numbers to CORRECT your aim, this is how you improve on the last "
                    f"attempt):\n{hint}\n"
                    "Apply the offsets: if an object ended up +N mm in x of where you aimed, "
                    "the target was N mm short of your aim, so shift your next aim by that "
                    "vector. Do not just repeat the same coordinates.\n"
                ) if hint else ""
                messages.append({"role": "user", "content": (
                    "That attempt did NOT complete the task. "
                    f"Verifier reason: {verdict['reason']}. "
                    f"Suggestions: {verdict.get('suggestions') or '(none)'}. "
                    f"Execution error: {exec_error or 'none'}.\n"
                    f"{measured}"
                    f"Current sensor snapshot: {json.dumps(snapshot, default=str)}. "
                    "If an object may have MOVED since you last detected it (e.g. after "
                    "a push, a knock, or a failed grasp), call sense_object_detections "
                    "again to read its CURRENT position before planning — do not reuse a "
                    "stale coordinate. Revise your approach and submit a new plan. If a "
                    "needed capability is missing, create_primitive (or edit_primitive) "
                    "first, then submit_plan."
                )})

        summary = {"ok": status == "completed", "status": status, "attempts": len(attempts)}
        return {
            "model": chosen_model,
            "instruction": instruction,
            "status": status,
            "plan": final_plan,
            "rationale": final_rationale,
            "results": final_results,
            "attempts": attempts,
            "verification": last_verdict,
            "executed": execute and status in ("completed", "incomplete"),
        }
    except BaseException as exc:
        summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        if owns_run:
            runlog.end_run(summary=summary)
