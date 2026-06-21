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
import re
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
def _dispatch_tool(block: Any, trail, verbose: bool) -> tuple[dict[str, Any], dict | None]:
    """Run one tool_use block. Returns (tool_result, captured_plan_or_None).

    ``captured_plan`` is non-None only for a valid ``submit_plan`` (a dict with
    ``plan`` + ``rationale``). Senses dispatch to ``inputs.read``; create/edit
    dispatch to the authoring helpers (which hot-reload the registry).
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
        except Exception as exc:
            payload = {"error": str(exc)}
        trail.thought("perceive", message=f"queried sense '{sense}'", sense=sense, result=payload)
        if verbose:
            print(f"[brain] perceived {sense}: {payload}")
        return result(payload), None

    return result({"error": f"unknown tool {name}"}, is_error=True), None


def _plan_turn(client, model, tools, system, messages, trail, verbose) -> dict | None:
    """Drive Claude until it commits a valid plan (or gives up). Mutates ``messages``.

    Returns ``{"plan", "rationale"}`` on success, or ``None`` if the model ended
    its turn without a usable plan. Perception, authoring, and plan-rejection
    feedback all happen here, so a single "attempt" can self-heal (e.g. create a
    missing primitive, or fix a bad argument) before it ever moves the arm.
    """
    create_kwargs: dict[str, Any] = {
        "model": model, "max_tokens": 16000, "tools": tools, "system": system,
    }
    if model == MODEL_CAPABLE:
        create_kwargs["thinking"] = {"type": "adaptive"}

    for _ in range(_MAX_PLAN_TURNS):
        resp = client.messages.create(messages=messages, **create_kwargs)

        if resp.stop_reason == "refusal":
            trail.thought("refused", message="model declined to plan this instruction")
            raise RuntimeError("the model refused to plan this instruction; rephrase and retry")

        # Capture any visible reasoning/narration for the thinking log.
        for block in resp.content:
            kind = getattr(block, "type", None)
            if kind == "thinking" and getattr(block, "thinking", ""):
                trail.thought("reasoning", block.thinking)
            elif kind == "text" and getattr(block, "text", ""):
                trail.thought("message", block.text)

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            # No tools called and the turn ended -> no plan produced.
            return None

        messages.append({"role": "assistant", "content": resp.content})

        tool_results: list[dict[str, Any]] = []
        captured: dict | None = None
        for block in tool_uses:
            tool_result, plan = _dispatch_tool(block, trail, verbose)
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
        "executed plan, the execution result, and the current sensor snapshot. "
        "Be LENIENT and practical, NOT a perfectionist. 'Close enough' counts as "
        "done: real grasps and placements land a centimetre or two off, objects lean, "
        "stacks aren't perfectly aligned — none of that is failure. Pass the task as "
        "long as the goal is essentially achieved (e.g. for a stack, the object is on "
        "top of the other even if slightly off-centre; for a move, it's roughly at the "
        "target). Only mark it INCOMPLETE on a CLEAR, unambiguous failure — execution "
        "errored, the object was never moved/grasped, it's nowhere near the target, or "
        "it fell on the floor. When in doubt, treat it as complete. Do NOT demand a "
        "re-do for small positional imperfection — over-correcting usually makes it "
        "worse. If it really is incomplete, say briefly what to change."
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

            captured = _plan_turn(client, chosen_model, tools, system_prompt, messages, trail, verbose)
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

            snapshot = inputs.snapshot()
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
                messages.append({"role": "user", "content": (
                    "That attempt did NOT complete the task. "
                    f"Verifier reason: {verdict['reason']}. "
                    f"Suggestions: {verdict.get('suggestions') or '(none)'}. "
                    f"Execution error: {exec_error or 'none'}. "
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
