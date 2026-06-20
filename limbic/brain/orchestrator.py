"""The orchestrator: run the perceive-plan-validate-execute loop with Claude.

This is the heart of the brain. :func:`plan_and_run` wires the tool definitions
and the live system prompt into a manual Anthropic agentic loop, lets Claude
query senses (``sense_*`` tools) and then commit a plan (``submit_plan``),
validates that plan against the registry, and — keeping execution in *our* hands
so the arm's safety layer always applies — runs it through the sequence runner.

:func:`choose_model` routes between a fast model (urgent / trivial single-action
instructions) and the most capable model (complex, multi-step, spatial-reasoning
instructions). The Anthropic SDK is imported lazily inside the functions so this
module imports cleanly even where ``anthropic`` is not installed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .. import runlog
from ..inputs import registry as inputs
from ..primitives import registry as primitives
from .system_prompt import build_system_prompt
from .tools import SENSE_PREFIX, input_tools, submit_plan_tool

# Literal model IDs — never date-suffixed. The brain defaults to the most capable
# model and steps down to the fast one only when routing says so.
MODEL_FAST = "claude-haiku-4-5"
MODEL_CAPABLE = "claude-opus-4-8"

# Single-word action verbs that are unambiguous one-shot commands; if an
# instruction is essentially just one of these, the fast model is plenty.
_SIMPLE_VERBS = {"home", "open", "close", "stop", "rest", "park", "release", "grip"}

# Above this many words an instruction is almost certainly multi-step / spatial
# enough to want the capable model.
_SHORT_WORD_LIMIT = 4


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

    # Signals that the task is genuinely multi-step / spatial — these force the
    # capable model even if the instruction happens to be short.
    complex_signals = (
        " then ",
        " and ",
        ",",
        "stack",
        "place",
        "left of",
        "right of",
        "on top",
        "next to",
        "between",
        "(",
    )
    if any(signal in f" {text} " for signal in complex_signals):
        return MODEL_CAPABLE

    # A lone obvious verb ("home", "open the gripper", "stop now").
    if words and words[0] in _SIMPLE_VERBS and len(words) <= _SHORT_WORD_LIMIT:
        return MODEL_FAST

    # Very short instructions with no complexity signals — treat as simple.
    if len(words) <= _SHORT_WORD_LIMIT:
        return MODEL_FAST

    # Default: the most capable model.
    return MODEL_CAPABLE


def _validate_plan(plan: Any) -> list[dict[str, Any]]:
    """Validate a submitted plan against the registry, returning clean steps.

    Checks shape (a list of ``{"primitive", "args"}``), that every primitive name
    exists in the registry, and that each step supplies all required arguments
    (those whose schema lacks a ``"default"``). Raises ``ValueError`` with a clear
    message on the first problem — *before* anything is allowed to move.
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
                f"plan step {index} names unknown primitive '{name}'. "
                f"Available: {', '.join(sorted(available)) or '(none registered)'}"
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


def plan_and_run(
    instruction: str,
    arm: Any,
    model: str | None = None,
    urgent: bool = False,
    execute: bool = True,
    verbose: bool = True,
    log: bool = True,
) -> dict[str, Any]:
    """Plan ``instruction`` with Claude, validate, and (optionally) execute it.

    Flow:
        1. Wire the live arm into the input registry (so motor/camera senses can
           reach it) via ``inputs.set_context(arm=arm)``.
        2. Build the perception + submit_plan tools and the system prompt from the
           live catalogs; choose the model via :func:`choose_model` if not given.
        3. Run a manual Anthropic tool-use loop: dispatch every ``sense_*`` call
           to ``inputs.read(...)``, and capture the single ``submit_plan`` call.
        4. Validate the captured plan against the registry.
        5. If ``execute`` is True, run it through ``run_sequence.run_plan(arm, plan)``
           so the arm's safety layer drives the hardware — never the model.

    Returns a dict:
        ``{"model", "instruction", "plan", "rationale", "perceptions", "results",
          "executed"}``.

    Logging: when ``log`` is True and no run is already active, this opens a run
    (``logs/<timestamp>-<instruction>/``) for the duration and records the whole
    decision trail — model choice, the model's reasoning, every sense query, the
    submitted plan, and its execution — into the thinking/data/movement streams.
    If a run is already active, it logs into that one instead.

    Raises ``RuntimeError`` (with guidance to the offline demo) if
    ``ANTHROPIC_API_KEY`` is not set, and ``ValueError`` if no/invalid plan comes
    back.
    """
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set, so the planner cannot reach Claude. "
            "Export your key (export ANTHROPIC_API_KEY=...) to plan from natural "
            "language, or run `examples/run_mock_demo.py` to exercise the full "
            "pipeline offline (it runs a fixed plan on the mock arm)."
        )

    import anthropic  # lazy: keep the package importable without the dependency

    # Senses (motor state, camera) can now reach the live arm.
    inputs.set_context(arm=arm)

    chosen_model = model or choose_model(instruction, urgent=urgent)

    # Open a run so the full decision trail AND the resulting motion are recorded
    # together — unless one is already active (then we just log into it), or
    # logging was switched off. The run is closed in the finally below.
    owns_run = log and not runlog.is_running()
    if owns_run:
        runlog.start_run(label=instruction, metadata={"component": "brain"})
    trail = runlog.current()

    summary: dict[str, Any] = {"ok": False, "error": "incomplete"}
    try:
        trail.thought("instruction", instruction)
        trail.thought(
            "model_choice",
            message=f"routed to {chosen_model}",
            model=chosen_model,
            auto_selected=model is None,
            urgent=urgent,
        )

        tools = input_tools() + [submit_plan_tool()]
        system_prompt = build_system_prompt()

        if verbose:
            print(f"[brain] instruction: {instruction!r}")
            print(f"[brain] model: {chosen_model}")

        client = anthropic.Anthropic()
        messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]

        perceptions: list[dict[str, Any]] = []
        submitted: dict[str, Any] | None = None

        # The fast model path keeps things lean; the capable path turns on adaptive
        # thinking for the spatial reasoning these plans need.
        create_kwargs: dict[str, Any] = {
            "model": chosen_model,
            "max_tokens": 16000,
            "tools": tools,
            "system": system_prompt,
        }
        if chosen_model == MODEL_CAPABLE:
            create_kwargs["thinking"] = {"type": "adaptive"}

        while True:
            resp = client.messages.create(messages=messages, **create_kwargs)

            # A refusal carries no usable content to parse — surface it plainly.
            if resp.stop_reason == "refusal":
                trail.thought("refused", message="model declined to plan this instruction")
                raise RuntimeError(
                    "the model refused to plan this instruction; rephrase and retry"
                )

            # Record any visible reasoning / narration the model produced this turn,
            # so the thinking log captures *why*, not just the final plan.
            for block in resp.content:
                kind = getattr(block, "type", None)
                if kind == "thinking" and getattr(block, "thinking", ""):
                    trail.thought("reasoning", block.thinking)
                elif kind == "text" and getattr(block, "text", ""):
                    trail.thought("message", block.text)

            if resp.stop_reason == "end_turn":
                # The model finished without committing a plan.
                break

            messages.append({"role": "assistant", "content": resp.content})

            tool_results: list[dict[str, Any]] = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue

                if block.name == "submit_plan":
                    # Capture the plan; echo a trivial result so the loop can close.
                    submitted = dict(block.input)
                    trail.thought(
                        "plan_submitted",
                        message=submitted.get("rationale", ""),
                        plan=submitted.get("plan"),
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps({"status": "plan received"}),
                        }
                    )
                elif block.name.startswith(SENSE_PREFIX):
                    sense_name = block.name[len(SENSE_PREFIX):]
                    try:
                        reading = inputs.read(sense_name, **(block.input or {}))
                        result_payload: Any = {"reading": reading}
                    except Exception as exc:  # report sense failures back to the model
                        result_payload = {"error": str(exc)}
                    perceptions.append({"sense": sense_name, "result": result_payload})
                    # The sensor reading itself lands in data.jsonl via inputs.read;
                    # here we note the brain's DECISION to query it.
                    trail.thought(
                        "perceive",
                        message=f"queried sense '{sense_name}'",
                        sense=sense_name,
                        result=result_payload,
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result_payload, default=str),
                        }
                    )
                    if verbose:
                        print(f"[brain] perceived {sense_name}: {result_payload}")
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps({"error": f"unknown tool {block.name}"}),
                            "is_error": True,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

            # Once the plan is in hand we have what we need; stop looping.
            if submitted is not None:
                break

        if submitted is None:
            trail.thought("no_plan", message="model ended without calling submit_plan")
            raise ValueError(
                "the model ended without calling submit_plan; no plan was produced"
            )

        plan = _validate_plan(submitted.get("plan"))
        rationale = submitted.get("rationale", "")
        trail.thought(
            "plan_validated",
            message=f"{len(plan)} step(s) validated against the registry",
            plan=plan,
            rationale=rationale,
        )

        if verbose:
            print(f"[brain] validated plan ({len(plan)} step(s)); rationale: {rationale}")

        results: list[Any] = []
        executed = False
        if execute:
            # Execute via the sequence runner so the arm's safety layer always
            # applies — our code drives the hardware, never the model directly.
            from ..primitives import run_sequence

            trail.thought("execute", message=f"executing {len(plan)} step(s)")
            results = run_sequence.run_plan(arm, plan, verbose=verbose)
            executed = True

        summary = {
            "ok": True,
            "model": chosen_model,
            "steps": len(plan),
            "executed": executed,
        }
        return {
            "model": chosen_model,
            "instruction": instruction,
            "plan": plan,
            "rationale": rationale,
            "perceptions": perceptions,
            "results": results,
            "executed": executed,
        }
    except BaseException as exc:
        summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        if owns_run:
            runlog.end_run(summary=summary)
