"""brain — The Mind: turn a natural-language instruction into a validated plan.

This subpackage is the top of the limbic stack. Given a plain-English
instruction ("pick up the block at (160, 40) and place it at (160, -40)"), it:

    1. Shows Claude the *live* catalog of motion primitives (the skills) and
       sensory inputs (the senses), described straight from the registries so
       the prompt is never stale.
    2. Lets Claude *perceive* first — it may call ``sense_*`` tools (camera,
       motor state, ...) to learn about the world before committing.
    3. Has Claude emit ONE ordered plan: a ``list[{"primitive", "args"}]`` via a
       single ``submit_plan`` tool call, plus a short rationale.
    4. VALIDATES that plan against the registry (every primitive exists, every
       required argument is present) *before* anything moves.
    5. Executes it through ``primitives.run_sequence.run_plan`` so the arm's
       safety layer (workspace clamp + soft limits + smoothing) always applies —
       our code, never the model, drives the hardware.

Why a tool-use loop rather than asking for JSON? Tool schemas let Claude both
*act* (query senses) and *commit* (submit the plan) with structured, validated
arguments, and they keep us — not the model — in control of execution. Why route
between models? A quick "go home" should not pay for deep reasoning; a spatial
multi-step stack should. See :func:`choose_model`.

Public surface:
    plan_and_run(instruction, arm, ...)  -- the full perceive-plan-validate-run flow
    choose_model(instruction, urgent)    -- pick a fast vs. capable model
"""

from __future__ import annotations

from .orchestrator import choose_model, plan_and_run

__all__ = ["plan_and_run", "choose_model"]
