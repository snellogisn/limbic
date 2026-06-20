"""Compose the planner's system prompt from the live registries.

The system prompt is assembled at call time from ``primitives.catalog()`` and
``inputs.catalog()`` so it always reflects the skills and senses that actually
exist — no hand-maintained list to drift. It carries three things the model must
internalise to plan good arm motions:

    * the table-frame coordinate convention (origin, axes, units, top-down pitch);
    * the hardware grasp rules learned from this gripper (claw offset, descend
      INTO the object, lift before retract, stay near the workspace centre);
    * the full primitive catalog and input catalog, with parameters, so the model
      knows exactly what it can do and perceive.

It closes with the contract: perceive with ``sense_*`` tools only if needed, then
call ``submit_plan`` exactly once with an ordered list of steps.
"""

from __future__ import annotations

from typing import Any

from ..inputs import registry as inputs
from ..primitives import registry as primitives

# Reuse the single source of truth for the frame description.
from .tools import TABLE_FRAME_NOTE

# Physical grasp heuristics for this tabletop gripper. These encode hard-won
# hardware reality the model cannot infer from the primitive names alone.
GRASP_RULES = """\
Hardware grasp rules (this gripper, learned the hard way):
- Claw offset: the gripper closes slightly off-centre, so aim a few millimetres
  to one side of the object's reported centre rather than dead-on.
- Descend INTO the object: lower to grasp height that is at or slightly below the
  top of the object so the fingers close around it, not above it.
- Lift before retract: after closing on an object, lift straight up first, THEN
  move laterally. Dragging sideways at grasp height knocks things over.
- Stay near the workspace centre: keep targets close to the centre of the
  reachable area; near the edges IK is strained and grasps are unreliable.
- Open the gripper before reaching down to grasp; close on the object; open again
  to release after lowering at the destination."""


def _format_primitive_catalog(catalog: list[dict[str, Any]]) -> str:
    """Render the primitive catalog as readable lines with parameters."""
    if not catalog:
        return "(no primitives are registered yet)"
    lines: list[str] = []
    for prim in catalog:
        lines.append(f"- {prim['name']}: {prim.get('summary', '')}")
        for arg, spec in (prim.get("parameters") or {}).items():
            spec = spec or {}
            required = "required" if "default" not in spec else f"default={spec['default']!r}"
            lines.append(
                f"    * {arg} ({spec.get('type', 'string')}, {required}): "
                f"{spec.get('description', '')}"
            )
    return "\n".join(lines)


def _format_input_catalog(catalog: list[dict[str, Any]]) -> str:
    """Render the sensory-input catalog, noting the ``sense_`` tool prefix."""
    if not catalog:
        return "(no senses are registered yet — plan without perception)"
    lines: list[str] = []
    for sense in catalog:
        lines.append(
            f"- sense_{sense['name']}: {sense.get('summary', '')}"
        )
        for arg, spec in (sense.get("parameters") or {}).items():
            spec = spec or {}
            required = "required" if "default" not in spec else f"default={spec['default']!r}"
            lines.append(
                f"    * {arg} ({spec.get('type', 'string')}, {required}): "
                f"{spec.get('description', '')}"
            )
    return "\n".join(lines)


def build_system_prompt() -> str:
    """Build the full planner system prompt from the live catalogs."""
    primitive_catalog = _format_primitive_catalog(primitives.catalog())
    input_catalog = _format_input_catalog(inputs.catalog())

    return f"""\
You are the planning mind of `limbic`, a tabletop robot arm. Your job is to turn \
a natural-language instruction into a single ordered plan of motion primitives \
that the arm will execute safely.

COORDINATE FRAME
{TABLE_FRAME_NOTE}

{GRASP_RULES}

MOTION PRIMITIVES YOU CAN CHAIN (use these names exactly):
{primitive_catalog}

SENSES YOU CAN QUERY (call the tool named sense_<name>):
{input_catalog}

HOW TO WORK
1. If the instruction depends on something you cannot know for certain (where an \
object is, whether the gripper holds something), call the relevant sense_* tools \
FIRST to perceive it. If the instruction already gives you everything (explicit \
coordinates), you may plan directly without perceiving.
2. Reason about the grasp rules above when choosing coordinates and ordering steps.
3. Then call `submit_plan` EXACTLY ONCE with an ordered list of steps, each being \
a primitive name and its arguments, that accomplishes the instruction. Provide \
only arguments the primitive declares; omit arguments that have a default unless \
you want to override them. Include a short rationale.

Do not narrate at length and do not ask the user questions — perceive if needed, \
then submit one plan."""
