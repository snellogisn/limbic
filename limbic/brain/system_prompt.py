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


def _format_catalog(
    catalog: list[dict[str, Any]], empty_message: str, name_prefix: str = ""
) -> str:
    """Render a primitive/input catalog as readable lines with parameters.

    Shared by both catalogs; ``name_prefix`` is "" for primitives and "sense_"
    for inputs (the tool name the model calls). ``empty_message`` is shown when
    nothing is registered.
    """
    if not catalog:
        return empty_message
    lines: list[str] = []
    for item in catalog:
        lines.append(f"- {name_prefix}{item['name']}: {item.get('summary', '')}")
        for arg, spec in (item.get("parameters") or {}).items():
            spec = spec or {}
            required = "required" if "default" not in spec else f"default={spec['default']!r}"
            lines.append(
                f"    * {arg} ({spec.get('type', 'string')}, {required}): "
                f"{spec.get('description', '')}"
            )
    return "\n".join(lines)


def build_system_prompt() -> str:
    """Build the full planner system prompt from the live catalogs."""
    primitive_catalog = _format_catalog(
        primitives.catalog(), empty_message="(no primitives are registered yet)"
    )
    input_catalog = _format_catalog(
        inputs.catalog(),
        empty_message="(no senses are registered yet — plan without perception)",
        name_prefix="sense_",
    )

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

DYNAMIC SKILLS
The primitive set above is not fixed. If NO existing primitive can accomplish a \
step, you may CREATE a new one with `create_primitive` (or fix an existing one \
with `edit_primitive`) — give it a name and the full Python file for a Primitive \
subclass whose `run(self, arm, **kwargs)` calls only RobotArm methods. It is \
validated and registered immediately; then reference it in your plan. Prefer \
existing primitives; invent one only when there's a real capability gap.

HOW TO WORK
1. If the instruction depends on something you cannot know for certain (where an \
object is, whether the gripper holds something), call the relevant sense_* tools \
FIRST to perceive it. If the instruction already gives you everything (explicit \
coordinates), you may plan directly without perceiving.
2. Reason about the grasp rules above when choosing coordinates and ordering steps.
3. If a needed skill is missing, create_primitive first.
4. Then call `submit_plan` EXACTLY ONCE with an ordered list of steps, each being \
a primitive name and its arguments. Provide only arguments the primitive declares; \
omit arguments that have a default unless you want to override them. Include a \
short rationale.

After execution your work is VERIFIED. If the task is judged incomplete, you will \
be told why and given a fresh sensor snapshot, and asked to try again — revise the \
plan (or author a better primitive) to actually complete it. Do not narrate at \
length and do not ask the user questions — perceive if needed, then submit a plan."""
