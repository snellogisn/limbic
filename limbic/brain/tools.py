"""Build the Anthropic tool definitions Claude uses to perceive and to plan.

Two kinds of tools are exposed to the model:

    * One ``sense_<name>`` tool per sensory input, so Claude can *perceive* the
      world (query the camera, read motor state, ...) before it commits. These
      are generated from the input registry's catalog, so adding a new sense in
      ``inputs/library/`` automatically gives the planner a new tool.
    * Exactly ONE ``submit_plan`` tool, the model's single act of *committing*:
      an ordered list of primitive calls plus a rationale. Constraining the
      primitive name to an ``enum`` of the live registry means the model can
      only name skills that actually exist, and our validation in the
      orchestrator catches the rest (required-arg presence) before any motion.

Everything here is derived from the *live* registries (``primitives.catalog()``,
``inputs.catalog()``, ``primitives.all_primitives()``) so the tool surface never
drifts from the real skills/senses. Import is side-effect free and tolerates an
empty catalog (the libraries may still be populating) — callers just get fewer
tools, never a crash.
"""

from __future__ import annotations

from typing import Any

from ..inputs import registry as inputs
from ..primitives import registry as primitives

# Prefix for the perception tools, so an input named "camera" becomes the tool
# "sense_camera" and can never collide with "submit_plan" or each other.
SENSE_PREFIX = "sense_"

# Map our lightweight ``parameters`` type strings onto JSON-Schema types. Our
# primitives/inputs describe args with friendly names ("float", "int", ...);
# Claude's input_schema wants JSON-Schema types. Anything unknown falls through
# to "string", the safe, lossless default.
_TYPE_MAP: dict[str, str] = {
    "float": "number",
    "double": "number",
    "number": "number",
    "int": "integer",
    "integer": "integer",
    "bool": "boolean",
    "boolean": "boolean",
    "str": "string",
    "string": "string",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}


def _params_to_input_schema(parameters: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Convert a primitive/input ``parameters`` dict to a JSON-Schema object.

    Each parameter ``{arg: {"type", "description", "default"?}}`` becomes a
    property; ``required`` lists every arg that lacks a ``"default"`` key (our
    convention for "the caller must supply this"). Types are mapped through
    :data:`_TYPE_MAP`; descriptions and any default are carried across so the
    model sees the full intent of each argument.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for arg, spec in parameters.items():
        spec = spec or {}
        prop: dict[str, Any] = {
            "type": _TYPE_MAP.get(str(spec.get("type", "string")).lower(), "string"),
        }
        if "description" in spec:
            prop["description"] = spec["description"]
        if "default" in spec:
            # Surface the default so the model knows the arg is optional and what
            # value it falls back to.
            prop["default"] = spec["default"]
        else:
            required.append(arg)
        properties[arg] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def input_tools() -> list[dict[str, Any]]:
    """One Anthropic tool per sensory input, named ``sense_<name>``.

    The description is the input's one-line summary, and the input_schema is
    derived from its declared query parameters. Returns an empty list if no
    senses are registered yet — the planner then simply has nothing to perceive
    with and must plan blind, which is fine for instructions that need no
    perception.
    """
    tools: list[dict[str, Any]] = []
    for sense in inputs.catalog():
        tools.append(
            {
                "name": f"{SENSE_PREFIX}{sense['name']}",
                "description": (
                    sense.get("summary", "")
                    or f"Read the '{sense['name']}' sense and return its current value."
                ),
                "input_schema": _params_to_input_schema(sense.get("parameters", {})),
            }
        )
    return tools


# The table-frame convention is repeated to the model in three places (this tool
# description, the submit_plan description, and the system prompt) on purpose:
# spatial framing is the single most important and most easily forgotten fact, so
# it is kept close to wherever the model is about to emit coordinates.
TABLE_FRAME_NOTE = (
    "All coordinates are in the table frame, in millimetres: the origin is "
    "directly under the pan (base rotation) axis, +x points forward away from "
    "the base, +y points to the arm's left, and +z points up. A top-down grasp "
    "uses approach pitch -90 degrees (tool pointing straight down)."
)


def submit_plan_tool() -> dict[str, Any]:
    """The single ``submit_plan`` tool: Claude's one ordered list of steps.

    The plan is an array of ``{"primitive": <enum>, "args": object}`` items,
    where the ``primitive`` enum is the set of names currently in the registry —
    so the model literally cannot name a non-existent skill. ``args`` is left as
    a free-form object (the per-primitive schemas vary and are validated by us in
    the orchestrator before execution). A ``rationale`` string captures the
    model's reasoning for the record.

    The description tells the model the table frame and the workflow: perceive
    with ``sense_*`` tools first *only if needed*, then call this once.
    """
    primitive_names = sorted(primitives.all_primitives().keys())

    primitive_property: dict[str, Any] = {
        "type": "string",
        "description": "Name of the motion primitive to run for this step.",
    }
    # Only constrain to an enum when there actually are primitives; an empty enum
    # would reject every plan, so degrade gracefully if the library is still
    # loading.
    if primitive_names:
        primitive_property["enum"] = primitive_names

    return {
        "name": "submit_plan",
        "description": (
            "Submit the FINAL ordered plan that accomplishes the instruction. "
            "Call this EXACTLY ONCE, and only after any perception you need. "
            "Use the sense_* tools first if you must perceive the world (e.g. "
            "locate an object or check the gripper); if the instruction is fully "
            "specified you may plan directly. The plan is an ordered list of "
            "primitive steps executed top to bottom. " + TABLE_FRAME_NOTE
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "description": "Ordered list of primitive calls to execute.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "primitive": primitive_property,
                            "args": {
                                "type": "object",
                                "description": (
                                    "Arguments for the primitive, matching its "
                                    "parameter schema. Omit args that have a default."
                                ),
                            },
                        },
                        "required": ["primitive", "args"],
                    },
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief explanation of why this plan accomplishes the instruction.",
                },
            },
            "required": ["plan", "rationale"],
        },
    }
