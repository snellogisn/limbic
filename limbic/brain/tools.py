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

from .._core import required_args
from ..inputs import registry as inputs
from ..primitives import authoring
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
        properties[arg] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    # "required" uses the one shared definition (no "default" key) — same rule the
    # Capability base validates against, so the schema and the runtime check agree.
    required = required_args(parameters)
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


def authoring_tools() -> list[dict[str, Any]]:
    """Tools that let the planner extend the skill set when it's stuck.

    The primitive library is *dynamic*: if no existing primitive accomplishes a
    step, the model can write a new one (``create_primitive``) or fix an existing
    one (``edit_primitive``). Each takes the full Python file contents for a single
    :class:`~limbic.primitives.base.Primitive` subclass; the brain validates and
    hot-reloads it (rolling back on any error) before it can be used in a plan.
    """
    contract = (
        "Provide `name` (lowercase identifier, e.g. 'slide_left') and `code` (the "
        "FULL contents of the .py file). The file must define ONE Primitive "
        "subclass whose `name` equals the name you pass, with `summary`, a "
        "`parameters` dict ({arg: {type, description, default?}}; an arg without a "
        "default is required), and a `run(self, arm, **kwargs)` that calls ONLY "
        "RobotArm methods (move_to_xyz, reach_above, descend_to, lift_by, "
        "open_gripper, close_gripper, set_joint, go_home) so safety + smoothing are "
        "inherited. After it validates you may reference it in submit_plan. "
        "Template:\n" + authoring.describe_template()
    )
    code_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The primitive's unique name."},
            "code": {"type": "string", "description": "Full Python file contents for the primitive."},
        },
        "required": ["name", "code"],
    }
    return [
        {
            "name": "create_primitive",
            "description": "Create a NEW motion primitive when no existing one fits. " + contract,
            "input_schema": code_schema,
        },
        {
            "name": "edit_primitive",
            "description": "Replace an EXISTING motion primitive (to fix or improve it). " + contract,
            "input_schema": code_schema,
        },
    ]


def submit_plan_tool() -> dict[str, Any]:
    """The single ``submit_plan`` tool: Claude's one ordered list of steps.

    The plan is an array of ``{"primitive": <name>, "args": object}`` items. The
    primitive name is a free string (NOT an enum) on purpose: the model may create
    a new primitive mid-task with ``create_primitive``, and an enum frozen at
    tool-build time couldn't include it. We validate every name against the live
    registry in the orchestrator before any motion, and feed any mistake back for
    the model to fix. ``args`` is a free-form object validated per-primitive there
    too. A ``rationale`` string captures the model's reasoning for the record.

    The description tells the model the table frame, the current skill names, and
    the workflow: perceive with ``sense_*`` first if needed, create a primitive if
    a capability is missing, then submit one plan.
    """
    primitive_names = sorted(primitives.all_primitives().keys())
    names_note = (
        f" Currently available primitives: {', '.join(primitive_names)}."
        if primitive_names
        else ""
    )

    primitive_property: dict[str, Any] = {
        "type": "string",
        "description": (
            "Name of the motion primitive to run for this step (an existing one, "
            "or one you just created with create_primitive)."
        ),
    }

    return {
        "name": "submit_plan",
        "description": (
            "Submit the FINAL ordered plan that accomplishes the instruction. "
            "Call this EXACTLY ONCE, after any perception and after creating any "
            "primitive you need. The plan is an ordered list of primitive steps "
            "executed top to bottom. " + TABLE_FRAME_NOTE + names_note
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
