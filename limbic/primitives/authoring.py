"""Let the brain author motion primitives at runtime — create new ones, edit old.

This is what makes the primitive library *dynamic*: when the planner hits a task
it can't accomplish with the existing skills, it can write a brand-new primitive
(or revise an obsolete one) as a single file in ``library/``, and have it become
immediately available. That capability is exposed to Claude as the
``create_primitive`` / ``edit_primitive`` tools in the brain.

Every write is guarded:
    * the name must be a safe identifier (no path tricks, one file in library/);
    * after writing, the file is imported and the registry is hot-reloaded, and
      we confirm it actually defined a usable :class:`Primitive` named ``name``;
    * if anything fails (syntax error, wrong shape, import blows up), we ROLL BACK
      to the previous file content — or delete a failed new file — and return the
      error so the caller can fix it. A broken write never corrupts the library.

Security note: authored code runs in this process with your permissions — this is
the same trust model as letting an agent edit files in the repo. It is deliberate
(the whole point is dynamic skills); there is no sandbox. Keep that in mind if you
ever expose this to untrusted input.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

from . import registry
from .base import Primitive

# Where primitive files live and the module path they import under.
_LIBRARY_DIR = Path(__file__).resolve().parent / "library"
_LIBRARY_PKG = "limbic.primitives.library"

# A valid primitive name == a valid module name == a valid Python identifier-ish
# slug. This is what stops "../../evil" or "os.system" style names.
_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


# A copy-pasteable template the brain can fill in. Returned by describe_template()
# so the model has the exact shape in front of it.
TEMPLATE = '''\
"""<one-line description of what this primitive does and why>."""

from __future__ import annotations

from typing import Any

from ..base import Primitive
from ...control import RobotArm


class {class_name}(Primitive):
    name = "{name}"
    summary = "<short catalog summary>"
    parameters = {{
        # "x_mm": {{"type": "float", "description": "..."}},                # required
        # "height_mm": {{"type": "float", "description": "...", "default": 60}},  # optional
    }}

    def run(self, arm: RobotArm, **kwargs: Any) -> Any:
        # Call ONLY RobotArm methods so safety + smoothing are inherited, e.g.:
        #   arm.move_to_xyz(x_mm, y_mm, z_mm)   arm.open_gripper()   arm.lift_by(dz)
        ...
'''


def describe_template() -> str:
    """Return the primitive-file template (for the create_primitive tool prompt)."""
    return TEMPLATE


def _safe_path(name: str) -> Path:
    """Resolve ``name`` to a file directly inside library/, or raise."""
    if not _SAFE_NAME.match(name):
        raise ValueError(
            f"invalid primitive name {name!r}: use lowercase letters, digits and "
            "underscores, starting with a letter (e.g. 'slide_left')."
        )
    path = (_LIBRARY_DIR / f"{name}.py").resolve()
    if path.parent != _LIBRARY_DIR.resolve():
        raise ValueError(f"refusing to write outside the primitive library: {path}")
    return path


def _reload_into_registry(name: str) -> Primitive:
    """Import/reload the named module, rebuild the registry, return the primitive.

    Raises ``KeyError``/``ValueError`` if the file didn't define a registered
    primitive of that name (so the caller can roll back).
    """
    modname = f"{_LIBRARY_PKG}.{name}"
    # importlib.import_module returns the CACHED module for an already-imported
    # file, so an edit wouldn't take effect — reload explicitly when present.
    if modname in sys.modules:
        importlib.reload(sys.modules[modname])
    else:
        importlib.import_module(modname)

    registry.reload()
    prims = registry.all_primitives()
    if name not in prims:
        raise ValueError(
            f"the file did not register a primitive named '{name}'. Make sure the "
            f"class subclasses Primitive and sets name = \"{name}\"."
        )
    primitive = prims[name]
    if not isinstance(primitive, Primitive):
        raise ValueError(f"'{name}' is registered but is not a Primitive instance.")
    return primitive


def _purge_module(name: str) -> None:
    sys.modules.pop(f"{_LIBRARY_PKG}.{name}", None)


def _save_primitive(name: str, code: str, *, must_exist: bool) -> dict[str, Any]:
    """Shared create/edit worker: write, validate, register, or roll back."""
    try:
        path = _safe_path(name)
    except ValueError as exc:
        return {"ok": False, "name": name, "error": str(exc)}

    existed = path.exists()
    if must_exist and not existed:
        return {
            "ok": False,
            "name": name,
            "error": f"no primitive '{name}' to edit. Use create_primitive to add it.",
        }
    if not must_exist and existed:
        return {
            "ok": False,
            "name": name,
            "error": f"primitive '{name}' already exists. Use edit_primitive to change it.",
        }

    backup = path.read_text(encoding="utf-8") if existed else None
    try:
        path.write_text(code, encoding="utf-8")
        primitive = _reload_into_registry(name)
    except Exception as exc:  # syntax error, bad shape, import failure, ...
        # Roll back to a known-good state so the library is never left broken.
        _purge_module(name)
        if backup is not None:
            path.write_text(backup, encoding="utf-8")
            try:
                _reload_into_registry(name)
            except Exception:
                registry.reload()
        else:
            path.unlink(missing_ok=True)
            registry.reload()
        return {
            "ok": False,
            "name": name,
            "error": f"{type(exc).__name__}: {exc}",
            "rolled_back": True,
        }

    return {
        "ok": True,
        "name": name,
        "path": str(path),
        "action": "edited" if existed else "created",
        "summary": primitive.summary,
        "parameters": primitive.parameters,
    }


def create_primitive(name: str, code: str) -> dict[str, Any]:
    """Create a NEW primitive file ``library/<name>.py`` from ``code``.

    ``code`` is the full contents of the file — a module defining one
    :class:`Primitive` subclass whose ``name`` equals ``name``. Returns
    ``{"ok": True, ...}`` on success (and the primitive is immediately usable in a
    plan), or ``{"ok": False, "error": ...}`` if it didn't validate (with the file
    rolled back / removed).
    """
    return _save_primitive(name, code, must_exist=False)


def edit_primitive(name: str, code: str) -> dict[str, Any]:
    """Replace an EXISTING primitive file with ``code`` (e.g. to fix/improve it).

    Same validation and rollback as :func:`create_primitive`; on any failure the
    previous, working version is restored so a bad edit can't break the skill.
    """
    return _save_primitive(name, code, must_exist=True)
