"""Per-run logging: a folder that records everything a run did and thought.

Every "run" — whether you drive the arm by hand, execute a saved plan, or let the
LLM brain plan one — can be captured in its own timestamped folder under
``logs/``. Three streams are recorded, so a run can be replayed, audited, or
debugged after the fact:

    movements.jsonl   every motion and its achieved result (the WHAT-IT-DID)
    data.jsonl        every sensor/input reading and state snapshot (the WHAT-IT-SAW)
    thinking.jsonl    the brain's model choice, reasoning, tool calls, and final
                      plan (the WHY-IT-DID-IT)        + a readable thinking.md mirror
    run.json          run metadata + a summary written when the run closes

Design:
    * Logging is tied to an explicit RUN. Until you ``start_run()`` (or use the
      ``run()`` context manager), the active logger is a no-op — direct,
      one-off use of ``RobotArm`` doesn't litter the disk.
    * One global "current run" so every layer (control, inputs, brain) can log
      without threading a logger object through every call. ``current()`` returns
      the active logger or a null logger.
    * Records are JSON Lines (one JSON object per line) for easy streaming/parsing,
      plus a human-readable ``thinking.md`` for the decision trail.
    * Pure standard library, cross-platform; never raises into the caller (a
      logging failure must not stop the arm).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _slug(text: str, max_len: int = 40) -> str:
    """Turn a free-text label into a filesystem-safe slug for the run folder."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return (cleaned[:max_len].rstrip("-")) or "run"


class RunLogger:
    """Writes the movement / data / thinking streams for a single run."""

    #: channel name -> filename
    _FILES = {
        "movements": "movements.jsonl",
        "data": "data.jsonl",
        "thinking": "thinking.jsonl",
    }

    def __init__(self, run_dir: Path, metadata: dict[str, Any] | None = None):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._start = time.monotonic()
        self._seq = 0
        self._counts = {channel: 0 for channel in self._FILES}
        self._metadata = dict(metadata or {})
        self._closed = False

        # Seed run.json and the readable thinking trail.
        self._write_run_json(status="running")
        self._thinking_md = self.run_dir / "thinking.md"
        title = self._metadata.get("label", self.run_dir.name)
        self._append_text(self._thinking_md, f"# Thinking log — {title}\n\n")

    # ------------------------------------------------------------------ #
    # Public logging surface (one method per stream)
    # ------------------------------------------------------------------ #
    def movement(self, action: str, **fields: Any) -> None:
        """Record one motion event (e.g. a move, a gripper actuation, a home)."""
        self._emit("movements", {"action": action, **fields})

    def data(self, source: str, reading: Any, **fields: Any) -> None:
        """Record one sensor/input reading or state snapshot."""
        self._emit("data", {"source": source, "reading": reading, **fields})

    def thought(self, phase: str, message: str = "", **fields: Any) -> None:
        """Record one reasoning/decision event from the brain.

        ``phase`` is a short tag ("model_choice", "perceive", "reasoning",
        "plan", "execute", ...). Everything is also appended to ``thinking.md``
        as a readable block.
        """
        record = {"phase": phase, "message": message, **fields}
        self._emit("thinking", record)
        self._append_thinking_md(record)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self, summary: dict[str, Any] | None = None) -> None:
        """Finalise the run: write the summary + counts into run.json."""
        if self._closed:
            return
        self._closed = True
        if summary:
            self._metadata["summary"] = summary
        self._write_run_json(status="finished")

    # ------------------------------------------------------------------ #
    # Internals — every write is best-effort and never raises outward
    # ------------------------------------------------------------------ #
    def _emit(self, channel: str, payload: dict[str, Any]) -> None:
        self._seq += 1
        self._counts[channel] += 1
        record = {
            "seq": self._seq,
            "t": _utc_now_iso(),
            "elapsed_s": round(time.monotonic() - self._start, 4),
            **payload,
        }
        line = json.dumps(record, default=str)
        self._append_text(self.run_dir / self._FILES[channel], line + "\n")

    def _write_run_json(self, status: str) -> None:
        meta = {
            "run": self.run_dir.name,
            "status": status,
            "started_at": self._metadata.get("started_at"),
            "updated_at": _utc_now_iso(),
            "record_counts": dict(self._counts),
            **self._metadata,
        }
        self._append_text(self.run_dir / "run.json", "", overwrite=json.dumps(meta, indent=2, default=str))

    def _append_thinking_md(self, record: dict[str, Any]) -> None:
        phase = record.get("phase", "")
        message = record.get("message", "")
        extras = {k: v for k, v in record.items() if k not in ("phase", "message")}
        block = f"## {phase}\n\n"
        if message:
            block += f"{message}\n\n"
        if extras:
            block += "```json\n" + json.dumps(extras, indent=2, default=str) + "\n```\n\n"
        self._append_text(self._thinking_md, block)

    @staticmethod
    def _append_text(path: Path, text: str, overwrite: str | None = None) -> None:
        try:
            if overwrite is not None:
                path.write_text(overwrite, encoding="utf-8")
            else:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(text)
        except OSError:
            # Logging must never break a run; swallow disk errors quietly.
            pass


class _NullLogger:
    """No-op logger returned when no run is active — same surface, does nothing."""

    def movement(self, *args: Any, **kwargs: Any) -> None: ...
    def data(self, *args: Any, **kwargs: Any) -> None: ...
    def thought(self, *args: Any, **kwargs: Any) -> None: ...
    def close(self, *args: Any, **kwargs: Any) -> None: ...


_NULL = _NullLogger()
_CURRENT: RunLogger | None = None


def base_log_dir() -> Path:
    """Where run folders are created. Override with ``$LIMBIC_LOG_DIR``."""
    return Path(os.environ.get("LIMBIC_LOG_DIR", "logs"))


def start_run(
    label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunLogger:
    """Begin a new run: create ``logs/<timestamp>-<label>/`` and make it current.

    Returns the :class:`RunLogger`. Any subsequent ``current()`` call (from the
    control, inputs, or brain layers) writes into this run until :func:`end_run`.
    """
    global _CURRENT
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = f"{stamp}-{_slug(label or 'run')}"
    meta = {
        "label": label or "run",
        "started_at": _utc_now_iso(),
        **(metadata or {}),
    }
    _CURRENT = RunLogger(base_log_dir() / folder, metadata=meta)
    return _CURRENT


def current() -> RunLogger | _NullLogger:
    """Return the active run logger, or a no-op logger if no run is running."""
    return _CURRENT if _CURRENT is not None else _NULL


def is_running() -> bool:
    """True if a run is currently active (i.e. logging is going somewhere real)."""
    return _CURRENT is not None


def end_run(summary: dict[str, Any] | None = None) -> None:
    """Finalise and detach the current run (no-op if none is active)."""
    global _CURRENT
    if _CURRENT is not None:
        _CURRENT.close(summary=summary)
        _CURRENT = None


@contextlib.contextmanager
def run(label: str | None = None, metadata: dict[str, Any] | None = None) -> Iterator[RunLogger]:
    """Context manager around a run: ``with runlog.run("pick demo") as log: ...``.

    Starts a run on entry and closes it on exit (even if the body raises),
    recording the exception in the summary if one occurred.
    """
    logger = start_run(label=label, metadata=metadata)
    summary: dict[str, Any] = {"ok": True}
    try:
        yield logger
    except BaseException as exc:  # record then re-raise — don't swallow
        summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        end_run(summary=summary)
