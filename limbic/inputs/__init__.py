"""The Senses: sensory inputs the LLM can query while solving an instruction.

    base      -- the Input contract (one sense per file, read-only)
    registry  -- auto-discovers everything in library/, exposes the catalog,
                 and injects runtime context (e.g. the connected arm)
    library/  -- the actual senses, one per file (motors, cameras, ...)

The LLM browses ``registry.catalog()`` to see what perceptions are available and
calls ``registry.read(name, ...)`` to use them. New senses are added by dropping
a file in ``library/``.
"""

from . import registry
from .base import Input

__all__ = ["Input", "registry"]
