"""Motion primitives: the reusable arm skills the LLM picks, chains, and authors.

    base      -- the Primitive contract (one skill per file)
    registry  -- auto-discovers everything in library/ and exposes the catalog
    library/  -- the actual primitives, one per file (browse/extend this folder)

The LLM brain reads ``registry.catalog()`` to see what's available, builds a list
of primitive calls, and the sequence runner executes them. New primitives are
added by dropping a file in ``library/``; nothing else needs editing.
"""

from . import registry
from .base import Primitive

__all__ = ["Primitive", "registry"]
