"""Print a full profile of THIS machine: arch, arm serial port, cameras-by-name,
and which capabilities (real arm / vision / kinematics) are installed here.

Run this whenever the rig moves to a new computer — it's the "plug in, profile,
go" check. Equivalent to `python -m limbic.platform_support`.

    python scripts/describe_machine.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from limbic.platform_support import format_profile

if __name__ == "__main__":
    print(format_profile())
