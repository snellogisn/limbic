"""limbic — a cross-platform, LLM-driven control stack for a tabletop robot arm.

Three layers, each its own subpackage:

    control/     The Body   -- movement, gripper, guardrails (RobotArm)
    primitives/  The Skills -- reusable motion primitives the LLM chains
    inputs/      The Senses -- motor + camera readings the LLM can query
    brain/       The Mind   -- turns an instruction into a list of primitive calls

Everything runs on macOS, Windows and Linux, with no physical arm required: the
control layer auto-falls-back to a software mock so the whole pipeline is
develop-and-test-anywhere.

Quick start:
    from limbic import RobotArm
    with RobotArm() as arm:        # auto: real arm if present, else mock
        arm.go_home()
        arm.move_to_xyz(180, 0, 60)
        arm.close_gripper()
"""

from . import runlog
from .control import RobotArm, load_config

__all__ = ["RobotArm", "load_config", "runlog"]
__version__ = "0.1.0"
