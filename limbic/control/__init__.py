"""The Body: cross-platform robot control — movement, gripper, and guardrails.

Public surface:
    RobotArm     -- the safe, smooth motion controller (the tool surface)
    ArmConfig / load_config -- env-driven connection + motion config
    safety       -- joint soft limits + workspace clamps (the single guardrail source)
    solve_ik / forward_kinematics -- pure-Python kinematics (no binary deps)
    make_backend / MockBackend / RealBackend -- the hardware abstraction
"""

from . import safety
from .arm import HOME_POSE, MotionStopped, RobotArm
from .backends import HardwareBackend, MockBackend, RealBackend, make_backend
from .config import ArmConfig, load_config
from .kinematics import IKSolution, forward_kinematics, solve_ik

__all__ = [
    "RobotArm",
    "MotionStopped",
    "HOME_POSE",
    "ArmConfig",
    "load_config",
    "safety",
    "solve_ik",
    "forward_kinematics",
    "IKSolution",
    "make_backend",
    "HardwareBackend",
    "MockBackend",
    "RealBackend",
]
