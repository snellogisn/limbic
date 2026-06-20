"""RobotArm: the stable, safe motion API that everything else is built on.

This is the contract the motion primitives and the LLM brain call. It owns:
    * connection lifecycle (context manager: ``with RobotArm() as arm: ...``)
    * Cartesian moves (table-frame mm) via the pure-Python IK
    * the gripper (open/close on a 0..100 scale)
    * single-joint moves and home
    * SAFETY on every path: workspace clamp + per-joint soft limits, always
    * SMOOTH motion: ease-in/ease-out interpolation so nothing jerks

It does NOT know about hardware specifics — that's the backend's job — so the
exact same code drives the mock simulator on a laptop and the real SO-101 over
USB. Keep these method signatures stable: they are the tool surface the LLM
composes plans from.
"""

from __future__ import annotations

import math
import time

from .. import runlog
from .backends import HardwareBackend, make_backend
from .config import (
    CONVERGE_TOL_DEG,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    GRIPPER_SETTLE_S,
    SLOW_DT_S,
    SLOW_STEP_DEG,
    SMOOTH_DT_S,
    SMOOTH_STEP_DEG,
    ArmConfig,
    load_config,
)
from .kinematics import forward_kinematics, solve_ik
from .safety import (
    ARM_JOINTS,
    GRIPPER_JOINT,
    JOINT_SOFT_LIMITS,
    clamp_joint,
    clamp_to_workspace,
)

# "Home" = every motor at the CENTRE of its range. The arm joints' soft limits
# are symmetric, so their midpoint is 0 deg; the gripper's midpoint is its
# halfway opening. go_home() drives all of them there.
HOME_POSE: dict[str, float] = {j: 0.0 for j in ARM_JOINTS}
_GRIPPER_HOME: float = sum(JOINT_SOFT_LIMITS[GRIPPER_JOINT]) / 2.0


class RobotArm:
    """High-level, safety-wrapped controller for one arm (real or simulated)."""

    def __init__(
        self,
        config: ArmConfig | None = None,
        backend: HardwareBackend | None = None,
        verbose: bool = True,
        logger=None,
    ):
        """Create a controller.

        Args:
            config: Connection/motion config. Defaults to env-driven ``load_config()``.
            backend: Inject a specific backend (e.g. a mock in tests). If omitted,
                one is chosen from ``config.backend`` ("auto"/"real"/"mock").
            verbose: Print motion/connection events (handy for the mock).
            logger: Optional explicit :class:`~limbic.runlog.RunLogger`. If omitted,
                movements are recorded to whatever run is active (``runlog.current()``),
                so any motion during a ``runlog.run(...)`` block is captured.
        """
        self.config = config or load_config()
        self.backend = backend or make_backend(self.config, verbose=verbose)
        self._verbose = verbose
        self._logger = logger

    def _runlog(self):
        """The logger to record to: the explicit one, else the active run (or null)."""
        return self._logger or runlog.current()

    # ------------------------------------------------------------------ #
    # Connection / context manager
    # ------------------------------------------------------------------ #
    def connect(self) -> "RobotArm":
        self.backend.connect()
        return self

    def disconnect(self) -> None:
        self.backend.disconnect()

    def __enter__(self) -> "RobotArm":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # ------------------------------------------------------------------ #
    # State / sensing
    # ------------------------------------------------------------------ #
    def read_joints(self) -> dict[str, float]:
        """Current joint positions ``{name: degrees}`` (gripper on 0..100)."""
        return self.backend.read_joints()

    def current_xyz(self) -> tuple[float, float, float]:
        """Current tool-tip position in the table frame (mm), via forward kinematics."""
        return forward_kinematics(self.read_joints())

    # ------------------------------------------------------------------ #
    # Cartesian primitives (table frame, mm) — the core tool surface
    # ------------------------------------------------------------------ #
    def move_to_xyz(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        approach_pitch_deg: float = -90.0,
        slow: bool = False,
    ) -> tuple[float, float, float]:
        """Move the tool tip to table-frame ``(x, y, z)`` mm.

        Args:
            x_mm, y_mm, z_mm: Target position (origin under the pan axis, +x
                forward, +y left, +z up).
            approach_pitch_deg: Tool angle; ``-90`` = straight down (default).
            slow: Use the finer/slower precision profile (for descend/grasp/place).

        The target is clamped into the safe workspace first, then solved with IK,
        then each joint is clamped to its soft limit, then the move is streamed
        smoothly. Out-of-reach requests stop at the nearest reachable point rather
        than raising. Returns the achieved tip position.
        """
        cx, cy, cz, was_clamped = clamp_to_workspace(x_mm, y_mm, z_mm)
        if was_clamped and self._verbose:
            print(
                f"  [safety] target ({x_mm:.0f},{y_mm:.0f},{z_mm:.0f}) outside "
                f"workspace -> nearest ({cx:.0f},{cy:.0f},{cz:.0f})"
            )

        solution = solve_ik(cx, cy, cz, approach_pitch_deg)
        if not solution.reachable and self._verbose:
            print("  [safety] target beyond reach; extending toward it as far as possible")

        self._drive_to(solution.joints, slow=slow)
        achieved = self.current_xyz()
        self._runlog().movement(
            "move_to_xyz",
            requested={"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm, "pitch_deg": approach_pitch_deg},
            target={"x_mm": round(cx, 2), "y_mm": round(cy, 2), "z_mm": round(cz, 2)},
            achieved={"x_mm": round(achieved[0], 2), "y_mm": round(achieved[1], 2), "z_mm": round(achieved[2], 2)},
            clamped=was_clamped,
            reachable=solution.reachable,
            slow=slow,
        )
        return achieved

    def reach_above(
        self, x_mm: float, y_mm: float, height_mm: float = 70.0
    ) -> tuple[float, float, float]:
        """Hover above table point ``(x, y)`` at ``height_mm``, ready to descend."""
        return self.move_to_xyz(x_mm, y_mm, height_mm)

    def descend_to(
        self, x_mm: float, y_mm: float, z_mm: float
    ) -> tuple[float, float, float]:
        """Lower the tip to grasp height ``z`` at ``(x, y)`` using the precision profile."""
        return self.move_to_xyz(x_mm, y_mm, z_mm, slow=True)

    def lift_by(self, dz_mm: float) -> tuple[float, float, float]:
        """Raise (or lower, if negative) the current tip position by ``dz`` mm."""
        x, y, z = self.current_xyz()
        return self.move_to_xyz(x, y, z + dz_mm)

    # ------------------------------------------------------------------ #
    # Joint-space primitives
    # ------------------------------------------------------------------ #
    def set_joint(self, name: str, degrees: float) -> dict[str, float]:
        """Move a single arm joint to ``degrees`` (others held), soft-limit clamped."""
        target = self.read_joints()
        target[name] = clamp_joint(name, degrees)
        # Drop the gripper from the smooth move; it's actuated separately.
        self._drive_to({j: target[j] for j in ARM_JOINTS})
        joints = self.read_joints()
        self._runlog().movement(
            "set_joint", joint=name, requested_deg=degrees, achieved_deg=round(joints[name], 2)
        )
        return joints

    def go_home(self) -> dict[str, float]:
        """Move every motor to the centre of its range (the neutral home pose).

        Home = all arm joints at 0 deg (their soft-limit midpoint) AND the
        gripper at its halfway opening — every motor centred. The arm is moved
        first, then the gripper is actuated in isolation (§0.6 gripper rule).
        """
        self._drive_to(HOME_POSE)
        self._set_gripper(_GRIPPER_HOME)
        joints = self.read_joints()
        self._runlog().movement(
            "go_home", achieved_joints={k: round(v, 2) for k, v in joints.items()}
        )
        return joints

    # ------------------------------------------------------------------ #
    # Gripper
    # ------------------------------------------------------------------ #
    def open_gripper(self) -> None:
        """Open the gripper fully."""
        self._set_gripper(GRIPPER_OPEN)

    def close_gripper(self) -> None:
        """Close the gripper to its grip position."""
        self._set_gripper(GRIPPER_CLOSED)

    def set_gripper(self, percent_open: float) -> None:
        """Set the gripper to an explicit 0..100 opening (0 = closed, 100 = open)."""
        self._set_gripper(clamp_joint(GRIPPER_JOINT, percent_open))

    def _set_gripper(self, value: float) -> None:
        # Hold the arm joints, move only the gripper, and let it fully actuate.
        command = {j: self.read_joints()[j] for j in ARM_JOINTS}
        command[GRIPPER_JOINT] = float(value)
        self.backend.send_joints(command)
        time.sleep(GRIPPER_SETTLE_S)
        self._runlog().movement("gripper", percent_open=float(value))

    # ------------------------------------------------------------------ #
    # Smooth interpolated motion (shared by every Cartesian/joint move)
    # ------------------------------------------------------------------ #
    def _drive_to(self, target_joints: dict[str, float], slow: bool = False) -> None:
        """Stream a smooth ease-in/ease-out trajectory to ``target_joints``.

        Interpolates from the current pose to the (soft-limit-clamped) target in
        fine sub-steps with a cosine velocity profile, then holds the goal until
        the joints converge. The gripper is preserved at its current value.
        """
        current = self.read_joints()
        gripper = current.get(GRIPPER_JOINT, GRIPPER_OPEN)

        target = {j: clamp_joint(j, float(target_joints[j])) for j in ARM_JOINTS}
        step = SLOW_STEP_DEG if slow else SMOOTH_STEP_DEG
        dt = SLOW_DT_S if slow else SMOOTH_DT_S

        max_travel = max(abs(target[j] - current[j]) for j in ARM_JOINTS)
        n_steps = max(1, math.ceil(max_travel / step))

        for i in range(1, n_steps + 1):
            ease = 0.5 - 0.5 * math.cos(math.pi * i / n_steps)  # 0 -> 1, smooth
            command = {
                j: current[j] + (target[j] - current[j]) * ease for j in ARM_JOINTS
            }
            command[GRIPPER_JOINT] = gripper
            self.backend.send_joints(command)
            time.sleep(dt)

        # Converge: the streamed trajectory is open-loop, so settle onto the goal.
        for _ in range(25):
            actual = self.read_joints()
            if max(abs(target[j] - actual[j]) for j in ARM_JOINTS) <= CONVERGE_TOL_DEG:
                break
            command = dict(target)
            command[GRIPPER_JOINT] = gripper
            self.backend.send_joints(command)
            time.sleep(0.02)
