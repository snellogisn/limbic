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
    GRIPPER_MAX_S,
    GRIPPER_OPEN,
    GRIPPER_POLL_S,
    GRIPPER_SETTLE_S,
    GRIPPER_STOP_TOL,
    HOME_SETTLE_S,
    HOME_STEP_DEG,
    MOVE_SETTLE_S,
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


# How close to the OPEN / CLOSED endpoints still counts as that named state.
_GRIPPER_STATE_TOL: float = 5.0


def _gripper_state_label(target: float | None) -> str:
    """Clean text for a commanded claw target: open / closed / partial / unknown."""
    if target is None:
        return "unknown"
    if target <= GRIPPER_CLOSED + _GRIPPER_STATE_TOL:
        return "closed"
    if target >= GRIPPER_OPEN - _GRIPPER_STATE_TOL:
        return "open"
    return "partial"


class MotionStopped(Exception):
    """Raised inside a move when a stop was requested (a cooperative e-stop).

    Stopping just means we stop streaming new setpoints; torque stays engaged, so
    the arm HOLDS the pose it had reached instead of going limp. The exception
    unwinds the current primitive/plan so callers can report "stopped".
    """


class RobotArm:
    """High-level, safety-wrapped controller for one arm (real or simulated)."""

    def __init__(
        self,
        config: ArmConfig | None = None,
        backend: HardwareBackend | None = None,
        verbose: bool = True,
        logger=None,
        stop_event=None,
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
            stop_event: Optional ``threading.Event`` (or anything with ``is_set()``).
                When set DURING a move, the move stops streaming and raises
                :class:`MotionStopped`, holding the pose. Lets another thread (e.g.
                a web "Stop" request) interrupt motion. See :meth:`bind_stop`.
        """
        self.config = config or load_config()
        self.backend = backend or make_backend(self.config, verbose=verbose)
        self._verbose = verbose
        self._logger = logger
        self._stop_event = stop_event
        # The COMMANDED claw target (0..100), our source of truth for the claw
        # state. We hold THIS through arm moves, never the read-back: a claw
        # clamped on an object reads its blocked position (~14%), and
        # re-commanding that = zero error = zero grip force = the object slips.
        # Holding the commanded target keeps the clamp pressure on. None until
        # the first open/close (then we hold the read-back, preserving old behaviour).
        self._gripper_target: float | None = None

    def bind_stop(self, stop_event) -> "RobotArm":
        """Attach a stop signal (``threading.Event``-like) checked during moves."""
        self._stop_event = stop_event
        return self

    def _check_stop(self) -> None:
        """Raise :class:`MotionStopped` if a stop was requested.

        We do NOT send any further setpoints: the servos hold the last commanded
        position under torque, so the arm freezes in place rather than dropping.
        """
        if self._stop_event is not None and self._stop_event.is_set():
            if self._verbose:
                print("  [stop] motion stopped by request — holding pose")
            self._runlog().movement("stopped", reason="stop requested")
            raise MotionStopped("motion stopped by request")

    def _settle(self, seconds: float = MOVE_SETTLE_S) -> None:
        """Pause briefly so the just-commanded motion physically completes.

        Open-loop servos lag their setpoints: even after we stop streaming, a
        joint keeps coasting onto its target, and the CLAW especially can take
        ~half a second to physically reach a commanded open/close. A short settle
        between key points (the end of each move, and after each claw actuation)
        keeps motions DISCRETE — the next action starts from a truly settled
        pose, which is what makes the claw act in real isolation (§0.6) instead
        of actuating mid-drift. Torque stays engaged, so the arm holds during it.
        """
        if seconds <= 0:
            return
        self._check_stop()
        time.sleep(seconds)

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
            # The pitch the wrist ACTUALLY reached (mink tilts as it reaches out).
            # When it's off -90 the IK pulled the target in to cancel the claw
            # overhang so the claw — not the bare tip — lands on the point.
            achieved_pitch_deg=round(solution.achieved_pitch_deg, 1),
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
        self._drive_to(HOME_POSE, coarse=True)  # close enough — speed over precision
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
        """Open the claw ALL THE WAY (state -> OPEN), in isolation.

        Drives to ``GRIPPER_OPEN`` (fully open) so a held object drops cleanly,
        with the arm held still. The only time the claw should move while the arm
        moves is a deliberate dynamic release (e.g. a throw), which is its own
        primitive — the normal open/close are always isolated.
        """
        self._set_gripper(GRIPPER_OPEN)

    def close_gripper(self) -> None:
        """Close the claw ALL THE WAY (state -> CLOSED), in isolation.

        Drives to ``GRIPPER_CLOSED`` (fully closed = 0). With nothing in the claw
        the fingers shut completely; with an object the servo stalls against it
        and keeps PRESSURE on (we hold this commanded target through the
        subsequent lift, so the grip never goes slack and the object can't slip).
        """
        self._set_gripper(GRIPPER_CLOSED)

    def set_gripper(self, percent_open: float) -> None:
        """Set the claw to an explicit 0..100 opening (0 = closed, 100 = open)."""
        self._set_gripper(clamp_joint(GRIPPER_JOINT, percent_open))

    @property
    def gripper_state(self) -> str:
        """The commanded claw state as clean text: 'open', 'closed', 'partial',
        or 'unknown' (before any open/close has been commanded)."""
        return _gripper_state_label(self._gripper_target)

    @property
    def is_gripping(self) -> bool:
        """True once the claw has been commanded CLOSED (i.e. holding/clamping)."""
        return self._gripper_target is not None and self._gripper_target <= GRIPPER_CLOSED + _GRIPPER_STATE_TOL

    def _set_gripper(self, value: float) -> None:
        """Actuate the claw to ``value`` (0..100) in STRICT ISOLATION, and only
        return once it has physically finished.

        Two §0.6 rules are enforced here, in hardware terms:

        * **Isolation.** The arm joints are read once and held FIXED at exactly
          their current position for the whole actuation — nothing on the arm
          moves while the hand opens or closes.
        * **Actually done before moving on.** The claw is one slow servo with no
          "finished" signal, and it's slower than any fixed sleep. So we DRIVE it
          and watch its read-back until it physically STOPS — it reached full
          open/close, or stalled/clamped on an object — then a final settle. Only
          then do we return, so the next arm move can't start mid-grip (which is
          what looked like "grabbing and lifting at the same time"). For a free
          close/open that means it goes ALL THE WAY; against an object it clamps
          and holds.
        """
        # Don't start actuating the claw if a stop was requested.
        self._check_stop()
        value = float(value)
        # Record the commanded target FIRST — it's the claw's source of truth and
        # what every following arm move will hold (keeps clamp pressure on).
        self._gripper_target = value
        # Snapshot the arm pose ONCE and hold exactly that; only the gripper moves.
        command = {j: self.read_joints()[j] for j in ARM_JOINTS}
        command[GRIPPER_JOINT] = value

        # Keep commanding the claw until its read-back goes quiet (two near-still
        # reads in a row = it has stopped travelling), or we hit the hard cap.
        prev: float | None = None
        quiet = 0
        deadline = time.time() + GRIPPER_MAX_S
        while True:
            self._check_stop()
            self.backend.send_joints(command)
            time.sleep(GRIPPER_POLL_S)
            now = self.read_joints().get(GRIPPER_JOINT, value)
            if prev is not None and abs(now - prev) <= GRIPPER_STOP_TOL:
                quiet += 1
                if quiet >= 2:
                    break
            else:
                quiet = 0
            prev = now
            if time.time() >= deadline:
                break

        # Final settle so the clamp pressure stabilises before anything else moves.
        self._settle(GRIPPER_SETTLE_S)
        state = _gripper_state_label(value)
        if self._verbose:
            print(f"  [claw] {state.upper()} (commanded {value:.0f}% open)")
        self._runlog().movement("gripper", percent_open=value, state=state)

    # ------------------------------------------------------------------ #
    # Smooth interpolated motion (shared by every Cartesian/joint move)
    # ------------------------------------------------------------------ #
    def _drive_to(self, target_joints: dict[str, float], slow: bool = False,
                  coarse: bool = False) -> None:
        """Stream a smooth ease-in/ease-out trajectory to ``target_joints``.

        Interpolates from the current pose to the (soft-limit-clamped) target in
        fine sub-steps with a cosine velocity profile, then holds the goal until
        the joints converge. The gripper is preserved at its current value.

        ``coarse`` trades precision for speed (used by ``go_home``): bigger
        sub-steps and NO fine convergence pass — the open-loop landing is "close
        enough" for a neutral home, and skipping the convergence retries + long
        settle is most of the time saving.
        """
        current = self.read_joints()
        # Hold the COMMANDED claw target, not the read-back: if the claw is
        # clamped on an object it reads its blocked position, and re-commanding
        # that releases the grip. Falls back to read-back only before the first
        # explicit open/close (target is None).
        gripper = self._gripper_target
        if gripper is None:
            gripper = current.get(GRIPPER_JOINT, GRIPPER_OPEN)

        target = {j: clamp_joint(j, float(target_joints[j])) for j in ARM_JOINTS}
        step = HOME_STEP_DEG if coarse else (SLOW_STEP_DEG if slow else SMOOTH_STEP_DEG)
        dt = SLOW_DT_S if slow else SMOOTH_DT_S

        max_travel = max(abs(target[j] - current[j]) for j in ARM_JOINTS)
        n_steps = max(1, math.ceil(max_travel / step))

        for i in range(1, n_steps + 1):
            self._check_stop()  # cooperative e-stop: freeze before the next setpoint
            ease = 0.5 - 0.5 * math.cos(math.pi * i / n_steps)  # 0 -> 1, smooth
            command = {
                j: current[j] + (target[j] - current[j]) * ease for j in ARM_JOINTS
            }
            command[GRIPPER_JOINT] = gripper
            self.backend.send_joints(command)
            time.sleep(dt)

        # Converge: the streamed trajectory is open-loop, so settle onto the goal.
        # Skipped in coarse mode — "close enough" is the goal for a neutral home.
        if not coarse:
            for _ in range(25):
                self._check_stop()
                actual = self.read_joints()
                if max(abs(target[j] - actual[j]) for j in ARM_JOINTS) <= CONVERGE_TOL_DEG:
                    break
                command = dict(target)
                command[GRIPPER_JOINT] = gripper
                self.backend.send_joints(command)
                time.sleep(0.02)

        # Settle between key points: hold here a beat so the arm physically
        # arrives before the next action (e.g. a claw close) begins — keeps each
        # point-to-point move discrete and the claw acting in isolation (§0.6).
        # Coarse home only needs a brief settle.
        self._settle(HOME_SETTLE_S if coarse else MOVE_SETTLE_S)
