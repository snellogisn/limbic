"""Hardware backends: one interface, a real arm and a simulated arm behind it.

This is the seam that makes the whole project develop-anywhere. ``RobotArm`` (see
``arm.py``) never talks to a motor directly — it talks to a ``HardwareBackend``.
There are two implementations:

    * ``MockBackend`` — pure software. Holds joint state in a dict, logs every
      command, and uses forward kinematics to report a believable tip position.
      Runs on a bare Mac with no arm, no SDK, no serial port. This is what you
      develop and test the LLM pipeline against.

    * ``RealBackend`` — wraps the LeRobot ``SO101Follower`` and drives real
      motors over the auto-detected serial port. Imported lazily so the package
      still works when ``lerobot`` isn't installed.

``make_backend(config)`` picks one:
    backend="mock" -> always mock
    backend="real" -> always real (errors clearly if it can't connect)
    backend="auto" -> real if a port is found AND lerobot imports, else mock
"""

from __future__ import annotations

import abc
import os

from .config import ArmConfig
from .kinematics import forward_kinematics
from .safety import ARM_JOINTS, GRIPPER_JOINT


class HardwareBackend(abc.ABC):
    """Minimal joint-level interface every backend must provide.

    Deliberately tiny: read joints, send a joint command, connect/disconnect.
    All the smarts (IK, interpolation, safety, primitives) live above this in
    ``RobotArm`` so they're shared by every backend.
    """

    #: Human-readable name for logs/UI, e.g. "mock" or "real (SO101 @ COM7)".
    name: str = "backend"

    @abc.abstractmethod
    def connect(self) -> None:
        """Open the connection (no-op for the mock)."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""

    @abc.abstractmethod
    def read_joints(self) -> dict[str, float]:
        """Return ``{joint_name: position}`` for all joints (degrees; gripper 0..100)."""

    @abc.abstractmethod
    def send_joints(self, command: dict[str, float]) -> None:
        """Send one joint-position command. Caller guarantees it's already clamped."""


class MockBackend(HardwareBackend):
    """In-memory simulated arm. No hardware, no dependencies, runs everywhere."""

    def __init__(self, verbose: bool = True):
        self.name = "mock"
        self._verbose = verbose
        # Start at the centred home pose with the gripper open.
        self._state: dict[str, float] = {j: 0.0 for j in ARM_JOINTS}
        self._state[GRIPPER_JOINT] = 100.0
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        if self._verbose:
            print("[mock] connected (simulated arm — no hardware in the loop)")

    def disconnect(self) -> None:
        self._connected = False
        if self._verbose:
            print("[mock] disconnected")

    def read_joints(self) -> dict[str, float]:
        return dict(self._state)

    def send_joints(self, command: dict[str, float]) -> None:
        # The simulated arm reaches commands instantly and perfectly.
        self._state.update(command)
        if self._verbose:
            arm = {j: round(self._state[j], 1) for j in ARM_JOINTS}
            x, y, z = forward_kinematics(self._state)
            print(
                f"[mock] joints={arm} grip={self._state[GRIPPER_JOINT]:.0f} "
                f"tip=({x:.0f},{y:.0f},{z:.0f})mm"
            )


class RealBackend(HardwareBackend):
    """Drives a physical LeRobot SO-101 follower arm over USB serial.

    Imports ``lerobot`` lazily inside ``connect`` so that merely *constructing*
    this object (or importing the module) never requires the SDK — only actually
    connecting does.
    """

    def __init__(self, config: ArmConfig):
        if config.port is None:
            raise RuntimeError(
                "RealBackend requires a serial port but none was found. Set "
                "$LIMBIC_PORT (e.g. COM7 on Windows, /dev/cu.usbserial-XX on "
                "macOS) or plug in the arm. Use backend='mock' to develop "
                "without hardware."
            )
        self._config = config
        self._robot = None
        self.name = f"real (SO101 @ {config.port})"

    def connect(self) -> None:
        try:
            from lerobot.robots.so_follower import (  # type: ignore
                SO101Follower,
                SO101FollowerConfig,
            )
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "The real backend needs the lerobot SDK. Install with "
                "`pip install \"lerobot[feetech]\"`, or use backend='mock'."
            ) from exc

        cfg = SO101FollowerConfig(
            port=self._config.port,
            id=self._config.robot_id,
            max_relative_target=None,                # smooth interpolation is the safety
            disable_torque_on_disconnect=not self._config.hold_torque,
        )
        self._robot = SO101Follower(cfg)
        # Connect WITHOUT lerobot's interactive calibration. With calibrate=False
        # it never calls calibrate(), which would block on input() (and offer to
        # run the destructive hand-guided range-of-motion routine). This is the
        # safe, non-interactive path for scripts and the web server alike.
        self._robot.connect(calibrate=False)
        # lerobot only loads the calibration file into the motors as part of
        # calibrate(); with calibrate=False we must do it ourselves when the
        # servos' stored calibration no longer matches the file (e.g. after a
        # power-cycle). This is exactly what pressing ENTER at lerobot's prompt
        # does -- load the existing file -- and NEVER the range-of-motion calib.
        if not self._robot.is_calibrated:
            if not getattr(self._robot, "calibration", None):
                raise RuntimeError(
                    f"No calibration found for robot id {self._config.robot_id!r}. "
                    "Refusing to connect (a missing calibration would otherwise "
                    "trigger lerobot's interactive range-of-motion routine)."
                )
            self._robot.bus.write_calibration(self._robot.calibration)
            self._robot.configure()

        self._apply_servo_acceleration()

    def _apply_servo_acceleration(self) -> None:
        """Optionally set the servos' internal acceleration for smooth motion.

        These Feetech servos run their own velocity PID + acceleration profile, so
        the real smoothness lever is the motor's ``Acceleration`` register: with a
        gentle value each streamed setpoint is RAMPED by the servo instead of
        snapped at max acceleration (the stop-start jerk). This is far safer and
        smoother than a software feedback loop over the serial bus.

        Opt-in and tunable via ``$LIMBIC_SERVO_ACCEL`` (an integer; lower = gentler
        ramp = smoother, higher = snappier; unset leaves whatever lerobot
        configured). Any failure is warned about but never breaks the connection.
        """
        accel = os.environ.get("LIMBIC_SERVO_ACCEL")
        if not accel:
            return
        try:
            value = int(accel)
            for motor in self._robot.bus.motors:
                self._robot.bus.write("Acceleration", motor, value, normalize=False)
        except Exception as exc:  # firmware/register differences must not break connect
            print(f"[limbic] could not set servo Acceleration={accel!r}: {exc}")

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()

    def read_joints(self) -> dict[str, float]:
        obs = self._robot.get_observation()
        return {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}

    def send_joints(self, command: dict[str, float]) -> None:
        self._robot.send_action({f"{name}.pos": float(v) for name, v in command.items()})


def make_backend(config: ArmConfig, verbose: bool = True) -> HardwareBackend:
    """Pick a backend from ``config.backend`` ("auto" | "real" | "mock")."""
    choice = config.backend

    if choice == "mock":
        return MockBackend(verbose=verbose)

    if choice == "real":
        return RealBackend(config)

    # "auto": prefer real hardware when it's actually available, else simulate.
    if config.port is None:
        if verbose:
            print("[limbic] no serial port found -> using mock backend.")
        return MockBackend(verbose=verbose)
    try:
        import lerobot  # type: ignore  # noqa: F401
    except ImportError:
        if verbose:
            print(
                "[limbic] serial port found but lerobot not installed -> mock "
                "backend. `pip install \"lerobot[feetech]\"` to drive the real arm."
            )
        return MockBackend(verbose=verbose)
    return RealBackend(config)
