"""Cross-platform helpers shared by the whole stack (macOS + Windows + Linux).

The original LeRobot/bronny code was Windows-only: it hard-coded ``COM7`` serial
ports, used ``pygrabber.dshow_graph`` (Windows DirectShow) to find cameras, and
opened them with the ``cv2.CAP_DSHOW`` backend (also Windows-only). None of that
runs on a Mac. This module is the single place where we paper over those
differences so every other module can stay platform-agnostic.

What lives here:
    * ``current_os()``             -> "mac" | "windows" | "linux"
    * ``detect_serial_port()``     -> auto-find the arm's USB serial port on any OS
    * ``list_serial_ports()``      -> every serial port, for diagnostics/UI
    * ``camera_backend()``         -> the right OpenCV capture backend for this OS
    * ``open_camera()``            -> open a webcam by index, OS-appropriate backend

Design rules:
    * Nothing here imports the robot SDK or anything heavy at module load.
    * Optional dependencies (``pyserial``, ``opencv-python``) are imported lazily
      and degrade gracefully, so the package still imports on a bare machine.
    * Everything is overridable by environment variable, so a user never has to
      edit code to point at a different port or camera.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Operating-system detection
# --------------------------------------------------------------------------- #
def current_os() -> str:
    """Return a short, normalized OS name: ``"mac"``, ``"windows"`` or ``"linux"``."""
    system = platform.system().lower()
    if system == "darwin":
        return "mac"
    if system == "windows":
        return "windows"
    return "linux"


IS_MAC = current_os() == "mac"
IS_WINDOWS = current_os() == "windows"
IS_LINUX = current_os() == "linux"


# --------------------------------------------------------------------------- #
# Serial-port discovery (replaces the hard-coded COM7)
# --------------------------------------------------------------------------- #
# USB-serial bridges used by hobby robot arms show up under different names per
# OS. We match the device *description* / hardware id against these hints rather
# than guessing an index, so a port survives reboots and works on every OS.
_USB_SERIAL_HINTS = (
    "usbserial",   # macOS  /dev/cu.usbserial-XXXX  (FTDI, CP210x, CH340)
    "usbmodem",    # macOS  /dev/cu.usbmodem-XXXX   (native-USB MCUs)
    "ch340",       # common Feetech/servo-bus bridge
    "ch341",
    "cp210",       # Silicon Labs CP210x
    "ftdi",        # FTDI
    "ft232",
    "ttyusb",      # Linux  /dev/ttyUSB0
    "ttyacm",      # Linux  /dev/ttyACM0
    "wch",         # WCH (CH340/CH341 vendor)
    "1a86",        # WCH/QinHeng USB vendor id (the SO-101 servo-bus bridge)
)


@dataclass(frozen=True)
class SerialPortInfo:
    """A discovered serial port and why we think it might be the arm."""

    device: str          # e.g. "/dev/cu.usbserial-10" or "COM7"
    description: str      # human-readable name from the OS
    hardware_id: str      # vendor/product id string (may be empty)
    looks_like_arm: bool  # True if it matched one of the USB-serial hints


def list_serial_ports() -> list[SerialPortInfo]:
    """Return every serial port on this machine (for diagnostics or a UI picker).

    Returns an empty list if ``pyserial`` isn't installed, rather than raising —
    the mock backend doesn't need a real port.
    """
    try:
        from serial.tools import list_ports  # type: ignore
    except ImportError:
        return []

    ports: list[SerialPortInfo] = []
    for port in list_ports.comports():
        blob = f"{port.device} {port.description} {port.hwid}".lower()
        ports.append(
            SerialPortInfo(
                device=port.device,
                description=port.description or "",
                hardware_id=port.hwid or "",
                looks_like_arm=any(hint in blob for hint in _USB_SERIAL_HINTS),
            )
        )
    return ports


def detect_serial_port(env_var: str = "LIMBIC_PORT") -> str | None:
    """Best guess at the arm's serial port, working on macOS, Windows and Linux.

    Resolution order:
        1. The ``$LIMBIC_PORT`` environment variable, if set (manual override —
           e.g. ``COM7`` on Windows or ``/dev/cu.usbserial-10`` on macOS).
        2. The single port whose description matches a known USB-serial bridge.
        3. The only port present, if there's exactly one.
        4. ``None`` — caller should fall back to the mock backend and warn.

    This never raises: a missing port is a normal condition during development on
    a machine with no arm attached.
    """
    override = os.environ.get(env_var)
    if override:
        return override

    ports = list_serial_ports()
    likely = [p for p in ports if p.looks_like_arm]
    if len(likely) == 1:
        return likely[0].device
    if len(ports) == 1:
        return ports[0].device
    # 0 ports, or several ambiguous candidates -> let the caller decide.
    return None


# --------------------------------------------------------------------------- #
# Camera backend selection (replaces cv2.CAP_DSHOW + pygrabber)
# --------------------------------------------------------------------------- #
def camera_backend() -> int:
    """Return the OpenCV capture backend best suited to the current OS.

    * macOS   -> AVFoundation (``CAP_AVFOUNDATION``)
    * Windows -> DirectShow   (``CAP_DSHOW``)        [what bronny used]
    * Linux   -> Video4Linux2 (``CAP_V4L2``)

    Falls back to ``CAP_ANY`` (let OpenCV choose) if a backend constant is
    missing or OpenCV isn't installed.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return 0  # cv2.CAP_ANY == 0; harmless placeholder when cv2 is absent.

    if IS_MAC:
        return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    if IS_WINDOWS:
        return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
    return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)


def open_camera(
    index: int | str = 0, width: int | None = None, height: int | None = None
):
    """Open a webcam with the OS-appropriate backend and return the capture.

    Args:
        index: Either a camera index (``0`` = default/built-in on most laptops)
            **or a name substring** (e.g. ``"C920"``, ``"Logitech"``). A name is
            resolved to an index by :func:`resolve_camera`, so the rig camera can
            be addressed by name even when indices shuffle between machines.
        width, height: Optional requested capture resolution. The driver picks the
            closest supported size; read it back with ``cap.get(...)`` if exact
            dimensions matter (as the LeRobot intrinsics code did).

    Returns:
        An opened ``cv2.VideoCapture``.

    Raises:
        ImportError: if ``opencv-python`` is not installed.
        RuntimeError: if the camera could not be opened (or a name didn't match).
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "opencv-python is required to open a camera. Install with "
            "`pip install opencv-python`."
        ) from exc

    if isinstance(index, str) and not index.isdigit():
        resolved = resolve_camera(index)
        if resolved is None:
            names = ", ".join(c.name for c in list_cameras()) or "(none detected)"
            raise RuntimeError(
                f"No camera matched name {index!r}. Detected cameras: {names}."
            )
        index = resolved
    else:
        index = int(index)

    cap = cv2.VideoCapture(index, camera_backend())
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {index} with backend {camera_backend()}. "
            "Check the index (try 0/1/2), permissions (macOS: System Settings → "
            "Privacy → Camera), and that no other app holds the camera."
        )
    return cap


# --------------------------------------------------------------------------- #
# Machine architecture
# --------------------------------------------------------------------------- #
def machine_arch() -> str:
    """Normalized CPU architecture: ``"arm64"`` | ``"x64"`` | ``"x86"`` | other.

    Why this matters here: the *vision* stack (Part B) needs PyTorch, whose
    wheel availability historically split on architecture. Everything else (arm
    control, localization, brain) is arch-agnostic. The profile uses this to say
    what a given machine can actually run — it doesn't hard-block anything.
    """
    m = platform.machine().lower()
    if m in ("arm64", "aarch64"):
        return "arm64"
    if m in ("amd64", "x86_64"):
        return "x64"
    if m in ("x86", "i386", "i686"):
        return "x86"
    return m or "unknown"


# --------------------------------------------------------------------------- #
# Camera discovery (by NAME, so a port/index shuffle doesn't break the rig)
# --------------------------------------------------------------------------- #
# Name fragments that mark a camera as a built-in/laptop sensor rather than the
# external USB webcam used as the overhead rig camera. Used only to *flag*, not
# to hide — every detected camera is still returned.
_BUILTIN_CAMERA_HINTS = (
    "surface camera", "qualcomm", "spectra", "ir camera", "integrated",
    "built-in", "facetime", "front", "rear",
)
# Name fragments that positively mark the external rig webcams on this setup.
_RIG_CAMERA_HINTS = ("c920", "c930", "logitech", "webcam", "usb")


@dataclass(frozen=True)
class CameraInfo:
    """A detected camera and the OpenCV index needed to open it."""

    index: int            # cv2 VideoCapture index (with this OS's backend)
    name: str             # human-readable device name
    looks_like_rig_cam: bool  # heuristic: external USB webcam, not a built-in


def list_cameras() -> list[CameraInfo]:
    """Enumerate cameras BY NAME, mapped to the index OpenCV needs to open them.

    Resolution strategy (best-effort, degrades gracefully):
        * Windows: ``pygrabber`` DirectShow enumeration — its device order is the
          same order ``cv2.VideoCapture(i, CAP_DSHOW)`` uses, giving a reliable
          name->index map. (Plain index probing is unreliable on Windows: dead
          indices falsely report "opened".)
        * Otherwise (or if pygrabber is absent): probe indices 0..7 and keep the
          ones that actually return a frame, naming them ``camera[i]``.
    """
    if IS_WINDOWS:
        named = _windows_cameras_by_name()
        if named:
            return named
    return _probe_camera_indices()


def _windows_cameras_by_name() -> list[CameraInfo]:
    """Windows name->index map via pygrabber's DirectShow enumeration."""
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
    except ImportError:
        return []
    try:
        names = FilterGraph().get_input_devices()
    except Exception:  # pragma: no cover - COM/driver quirks
        return []
    cams: list[CameraInfo] = []
    for i, name in enumerate(names):
        cams.append(CameraInfo(index=i, name=name, looks_like_rig_cam=_is_rig_cam(name)))
    return cams


def _probe_camera_indices(max_probe: int = 8) -> list[CameraInfo]:
    """Cross-platform fallback: keep indices that actually deliver a frame."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return []
    backend = camera_backend()
    cams: list[CameraInfo] = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, backend)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            cams.append(CameraInfo(index=i, name=f"camera[{i}]", looks_like_rig_cam=True))
    return cams


def _is_rig_cam(name: str) -> bool:
    low = name.lower()
    if any(h in low for h in _BUILTIN_CAMERA_HINTS):
        return False
    return any(h in low for h in _RIG_CAMERA_HINTS)


def resolve_camera(spec: str | int, env_var: str = "LIMBIC_CAMERA") -> int | None:
    """Resolve a camera *index or name substring* to an OpenCV index.

    Order: an explicit ``$LIMBIC_CAMERA`` override (index or name) wins; then the
    passed ``spec`` (an int/digit -> that index; a name -> first case-insensitive
    name match). Returns ``None`` if nothing matches.
    """
    override = os.environ.get(env_var)
    if override:
        spec = override
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return int(spec)
    hint = str(spec).lower()
    for cam in list_cameras():
        if hint in cam.name.lower():
            return cam.index
    return None


# --------------------------------------------------------------------------- #
# Capability probing + the one-call machine profile
# --------------------------------------------------------------------------- #
def _module_available(name: str) -> bool:
    """True if ``name`` can be imported, without paying the cost of importing it."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _opencv_variant() -> str:
    """Which OpenCV build is installed: ``gui`` | ``headless`` | ``conflict`` | ``none``.

    The ``-headless`` build has no ``imshow``/window support, which the Stage 3/4
    click UI needs — worth surfacing rather than discovering at a black screen.
    """
    try:
        from importlib.metadata import distributions
    except ImportError:  # pragma: no cover
        return "unknown"
    names: set[str] = set()
    for dist in distributions():
        try:
            names.add(dist.metadata["Name"].lower())
        except Exception:  # pragma: no cover - malformed metadata
            pass
    gui = bool({"opencv-python", "opencv-contrib-python"} & names)
    headless = bool({"opencv-python-headless", "opencv-contrib-python-headless"} & names)
    if gui and headless:
        return "conflict"
    if headless:
        return "headless"
    if gui:
        return "gui"
    return "none"


@dataclass(frozen=True)
class MachineProfile:
    """A one-call snapshot of what THIS machine is and what it can run.

    Built so that moving the rig to a new computer is "plug in -> profile -> go":
    it auto-finds the arm's serial port, lists the cameras by name, and reports
    which capabilities (real arm, vision, IK) are installed here.
    """

    os: str                              # "mac" | "windows" | "linux"
    arch: str                            # "arm64" | "x64" | ...
    python_version: str
    arm_port: str | None                 # auto-detected serial port (or None)
    arm_port_identified: bool            # matched a known USB-serial bridge
    serial_ports: tuple[SerialPortInfo, ...]
    cameras: tuple[CameraInfo, ...]
    has_lerobot: bool                    # real SO-101 backend available
    has_torch: bool                      # vision (Part B) can run here
    has_ikpy: bool                       # URDF-based kinematics available
    opencv_variant: str                  # gui | headless | conflict | none

    @property
    def can_drive_arm(self) -> bool:
        """True when a real arm could be driven from this machine right now."""
        return self.has_lerobot and self.arm_port is not None

    @property
    def can_run_vision(self) -> bool:
        """True when the PyTorch-based vision stack can run here."""
        return self.has_torch

    @property
    def rig_cameras(self) -> tuple[CameraInfo, ...]:
        """Just the cameras that look like external rig webcams."""
        return tuple(c for c in self.cameras if c.looks_like_rig_cam)


def machine_profile() -> MachineProfile:
    """Probe this machine end-to-end and return a :class:`MachineProfile`."""
    ports = list_serial_ports()
    arm_port = detect_serial_port()
    identified = any(p.device == arm_port and p.looks_like_arm for p in ports)
    return MachineProfile(
        os=current_os(),
        arch=machine_arch(),
        python_version=platform.python_version(),
        arm_port=arm_port,
        arm_port_identified=identified,
        serial_ports=tuple(ports),
        cameras=tuple(list_cameras()),
        has_lerobot=_module_available("lerobot"),
        has_torch=_module_available("torch"),
        has_ikpy=_module_available("ikpy"),
        opencv_variant=_opencv_variant(),
    )


def format_profile(profile: MachineProfile | None = None) -> str:
    """Render a machine profile as a human-readable report with next-steps."""
    p = profile or machine_profile()
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  limbic machine profile")
    lines.append("=" * 60)
    lines.append(f"  OS / arch    : {p.os} / {p.arch}")
    lines.append(f"  Python       : {p.python_version}")

    # Arm
    if p.arm_port:
        tag = "identified as the arm bridge" if p.arm_port_identified else "only/likely port"
        lines.append(f"  Arm port     : {p.arm_port}  ({tag})")
    else:
        lines.append("  Arm port     : NONE found (plug in the arm / set $LIMBIC_PORT)")
    if len(p.serial_ports) > 1:
        for sp in p.serial_ports:
            mark = "*" if sp.device == p.arm_port else " "
            lines.append(f"      {mark} {sp.device}  {sp.description}")

    # Cameras
    if p.cameras:
        lines.append("  Cameras      :")
        for c in p.cameras:
            tag = "RIG" if c.looks_like_rig_cam else "built-in"
            lines.append(f"      [{c.index}] {c.name}  ({tag})")
    else:
        lines.append("  Cameras      : none detected")

    # Capabilities
    lines.append("  Capabilities :")
    lines.append(f"      real arm (lerobot) : {'yes' if p.has_lerobot else 'no'}")
    lines.append(f"      vision   (torch)   : {'yes' if p.has_torch else 'no'}")
    lines.append(f"      kinematics (ikpy)  : {'yes' if p.has_ikpy else 'no'}")
    lines.append(f"      opencv build       : {p.opencv_variant}")

    # Bottom-line recommendations
    lines.append("-" * 60)
    lines.append(f"  -> drive the real arm here : {'YES' if p.can_drive_arm else 'no'}")
    lines.append(f"  -> run vision (Part B) here: {'YES' if p.can_run_vision else 'no'}")
    if p.opencv_variant == "headless":
        lines.append("  ! opencv is headless: imshow/click UI won't show windows.")
        lines.append("    fix: pip install --force-reinstall opencv-python")
    elif p.opencv_variant == "conflict":
        lines.append("  ! both opencv-python and -headless are installed (import order")
        lines.append("    decides which wins). fix: uninstall one.")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_profile())
