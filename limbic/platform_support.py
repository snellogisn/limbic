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


def open_camera(index: int = 0, width: int | None = None, height: int | None = None):
    """Open webcam ``index`` with the OS-appropriate backend and return the capture.

    Args:
        index: Camera index (0 = default/built-in on most laptops).
        width, height: Optional requested capture resolution. The driver picks the
            closest supported size; read it back with ``cap.get(...)`` if exact
            dimensions matter (as the LeRobot intrinsics code did).

    Returns:
        An opened ``cv2.VideoCapture``.

    Raises:
        ImportError: if ``opencv-python`` is not installed.
        RuntimeError: if the camera could not be opened.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "opencv-python is required to open a camera. Install with "
            "`pip install opencv-python`."
        ) from exc

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
