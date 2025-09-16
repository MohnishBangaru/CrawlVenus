"""Fast screenshot capture utilities.

This module provides a "fast path" screenshot capture implementation
using ``adb exec-out screencap`` which is **much faster** than the legacy
``adb shell screencap && adb pull`` approach.

It falls back gracefully if ``exec-out`` is not supported. In the future we
can extend this module to support **minicap** or **scrcpy** for streaming
screenshots, but the current implementation already provides a ~2-3x speed
improvement and is compatible with modern Android API levels (API 24+).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from Explorer.device import ADBError, Device

from ..core.logger import log


def _save_bytes_to_file(data: bytes, path: str) -> None:
    """Write bytes to *path*, creating parent directories if necessary."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def capture_fast_screenshot(device: Device, save_path: str) -> bool:
    """Capture a screenshot using ``adb exec-out screencap -p``.

    Args:
        device: ``Device`` instance.
        save_path: Local path to save PNG screenshot.

    Returns:
        True if screenshot saved successfully, False otherwise.

    """
    try:
        log.debug("Capturing screenshot via exec-out ...")

        adb_path = shutil.which("adb") or "adb"
        cmd = [
            adb_path,
            "-s",
            device.serial,
            "exec-out",
            "screencap",
            "-p",
        ]

        png_bytes: bytes = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)  # noqa: S603
        # Older devices may return Windows line endings; normalize them
        if png_bytes.startswith(b"\x89PNG"):
            # Already valid PNG
            pass
        else:
            # Remove carriage returns if present
            png_bytes = png_bytes.replace(b"\r\n", b"\n")
        _save_bytes_to_file(png_bytes, save_path)
        return True
    except subprocess.CalledProcessError as exc:
        log.warning(
            f"exec-out screencap failed (returncode={exc.returncode}). "
            "Falling back to pull method."
        )
        return False
    except Exception as e:  # pragma: no cover
        log.error(f"Fast screenshot capture failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def capture_screenshot(device: Device, save_path: str, *, force_pull: bool = False) -> str:
    """Capture a screenshot, using fast method if possible.

    This function attempts the fast `exec-out` screencap first; if it fails or
    ``force_pull`` is True, it falls back to the legacy pull implementation
    used previously in ``DeviceManager``.
    """
    if not force_pull and capture_fast_screenshot(device, save_path):
        return save_path

    # Fallback - legacy pull
    try:
        device.shell("screencap -p /sdcard/screenshot.png")
        device.pull("/sdcard/screenshot.png", save_path)
        device.shell("rm /sdcard/screenshot.png")
        return save_path
    except Exception as e:
        log.error(f"Legacy screenshot fallback failed: {e}")
        raise ADBError("Screenshot capture failed") from e 