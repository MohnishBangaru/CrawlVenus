from __future__ import annotations

import subprocess


class ADBError(RuntimeError):
    """Custom exception raised when an ADB-related error occurs."""


class Device:
    """Lightweight wrapper around `adb` for interacting with a single Android device.

    The design purposefully avoids heavy external dependencies by using Python's
    built-in ``subprocess`` module, keeping the framework lean and easily
    installable. Only a running ``adb`` binary (bundled with the Android SDK)
    is required.
    """

    def __init__(self, serial: str):
        self.serial = serial

    # ---------------------------------------------------------------------
    # Factory helpers
    # ---------------------------------------------------------------------
    @classmethod
    def list_devices(cls) -> list[str]:
        """Return a list of connected device/emulator serial numbers."""
        output = subprocess.check_output(["adb", "devices"], encoding="utf-8")
        lines = output.strip().splitlines()[1:]  # Skip the header
        serials: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1] == "device":
                serials.append(parts[0])
        return serials

    @classmethod
    def from_emulator(cls) -> Device:
        """Return a ``Device`` instance pointing at the first running emulator.

        Raises:
            ADBError: If no emulator device is detected.

        """
        serials = cls.list_devices()
        emulators = [s for s in serials if s.startswith("emulator-")]
        if not emulators:
            raise ADBError("No Android emulator detected. Start an emulator in Android Studio and ensure 'adb devices' lists it.")
        # Prefer the first emulator found for now. Future improvements may
        # allow selector logic.
        return cls(emulators[0])

    # ---------------------------------------------------------------------
    # Basic operations
    # ---------------------------------------------------------------------
    def shell(self, command: str, *, timeout: int | None = None) -> str:
        """Execute an ADB shell command and return stdout as a string."""
        cmd = ["adb", "-s", self.serial, "shell", command]
        return subprocess.check_output(cmd, encoding="utf-8", timeout=timeout)

    def install_apk(self, apk_path: str) -> None:
        """Install an APK on the device, replacing it if present."""
        subprocess.check_call(["adb", "-s", self.serial, "install", "-r", apk_path])

    def push(self, local: str, remote: str) -> None:
        """Push a file or directory to the device."""
        subprocess.check_call(["adb", "-s", self.serial, "push", local, remote])

    def pull(self, remote: str, local: str) -> None:
        """Pull a file or directory from the device."""
        subprocess.check_call(["adb", "-s", self.serial, "pull", remote, local])

    # ---------------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover – string representation only
        return f"<Device serial={self.serial!r}>" 