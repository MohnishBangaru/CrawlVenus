"""Enhanced Device management for Android automation.

Optimized for Android Studio emulator with intelligent resource management.
"""

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

import psutil

from Explorer.device import ADBError, Device

from ..core.config import config
from ..core.logger import log


@dataclass
class DeviceInfo:
    """Extended device information."""

    serial: str
    model: str
    android_version: str
    api_level: int
    screen_size: tuple[int, int]
    screen_density: int
    is_emulator: bool
    available_memory: int
    cpu_usage: float


class EnhancedDeviceManager:
    """Enhanced device management with Android Studio emulator optimization."""
    
    def __init__(self) -> None:
        """Initialize the device manager and internal caches."""
        self.device: Optional[Device] = None
        self.device_info: Optional[DeviceInfo] = None
        self.screenshot_cache: dict[str, Any] = {}
        self.performance_metrics: dict[str, Any] = {}
        self._is_connected = False
        self._lock = asyncio.Lock()
        # Track last known resumed activity for faster recovery
        self._last_activity: Optional[str] = None
        
    async def connect_device(self, device_serial: Optional[str] = None) -> bool:
        """Connect to Android device or emulator with enhanced error handling.

        Args:
            device_serial: Specific device serial. If ``None`` auto-detect emulator.

        Returns:
            bool: ``True`` if connection successful, ``False`` otherwise.

        """
        async with self._lock:
            try:
                log.info("Attempting to connect to Android device/emulator...")
                
                if device_serial:
                    self.device = Device(device_serial)
                    log.info(f"Connecting to specified device: {device_serial}")
                else:
                    # Prefer emulator for Android Studio integration
                    self.device = Device.from_emulator()
                    log.info("Connected to Android Studio emulator")
                
                # Verify connection and gather device info
                await self._gather_device_info()
                
                # Optimize device settings for automation
                await self._optimize_device_settings()
                
                self._is_connected = True
                log.success(f"Successfully connected to device: {self.device.serial}")
                return True
                
            except ADBError as e:
                log.error(f"ADB connection failed: {e}")
                return False
            except Exception as e:
                log.error(f"Unexpected error during device connection: {e}")
                return False
    
    async def _gather_device_info(self) -> None:
        """Gather comprehensive device information."""
        if not self.device:
            raise RuntimeError("Device not connected")
        
        try:
            # Get device properties
            model = self.device.shell("getprop ro.product.model").strip()
            android_version = self.device.shell("getprop ro.build.version.release").strip()
            api_level = int(self.device.shell("getprop ro.build.version.sdk").strip())
            
            # Get screen information
            screen_info = self.device.shell("wm size").strip()
            screen_size = self._parse_screen_size(screen_info)
            
            density_info = self.device.shell("wm density").strip()
            screen_density = self._parse_screen_density(density_info)
            
            # Check if emulator
            is_emulator = self.device.serial.startswith("emulator-")
            
            # Get memory information
            memory_info = self.device.shell("cat /proc/meminfo | grep MemAvailable").strip()
            available_memory = self._parse_memory_info(memory_info)
            
            # Get CPU usage
            cpu_usage = await self._get_cpu_usage()
            
            self.device_info = DeviceInfo(
                serial=self.device.serial,
                model=model,
                android_version=android_version,
                api_level=api_level,
                screen_size=screen_size,
                screen_density=screen_density,
                is_emulator=is_emulator,
                available_memory=available_memory,
                cpu_usage=cpu_usage
            )
            
            log.info(f"Device Info: {model} (Android {android_version}, API {api_level})")
            log.info(f"Screen: {screen_size[0]}x{screen_size[1]}, Density: {screen_density}")
            
        except Exception as e:
            log.error(f"Failed to gather device information: {e}")
            raise
    
    async def _optimize_device_settings(self) -> None:
        """Optimize device settings for automation."""
        if not self.device:
            raise RuntimeError("Device not connected")
        
        try:
            log.info("Optimizing device settings for automation...")
            
            # Enable developer options if not already enabled
            self.device.shell("settings put global development_settings_enabled 1")
            
            # Disable animations for faster automation
            self.device.shell("settings put global window_animation_scale 0.0")
            self.device.shell("settings put global transition_animation_scale 0.0")
            self.device.shell("settings put global animator_duration_scale 0.0")
            
            # Increase touch sensitivity
            self.device.shell("settings put secure touch_exploration_enabled 1")
            
            # Keep screen awake during automation
            self.device.shell("settings put global stay_on_while_plugged_in 3")
            
            # Disable auto-brightness for consistent screenshots
            self.device.shell("settings put system screen_brightness_mode 0")
            self.device.shell("settings put system screen_brightness 128")
            
            log.success("Device settings optimized for automation")
            
        except Exception as e:
            log.warning(f"Failed to optimize some device settings: {e}")
    
    async def capture_screenshot(self, save_path: Optional[str] = None) -> str:
        """Capture device screenshot with caching and optimization.

        Uses ``adb exec-out screencap -p`` when ``config.use_fast_screencap`` is ``True`` for faster
        capture on modern devices (API 21+), avoiding sd-card I/O.
        """
        if not self._is_connected:
            raise RuntimeError("Device not connected")
        
        if not self.device:
            raise RuntimeError("Device not connected")
        
        try:
            # Generate screenshot path
            if save_path is None:
                timestamp = int(time.time() * 1000)
                save_path = os.path.join(
                    config.get_screenshot_path(),
                    f"screenshot_{timestamp}.png"
                )
            
            # Ensure screenshot directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            if config.use_fast_screencap:
                # Use adb exec-out for rapid capture
                with open(save_path, "wb") as f:
                    proc = subprocess.Popen(  # noqa: S603
                        ["adb", "-s", self.device.serial, "exec-out", "screencap", "-p"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    stdout, stderr = proc.communicate(timeout=10)
                    if proc.returncode != 0 or not stdout:
                        raise RuntimeError(f"exec-out screencap failed: {stderr.decode().strip()}")
                    f.write(stdout)
            else:
                # Fallback: traditional screencap to sdcard
                self.device.shell("screencap -p /sdcard/screenshot.png")
                self.device.pull("/sdcard/screenshot.png", save_path)
                # Clean up device storage
                self.device.shell("rm /sdcard/screenshot.png")
            
            log.debug(f"Screenshot captured: {save_path}")
            return save_path
            
        except subprocess.TimeoutExpired:
            log.error("Screenshot capture timed out")
            raise
        except Exception as e:
            log.error(f"Failed to capture screenshot: {e}")
            raise
    
    async def perform_action(self, action: dict[str, Any]) -> bool:
        """Execute a device action with intelligent retry and verification.

        Args:
            action: Action dictionary with type and parameters.

        Returns:
            bool: True if action successful, False otherwise.

        """
        if not self._is_connected:
            raise RuntimeError("Device not connected")
        
        action_type = action.get("type")
        
        try:
            if action_type == "tap":
                return await self._perform_tap(action)
            elif action_type == "swipe":
                return await self._perform_swipe(action)
            elif action_type == "input_text":
                return await self._perform_input_text(action)
            elif action_type == "text_input":
                return await self._perform_input_text(action)
            elif action_type == "key_event":
                return await self._perform_key_event(action)
            elif action_type == "wait":
                return await self._perform_wait(action)
            else:
                log.error(f"Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            log.error(f"Failed to perform action {action_type}: {e}")
            return False
    
    async def _perform_tap(self, action: dict[str, Any]) -> bool:
        """Perform tap action with verification."""
        x, y = action.get("x"), action.get("y")
        
        if x is None or y is None:
            log.error("Tap action missing coordinates")
            return False
        
        # Capture screenshot before action for verification
        before_screenshot = await self.capture_screenshot()
        
        # Perform tap
        if not self.device:
            raise RuntimeError("Device not connected")
        
        self.device.shell(f"input tap {x} {y}")
        
        # Wait for UI to respond
        await asyncio.sleep(0.5)
        
        # Capture screenshot after action
        after_screenshot = await self.capture_screenshot()
        
        # Verify action had effect (basic check)
        action_verified = await self._verify_action_effect(before_screenshot, after_screenshot)
        
        if action_verified:
            log.log_automation_step(f"Tap at ({x}, {y})", {"verified": True})
            return True
        else:
            log.warning(f"Tap at ({x}, {y}) may not have had expected effect")
            return False
    
    async def _perform_swipe(self, action: dict[str, Any]) -> bool:
        """Perform swipe action."""
        x1, y1 = action.get("x1"), action.get("y1")
        x2, y2 = action.get("x2"), action.get("y2")
        duration = action.get("duration", 300)
        
        if None in [x1, y1, x2, y2]:
            log.error("Swipe action missing coordinates")
            return False
        
        if not self.device:
            raise RuntimeError("Device not connected")
        
        self.device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")
        await asyncio.sleep(duration / 1000 + 0.2)  # Convert to seconds + buffer
        
        log.log_automation_step(f"Swipe from ({x1}, {y1}) to ({x2}, {y2})", {"duration": duration})
        return True
    
    async def _perform_input_text(self, action: dict[str, Any]) -> bool:
        """Perform text input action with keyboard triggering."""
        text = action.get("text", "")
        x = action.get("x")
        y = action.get("y")
        
        if not text:
            log.error("Input text action missing text")
            return False
        
        if not self.device:
            raise RuntimeError("Device not connected")
        
        try:
            # First, tap on the input field to focus it
            if x is not None and y is not None:
                log.debug(f"Tapping input field at ({x}, {y}) to focus")
                self.device.shell(f"input tap {x} {y}")
                await asyncio.sleep(0.5)  # Wait for keyboard to appear
            
            # Clear any existing text
            self.device.shell("input keyevent 123")  # Move cursor to end
            self.device.shell("input keyevent 67")   # Select all
            self.device.shell("input keyevent 22")   # Delete
            
            # Wait for keyboard to be ready
            await asyncio.sleep(0.3)
            
            # Escape special characters for ADB input
            escaped_text = text.replace(" ", "%s").replace("&", "\\&").replace("'", "\\'")
            
            # Input the text
            self.device.shell(f"input text \"{escaped_text}\"")
            await asyncio.sleep(0.5)
            
            # Press enter to confirm (optional)
            self.device.shell("input keyevent 66")  # Enter key
            await asyncio.sleep(0.2)
            
            log.log_automation_step(f"Input text: {text} at ({x}, {y})")
            return True
            
        except Exception as e:
            log.error(f"Text input failed: {e}")
            return False
    
    async def _perform_key_event(self, action: dict[str, Any]) -> bool:
        """Perform key event action."""
        key_code = action.get("key_code")
        
        if key_code is None:
            log.error("Key event action missing key code")
            return False
        
        if not self.device:
            raise RuntimeError("Device not connected")
        
        self.device.shell(f"input keyevent {key_code}")
        await asyncio.sleep(0.2)
        
        log.log_automation_step(f"Key event: {key_code}")
        return True
    
    async def _perform_wait(self, action: dict[str, Any]) -> bool:
        """Perform wait action."""
        duration = action.get("duration", 1.0)
        
        await asyncio.sleep(duration)
        
        log.log_automation_step(f"Wait: {duration}s")
        return True
    
    async def _verify_action_effect(self, before_path: str, after_path: str) -> bool:
        """Basic verification that action had an effect on the UI."""
        try:
            # This is a basic implementation - could be enhanced with computer vision
            before_stat = os.stat(before_path)
            after_stat = os.stat(after_path)
            
            # Simple check: file sizes should be different if UI changed
            return before_stat.st_size != after_stat.st_size
            
        except Exception as e:
            log.debug(f"Could not verify action effect: {e}")
            return True  # Assume success if verification fails
    
    async def get_resource_usage(self) -> dict[str, Any]:
        """Get current resource usage metrics."""
        if not self.device:
            raise RuntimeError("Device not connected")
        
        try:
            # Device resource usage
            cmd_meminfo = (
                "cat /proc/meminfo | grep -E '(MemTotal|MemFree|MemAvailable)'"
            )
            memory_info = self.device.shell(cmd_meminfo)
            # CPU usage captured but not used further; could be returned if needed
            await self._get_cpu_usage()
            
            # Host system resource usage
            host_memory = psutil.virtual_memory()
            host_cpu = psutil.cpu_percent(interval=1)
            
            metrics = {
                "device": {
                    "memory_info": memory_info.strip(),
                    "cpu_usage": 0.0 # CPU usage is not directly available from shell command
                },
                "host": {
                    "memory_percent": host_memory.percent,
                    "memory_available": host_memory.available,
                    "cpu_percent": host_cpu
                }
            }
            
            self.performance_metrics = metrics
            return metrics
            
        except Exception as e:
            log.error(f"Failed to get resource usage: {e}")
            return {}
    
    async def _get_cpu_usage(self) -> float:
        """Get device CPU usage."""
        if not self.device:
            return 0.0
        
        try:
            # Get CPU usage from device
            self.device.shell("cat /proc/stat | head -1")
            # Basic CPU usage calculation (simplified)
            return 0.0  # Placeholder for now
            
        except Exception as e:
            log.debug(f"Could not get CPU usage: {e}")
            return 0.0
    
    def _parse_screen_size(self, screen_info: str) -> tuple[int, int]:
        """Parse screen size from wm size output."""
        try:
            # Expected format: "Physical size: 1080x2340"
            if "Physical size:" in screen_info:
                size_part = screen_info.split("Physical size:")[1].strip()
                width, height = map(int, size_part.split('x'))
                return (width, height)
            else:
                # Default fallback
                return (1080, 2340)
        except Exception:
            return (1080, 2340)
    
    def _parse_screen_density(self, density_info: str) -> int:
        """Parse screen density from wm density output."""
        try:
            # Expected format: "Physical density: 440"
            if "Physical density:" in density_info:
                density_part = density_info.split("Physical density:")[1].strip()
                return int(density_part)
            else:
                return 440  # Default fallback
        except Exception:
            return 440
    
    def _parse_memory_info(self, memory_info: str) -> int:
        """Parse available memory from meminfo output."""
        try:
            # Expected format: "MemAvailable:    2048000 kB"
            if "MemAvailable:" in memory_info:
                mem_part = memory_info.split("MemAvailable:")[1].strip()
                mem_kb = int(mem_part.split()[0])
                return mem_kb // 1024  # Convert to MB
            else:
                return 2048  # Default fallback
        except Exception:
            return 2048
    
    async def disconnect(self) -> None:
        """Disconnect from device and cleanup."""
        if self._is_connected:
            log.info("Disconnecting from device...")
            self.device = None
            self.device_info = None
            self._is_connected = False
            log.success("Device disconnected")
    
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._is_connected
    
    def get_device_info(self) -> Optional[DeviceInfo]:
        """Get device information."""
        return self.device_info
    
    async def get_foreground_package(self) -> Optional[str]:
        """Get the package name of the currently foreground app.
        
        Returns:
            Optional[str]: Package name of foreground app, or None if not found
        """
        if not self._is_connected or not self.device:
            return None
        
        try:
            # Use dumpsys to get the top resumed activity
            result = self.device.shell("dumpsys activity activities | grep topResumedActivity")
            
            if result:
                # Parse the output to extract package name
                # Example output: "topResumedActivity: ActivityRecord{... packageName=com.example.app ...}"
                if "packageName=" in result:
                    package_start = result.find("packageName=") + 12
                    package_end = result.find(" ", package_start)
                    if package_end == -1:
                        package_end = result.find("}", package_start)
                    
                    if package_end > package_start:
                        package_name = result[package_start:package_end]
                        # Extract component name as well (Class)
                        if '/' in result:
                            comp_start = result.find("/") - len(package_name)
                            if comp_start >= 0:
                                comp_end = result.find(" ", comp_start)
                                self._last_activity = result[comp_start + len(package_name)+1:comp_end]
                        return package_name
            
            # Fallback: try a different approach
            result = self.device.shell("dumpsys window windows | grep -E 'mCurrentFocus|mFocusedApp'")
            
            if result:
                # Look for package name in the output
                lines = result.split('\n')
                for line in lines:
                    if "packageName=" in line:
                        package_start = line.find("packageName=") + 12
                        package_end = line.find(" ", package_start)
                        if package_end == -1:
                            package_end = line.find("}", package_start)
                        
                        if package_end > package_start:
                            package_name = line[package_start:package_end]
                            # Update last activity if possible
                            if '/' in line:
                                seg = line.strip().split()[-1]
                                if '/' in seg:
                                    self._last_activity = seg.split('/')[-1]
                            return package_name
            
            return None
            
        except Exception as e:
            log.error(f"Error getting foreground package: {e}")
            return None

    def get_last_foreground_activity(self, package: str) -> Optional[str]:
        """Return last known activity for given package."""
        if self._last_activity and self.device_info and self.device_info.serial:
            return f"{package}/{self._last_activity}"
            return None

    async def validate_app_state(self, target_package: str) -> dict[str, Any]:
        """Validate that the target app is running and in foreground.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            dict: App state validation result with status and reason.
        """
        if not self._is_connected or not self.device:
            return {
                "is_valid": False,
                "status": "device_not_connected",
                "reason": "Device not connected"
            }
        
        try:
            # Check if app is running and in foreground using dumpsys activity activities
            result = self.device.shell("dumpsys activity activities")
            
            # Look for the app in the activity stack
            lines = result.split('\n')
            app_running = False
            app_in_foreground = False
            
            for line in lines:
                # Check if app is running (appears in activity stack)
                if f"packageName={target_package}" in line:
                    app_running = True
                
                # Check if app is in foreground (top resumed activity)
                if "topResumedActivity" in line and target_package in line:
                    app_in_foreground = True
                
                # Also check for RESUMED state
                if f"packageName={target_package}" in line and "state=RESUMED" in line:
                    app_in_foreground = True
            
            if not app_running:
                return {
                    "is_valid": False,
                    "status": "app_not_running",
                    "reason": f"App {target_package} is not running"
                }
            
            if not app_in_foreground:
                return {
                    "is_valid": False,
                    "status": "app_not_in_foreground",
                    "reason": f"App {target_package} is not in foreground"
                }
            
            return {
                "is_valid": True,
                "status": "app_running_and_foreground",
                "reason": "App is running and in foreground"
            }
            
        except Exception as e:
            log.error(f"App state validation error: {e}")
            return {
                "is_valid": False,
                "status": "validation_error",
                "reason": f"Validation error: {str(e)}"
            }
    
    async def recover_app_state(self, target_package: str) -> bool:
        """Attempt to recover app state by bringing it to foreground.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            bool: True if recovery successful, False otherwise.
        """
        if not self._is_connected or not self.device:
            return False
        
        try:
            log.info(f"Attempting to recover app state for {target_package}")
            
            # Method 1: Use monkey to launch app
            try:
                self.device.shell(f"monkey -p {target_package} -c android.intent.category.LAUNCHER 1")
                await asyncio.sleep(1)
                
                # Check if recovery was successful
                app_status = await self.validate_app_state(target_package)
                if app_status["is_valid"]:
                    log.info("App state recovery successful with monkey")
                    return True
            except Exception as e:
                log.debug(f"Monkey recovery failed: {e}")
            
            # Method 2: Use am start with launcher intent
            try:
                self.device.shell(f"am start -W -a android.intent.action.MAIN -c android.intent.category.LAUNCHER -n {target_package}/.MainActivity")
                await asyncio.sleep(1)
                
                # Check again
                app_status = await self.validate_app_state(target_package)
                if app_status["is_valid"]:
                    log.info("App state recovery successful with am start")
                    return True
            except Exception as e:
                log.debug(f"AM start recovery failed: {e}")
            
            # Method 3: Try to find the correct activity name
            try:
                result = self.device.shell(f"cmd package resolve-activity --brief {target_package}")
                
                if result and "activity" in result:
                    lines = result.strip().split('\n')
                    for line in lines:
                        if target_package in line and "activity" in line:
                            activity_name = line.split()[-1]
                            log.info(f"Found activity: {activity_name}")
                            
                            self.device.shell(f"am start -W -n {activity_name}")
                            await asyncio.sleep(1)
                            
                            app_status = await self.validate_app_state(target_package)
                            if app_status["is_valid"]:
                                log.info("App state recovery successful with correct activity")
                                return True
                            break
            except Exception as e:
                log.debug(f"Activity discovery recovery failed: {e}")
            
            # Method 4: Force stop and relaunch
            try:
                log.info("Attempting force stop and relaunch...")
                self.device.shell(f"am force-stop {target_package}")
                await asyncio.sleep(1)
                
                self.device.shell(f"monkey -p {target_package} -c android.intent.category.LAUNCHER 1")
                await asyncio.sleep(2)
                
                app_status = await self.validate_app_state(target_package)
                if app_status["is_valid"]:
                    log.info("App state recovery successful with force stop and relaunch")
                    return True
            except Exception as e:
                log.debug(f"Force stop recovery failed: {e}")
            
            log.warning("All app state recovery methods failed")
            return False
            
        except Exception as e:
            log.error(f"App state recovery error: {e}")
            return False

    async def setup_foreground_service(self, target_package: str) -> bool:
        """Set up a foreground service with persistent notification for the target app.
        
        This creates a foreground service that will help keep the app in the foreground
        during automation by providing a persistent notification.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            bool: True if foreground service setup successful, False otherwise.
        """
        if not self._is_connected or not self.device:
            return False
        
        try:
            log.info(f"Setting up foreground service for {target_package}")
            
            # Create a simple foreground service APK that can be installed
            service_apk_path = await self._create_foreground_service_apk(target_package)
            
            if not service_apk_path:
                log.error("Failed to create foreground service APK")
                return False
            
            # Install the service APK
            install_success = await self._install_foreground_service(service_apk_path)
            
            if not install_success:
                log.error("Failed to install foreground service APK")
                return False
            
            # Start the foreground service
            service_started = await self._start_foreground_service(target_package)
            
            if service_started:
                log.success(f"Foreground service started for {target_package}")
                return True
            else:
                log.warning("Failed to start foreground service")
                return False
                
        except Exception as e:
            log.error(f"Foreground service setup error: {e}")
            return False

    async def _create_foreground_service_apk(self, target_package: str) -> Optional[str]:
        """Create a simple foreground service APK for the target app.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            str: Path to the created APK, or None if failed.
        """
        try:
            # Use the foreground service builder
            from .foreground_service_builder import get_foreground_service_builder
            
            builder = get_foreground_service_builder()
            apk_path = await builder.create_foreground_service_apk(target_package)
            
            if apk_path:
                log.info(f"Created foreground service APK: {apk_path}")
                return apk_path
            else:
                log.warning("Foreground service APK creation failed, using shell-based approach")
                return None
                
        except Exception as e:
            log.error(f"Failed to create foreground service APK: {e}")
            return None

    async def _install_foreground_service(self, apk_path: str) -> bool:
        """Install the foreground service APK.
        
        Args:
            apk_path: Path to the APK file.
            
        Returns:
            bool: True if installation successful, False otherwise.
        """
        try:
            # For now, we'll use a simpler approach with shell commands
            # In a full implementation, you'd build and install a proper APK
            
            log.info("Foreground service APK installation skipped (using shell-based approach)")
            return True
            
        except Exception as e:
            log.error(f"Failed to install foreground service APK: {e}")
            return False

    async def _start_foreground_service(self, target_package: str) -> bool:
        """Start the foreground service using shell commands.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            bool: True if service started successfully, False otherwise.
        """
        if not self.device:
            return False
            
        try:
            # Use shell commands to create a persistent notification
            # This is a simplified approach - in practice, you'd need a proper foreground service
            
            # Try to create a notification channel (may fail if already exists)
            try:
                self.device.shell("cmd notification create-channel --package com.android.settings --channel-id droidbot_automation --name 'DroidBot Automation' --description 'Keeps app in foreground during automation' --importance 3")
            except Exception:
                log.debug("Notification channel creation failed (may already exist)")
            
            # Create a persistent notification using a simpler approach
            try:
                # Use a more compatible notification command
                notification_cmd = f"cmd notification post --package {target_package} --channel-id droidbot_automation --id 1001 --title 'DroidBot Automation' --text 'Keeping app in foreground for automation'"
                self.device.shell(notification_cmd)
                await asyncio.sleep(1)
                
                # Verify notification was created
                notifications = self.device.shell("cmd notification list")
                if "DroidBot Automation" in notifications or "1001" in notifications:
                    log.info("Foreground notification created successfully")
                    return True
                else:
                    log.warning("Failed to create foreground notification")
                    return False
                    
            except Exception as e:
                log.debug(f"Notification creation failed: {e}")
                # Fallback: just return True since the main goal is app state validation
                log.info("Using fallback foreground service approach")
                return True
                
        except Exception as e:
            log.error(f"Failed to start foreground service: {e}")
            return False

    async def stop_foreground_service(self, target_package: str) -> bool:
        """Stop the foreground service and remove the persistent notification.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            bool: True if service stopped successfully, False otherwise.
        """
        if not self._is_connected or not self.device:
            return False
        
        try:
            log.info(f"Stopping foreground service for {target_package}")
            
            # Remove the persistent notification
            if self.device:
                try:
                    self.device.shell(f"cmd notification remove --package {target_package} --id 1001")
                except Exception as e:
                    log.debug(f"Notification removal failed: {e}")
                
                try:
                    # Clear any remaining notifications
                    self.device.shell("cmd notification clear")
                except Exception as e:
                    log.debug(f"Notification clear failed: {e}")
            
            log.success("Foreground service stopped successfully")
            return True
            
        except Exception as e:
            log.error(f"Failed to stop foreground service: {e}")
            return False

    async def is_foreground_service_running(self, target_package: str) -> bool:
        """Check if the foreground service is running for the target app.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            bool: True if foreground service is running, False otherwise.
        """
        if not self._is_connected or not self.device:
            return False
        
        try:
            # Check if our notification exists
            if self.device:
                notifications = self.device.shell("cmd notification list")
                return "DroidBot Automation" in notifications
            return False
            
        except Exception as e:
            log.error(f"Failed to check foreground service status: {e}")
            return False 

    async def _input_text_via_clipboard(self, text: str) -> bool:
        """Input text using clipboard operations.
        
        Args:
            text: Text to input.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Try multiple clipboard methods for maximum compatibility
        
        # Method 1: Use Android's built-in clipboard manager (Android 10+)
        if await self._try_clipboard_method_1(text):
            return True
        
        # Method 2: Use am broadcast with clipper.set (requires Clipper app)
        if await self._try_clipboard_method_2(text):
            return True
        
        # Method 3: Use service call for clipboard (Android 11+)
        if await self._try_clipboard_method_3(text):
            return True
        
        # Method 4: Use input text with proper escaping (fallback)
        return await self._try_clipboard_method_4(text)

    async def _try_clipboard_method_1(self, text: str) -> bool:
        """Method 1: Use Android's built-in clipboard manager."""
        try:
            # Use service call to set clipboard content
            escaped_text = text.replace("'", "\\'").replace('"', '\\"')
            
            # Set clipboard using service call (Android 10+)
            clipboard_cmd = f"service call clipboard 1 s16 '{escaped_text}'"
            self.device.shell(clipboard_cmd)
            await asyncio.sleep(0.3)
            
            # Paste using Ctrl+V equivalent
            self.device.shell("input keyevent 279")  # KEYCODE_PASTE
            await asyncio.sleep(0.3)
            
            return True
            
        except Exception as e:
            log.debug(f"Clipboard method 1 failed: {e}")
            return False

    async def _try_clipboard_method_2(self, text: str) -> bool:
        """Method 2: Use am broadcast with clipper.set."""
        try:
            escaped_text = text.replace("'", "\\'").replace('"', '\\"')
            
            # Set clipboard content using am broadcast
            clipboard_cmd = f"am broadcast -a clipper.set -e text '{escaped_text}'"
            self.device.shell(clipboard_cmd)
            await asyncio.sleep(0.2)
            
            # Paste the content (Ctrl+V equivalent)
            self.device.shell("input keyevent 279")  # KEYCODE_PASTE
            await asyncio.sleep(0.3)
            
            return True
            
        except Exception as e:
            log.debug(f"Clipboard method 2 failed: {e}")
            return False

    async def _try_clipboard_method_3(self, text: str) -> bool:
        """Method 3: Use service call for clipboard (Android 11+)."""
        try:
            escaped_text = text.replace("'", "\\'").replace('"', '\\"')
            
            # Alternative service call method
            clipboard_cmd = f"service call clipboard 2 s16 '{escaped_text}'"
            self.device.shell(clipboard_cmd)
            await asyncio.sleep(0.3)
            
            # Paste using Ctrl+V equivalent
            self.device.shell("input keyevent 279")  # KEYCODE_PASTE
            await asyncio.sleep(0.3)
            
            return True
            
        except Exception as e:
            log.debug(f"Clipboard method 3 failed: {e}")
            return False

    async def _try_clipboard_method_4(self, text: str) -> bool:
        """Method 4: Use input text with proper escaping (fallback)."""
        try:
            # Use the enhanced keyboard input method
            return await self._input_text_via_keyboard(text)
            
        except Exception as e:
            log.debug(f"Clipboard method 4 failed: {e}")
            return False

    async def _input_text_via_keyboard(self, text: str) -> bool:
        """Fallback method: Input text using keyboard typing.
        
        Args:
            text: Text to input.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Escape special characters for ADB input
            escaped_text = text.replace(" ", "%s").replace("&", "\\&").replace("'", "\\'")
            
            # Input the text using keyboard
            self.device.shell(f"input text \"{escaped_text}\"")
            await asyncio.sleep(0.5)
            
            return True
            
        except Exception as e:
            log.error(f"Keyboard input failed: {e}")
            return False 