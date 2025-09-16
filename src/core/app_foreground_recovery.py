"""App Foreground Recovery System for DroidBot-GPT framework.

This module provides intelligent detection and recovery logic for when the home or
switch button is pressed while the app is active. It ensures that the first action
after such events is to recover the app back to the foreground, not press back button.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .logger import log
from .device_manager import EnhancedDeviceManager


class NavigationEvent(Enum):
    """Types of navigation events that can cause app to leave foreground."""
    HOME_BUTTON = "home_button"  # Key code 3
    APP_SWITCH = "app_switch"    # Key code 187
    BACK_BUTTON = "back_button"  # Key code 4
    UNKNOWN = "unknown"


@dataclass
class NavigationEventRecord:
    """Record of a navigation event with context."""
    event_type: NavigationEvent
    timestamp: float
    target_package: str
    app_was_foreground: bool
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def __post_init__(self):
        if self.target_package is None:
            raise ValueError("target_package cannot be None")


class AppForegroundRecoveryManager:
    """Manages app foreground recovery after navigation events."""
    
    def __init__(self, device_manager: EnhancedDeviceManager):
        """Initialize the app foreground recovery manager.
        
        Args:
            device_manager: The device manager instance
        """
        self.device_manager = device_manager
        self.target_package: Optional[str] = None
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.navigation_events: List[NavigationEventRecord] = []
        self.last_app_state = False  # True if app was in foreground
        self.recovery_callback: Optional[Callable] = None
        
        # Key codes for navigation events
        self.navigation_key_codes = {
            3: NavigationEvent.HOME_BUTTON,    # HOME
            187: NavigationEvent.APP_SWITCH,   # APP_SWITCH
            4: NavigationEvent.BACK_BUTTON     # BACK
        }
        
        # Single recovery strategy - direct app launch
        self.recovery_strategy = self._recover_with_app_launch
    
    async def start_monitoring(self, target_package: str, recovery_callback: Optional[Callable] = None) -> bool:
        """Start monitoring for navigation events that could cause app to leave foreground.
        
        Args:
            target_package: Package name of the target app
            recovery_callback: Optional callback function to call when recovery is needed
            
        Returns:
            bool: True if monitoring started successfully
        """
        if self.is_monitoring:
            log.warning("App foreground recovery monitoring already active")
            return True
        
        self.target_package = target_package
        self.recovery_callback = recovery_callback
        self.is_monitoring = True
        self.navigation_events.clear()
        
        log.info(f"Starting app foreground recovery monitoring for {target_package}")
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop()
        )
        
        return True
    
    async def stop_monitoring(self) -> bool:
        """Stop the navigation event monitoring.
        
        Returns:
            bool: True if monitoring stopped successfully
        """
        if not self.is_monitoring:
            return True
        
        log.info("Stopping app foreground recovery monitoring")
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.target_package = None
        self.recovery_callback = None
        return True
    
    async def _monitoring_loop(self):
        """Main monitoring loop that continuously monitors foreground activity and triggers recovery immediately."""
        while self.is_monitoring and self.target_package:
            try:
                # Check if app is currently in foreground
                current_app_foreground = await self._is_app_foreground()
                
                # Detect state changes
                if current_app_foreground != self.last_app_state:
                    if self.last_app_state and not current_app_foreground:
                        # App just left foreground - trigger immediate recovery
                        log.warning(f"App {self.target_package} left foreground - triggering immediate recovery")
                        
                        # Record the event and attempt recovery immediately
                        await self._handle_app_foreground_loss()
                    
                    self.last_app_state = current_app_foreground
                
                # Wait before next check (faster monitoring for quicker response)
                await asyncio.sleep(0.5)  # Check every 500ms for faster response
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in foreground monitoring loop: {e}")
                await asyncio.sleep(0.5)
    
    async def _is_app_foreground(self) -> bool:
        """Check if the target app is currently in foreground.
        
        Returns:
            bool: True if app is in foreground, False otherwise
        """
        try:
            if not self.target_package:
                return False
            
            foreground_package = await self.device_manager.get_foreground_package()
            return foreground_package == self.target_package
            
        except Exception as e:
            log.error(f"Error checking app foreground state: {e}")
            return False
    
    async def _handle_app_foreground_loss(self):
        """Handle the case when the app leaves foreground."""
        log.warning(f"App {self.target_package} left foreground, initiating quick recovery")
        
        # Record the navigation event
        event_record = NavigationEventRecord(
            event_type=NavigationEvent.UNKNOWN,  # We don't know the exact cause
            timestamp=time.time(),
            target_package=self.target_package,
            app_was_foreground=True
        )
        self.navigation_events.append(event_record)
        
        # Use quick recovery (direct command)
        recovery_success = await self._quick_recovery()
        
        event_record.recovery_attempted = True
        event_record.recovery_successful = recovery_success
        
        if recovery_success:
            log.success(f"Quick recovery successful for {self.target_package}")
        else:
            log.error(f"Quick recovery failed for {self.target_package}")
        
        # Call recovery callback if provided
        if self.recovery_callback:
            try:
                await self.recovery_callback(recovery_success)
            except Exception as e:
                log.error(f"Error in recovery callback: {e}")
    
    async def _quick_recovery(self) -> bool:
        """Quick recovery using direct app launch command.
        
        Returns:
            bool: True if recovery successful
        """
        try:
            if not self.device_manager.device or not self.target_package:
                return False
            
            log.info(f"Quick recovery: launching {self.target_package}")
            
            # Try last known resumed activity first if available
            last_activity = self.device_manager.get_last_foreground_activity(self.target_package)
            candidate_activities: list[str] = []
            if last_activity:
                candidate_activities.append(last_activity)

            # Fallback to common default main activity
            candidate_activities.append(f"{self.target_package}/.MainActivity")
            
            for activity in candidate_activities:
                try:
                    log.info(f"Quick recovery: trying {activity}")
                    self.device_manager.device.shell(f"am start -n {activity}")
                    await asyncio.sleep(0.2)
                    
                    if await self._is_app_foreground():
                        log.success(f"Quick recovery successful with {activity}")
                        return True
                except Exception as e:
                    log.debug(f"Activity {activity} failed: {e}")
                    continue
            
            # Additional quick fallback common activities
            for fallback_activity in [
                f"{self.target_package}/.LauncherActivity",
                f"{self.target_package}/.ui.MainActivity"
            ]:
                try:
                    log.info(f"Quick recovery fallback: trying {fallback_activity}")
                    self.device_manager.device.shell(f"am start -n {fallback_activity}")
                    await asyncio.sleep(0.2)
                    if await self._is_app_foreground():
                        log.success(f"Quick recovery successful with {fallback_activity}")
                        return True
                except Exception:
                    continue
            
            log.warning("Quick recovery failed, app still not in foreground")
            return False
            
        except Exception as e:
            log.error(f"Quick recovery failed: {e}")
            return False
    
    async def _attempt_recovery(self) -> bool:
        """Attempt to recover the app to foreground using the single recovery strategy.
        
        Returns:
            bool: True if recovery successful
        """
        if not self.target_package:
            return False
        
        log.info(f"Attempting to recover {self.target_package} to foreground")
        
        try:
            success = await self.recovery_strategy()
            
            if success:
                log.success("Recovery successful with direct app launch")
                return True
            else:
                log.warning("Recovery failed with direct app launch")
                return False
                
        except Exception as e:
            log.error(f"Error in recovery strategy: {e}")
            return False
    

    
    async def _recover_with_app_launch(self) -> bool:
        """Recover app by launching it directly using adb shell am start.
        
        Returns:
            bool: True if recovery successful
        """
        try:
            if not self.device_manager.device:
                return False
            
            # Use cached activity or try common activities first (faster than discovery)
            common_activities = [
                f"{self.target_package}/.MainActivity",
                f"{self.target_package}/.ui.MainActivity",
                f"{self.target_package}/.activities.MainActivity",
                f"{self.target_package}/.LauncherActivity"
            ]
            
            # Try common activities first (fast path)
            for activity in common_activities:
                try:
                    log.info(f"Trying to launch: {activity}")
                    self.device_manager.device.shell(f"am start -n {activity}")
                    
                    # Quick check after a short delay
                    await asyncio.sleep(0.3)
                    
                    if await self._is_app_foreground():
                        log.success(f"Successfully launched app with: {activity}")
                        return True
                except Exception as e:
                    log.debug(f"Failed to launch {activity}: {e}")
                    continue
            
            # If common activities failed, try activity discovery (slower but more thorough)
            log.info("Common activities failed, trying activity discovery...")
            main_activity = await self._discover_main_activity()
            
            if main_activity:
                log.info(f"Launching app with discovered activity: {main_activity}")
                self.device_manager.device.shell(f"am start -n {main_activity}")
                await asyncio.sleep(0.5)
                
                if await self._is_app_foreground():
                    log.success(f"Successfully launched app with discovered activity: {main_activity}")
                    return True
            
            # Final check
            await asyncio.sleep(0.2)
            return await self._is_app_foreground()
            
        except Exception as e:
            log.error(f"App launch recovery failed: {e}")
            return False
    
    async def _discover_main_activity(self) -> str | None:
        """Discover the main activity for the target package.
        
        Returns:
            str | None: Main activity name or None if not found
        """
        try:
            if not self.device_manager.device or not self.target_package:
                return None
            
            # Use cmd package resolve-activity to find the main activity
            result = self.device_manager.device.shell(
                f"cmd package resolve-activity --brief {self.target_package}"
            )
            
            if result and "activity" in result:
                lines = result.strip().split('\n')
                for line in lines:
                    if self.target_package and self.target_package in line and "activity" in line:
                        # Extract the activity name from the output
                        # Format: package_name/activity_name
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            activity_name = parts[-1]  # Last part is usually the activity
                            log.info(f"Discovered main activity: {activity_name}")
                            return activity_name
            
            # Alternative: use dumpsys to find activities
            result = self.device_manager.device.shell(
                f"dumpsys package {self.target_package} | grep -A 5 'android.intent.action.MAIN'"
            )
            
            if result:
                lines = result.strip().split('\n')
                for line in lines:
                    if "android.intent.action.MAIN" in line and "android.intent.category.LAUNCHER" in line:
                        # Extract activity name from the line
                        if self.target_package and self.target_package in line:
                            # Look for the activity name pattern
                            import re
                            match = re.search(rf'{self.target_package}/([^\s]+)', line)
                            if match:
                                activity_name = f"{self.target_package}/{match.group(1)}"
                                log.info(f"Discovered launcher activity: {activity_name}")
                                return activity_name
            
            return None
            
        except Exception as e:
            log.debug(f"Failed to discover main activity: {e}")
            return None
    

    
    async def record_navigation_event(self, event_type: NavigationEvent, key_code: int) -> None:
        """Record a navigation event that was detected.
        
        Args:
            event_type: Type of navigation event
            key_code: The key code that was pressed
        """
        if not self.target_package:
            return
        
        app_was_foreground = await self._is_app_foreground()
        target_package = self.target_package  # Type narrowing
        if target_package is None:
            return
            
        event_record = NavigationEventRecord(
            event_type=event_type,
            timestamp=time.time(),
            target_package=target_package,
            app_was_foreground=app_was_foreground
        )
        self.navigation_events.append(event_record)
        
        log.info(f"Recorded navigation event: {event_type.value} (key_code: {key_code})")
    
    async def should_recover_after_action(self, action: Dict[str, Any]) -> bool:
        """Check if recovery should be attempted after a specific action.
        
        Args:
            action: The action that was just executed
            
        Returns:
            bool: True if recovery should be attempted
        """
        action_type = action.get("type")
        
        # Check if this is a navigation key event
        if action_type == "key_event":
            key_code = action.get("key_code")
            if key_code in self.navigation_key_codes:
                event_type = self.navigation_key_codes[key_code]
                
                # Record the navigation event
                await self.record_navigation_event(event_type, key_code)
                
                # For home and app switch, we should recover
                if event_type in [NavigationEvent.HOME_BUTTON, NavigationEvent.APP_SWITCH]:
                    return True
        
        return False
    
    async def ensure_app_foreground_after_action(self, action: Dict[str, Any]) -> bool:
        """Ensure app is in foreground after executing an action.
        
        Args:
            action: The action that was just executed
            
        Returns:
            bool: True if app is in foreground after recovery attempt
        """
        # Simple check - if app is not in foreground, recover immediately
        if not await self._is_app_foreground():
            log.warning("App not in foreground, attempting quick recovery")
            return await self._quick_recovery()
        
        return True
    
    def get_navigation_events(self) -> List[NavigationEventRecord]:
        """Get the history of navigation events.
        
        Returns:
            List of navigation event records
        """
        return self.navigation_events.copy()
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics.
        
        Returns:
            Dictionary with recovery statistics
        """
        total_events = len(self.navigation_events)
        recovery_attempts = sum(1 for event in self.navigation_events if event.recovery_attempted)
        successful_recoveries = sum(1 for event in self.navigation_events if event.recovery_successful)
        
        return {
            "total_navigation_events": total_events,
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": successful_recoveries / recovery_attempts if recovery_attempts > 0 else 0.0,
            "is_monitoring": self.is_monitoring,
            "target_package": self.target_package
        }
    
    def clear_history(self) -> None:
        """Clear the navigation event history."""
        self.navigation_events.clear()


# Global instance management
_app_foreground_recovery_manager: Optional[AppForegroundRecoveryManager] = None


def get_app_foreground_recovery_manager(device_manager: Optional[EnhancedDeviceManager] = None) -> AppForegroundRecoveryManager:
    """Get the global app foreground recovery manager instance.
    
    Args:
        device_manager: Device manager instance (required for first call)
        
    Returns:
        AppForegroundRecoveryManager instance
    """
    global _app_foreground_recovery_manager
    
    if _app_foreground_recovery_manager is None:
        if device_manager is None:
            raise ValueError("Device manager is required for first initialization")
        _app_foreground_recovery_manager = AppForegroundRecoveryManager(device_manager)
    
    return _app_foreground_recovery_manager 