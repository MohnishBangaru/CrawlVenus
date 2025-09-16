"""Intuitive App Recovery System for DroidBot-GPT framework.

This module provides comprehensive app recovery logic that ensures the target app
is always recovered if it leaves the foreground, and only remains active while
the target app is running. It includes intelligent detection, recovery strategies,
and monitoring capabilities.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .logger import log
from .device_manager import EnhancedDeviceManager


class RecoveryStrategy(Enum):
    """Different recovery strategies for app restoration."""
    FOREGROUND_SERVICE = "foreground_service"
    APP_LAUNCH = "app_launch"
    RECENT_APPS = "recent_apps"
    HOME_AND_LAUNCH = "home_and_launch"
    FORCE_STOP_RESTART = "force_stop_restart"
    CLEAR_RECENTS_LAUNCH = "clear_recents_launch"


class AppState(Enum):
    """App state enumeration."""
    FOREGROUND = "foreground"
    BACKGROUND = "background"
    STOPPED = "stopped"
    CRASHED = "crashed"
    UNKNOWN = "unknown"


@dataclass
class RecoveryAttempt:
    """Represents a recovery attempt with details."""
    strategy: RecoveryStrategy
    success: bool
    duration: float
    error_message: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class AppRecoveryConfig:
    """Configuration for app recovery behavior."""
    max_recovery_attempts: int = 5
    recovery_timeout: float = 30.0
    check_interval: float = 2.0
    enable_foreground_service: bool = True
    enable_force_restart: bool = True
    enable_clear_recents: bool = False
    recovery_strategies: List[RecoveryStrategy] = None
    
    def __post_init__(self):
        if self.recovery_strategies is None:
            self.recovery_strategies = [
                RecoveryStrategy.FOREGROUND_SERVICE,
                RecoveryStrategy.APP_LAUNCH,
                RecoveryStrategy.RECENT_APPS,
                RecoveryStrategy.HOME_AND_LAUNCH,
                RecoveryStrategy.FORCE_STOP_RESTART
            ]


class AppRecoveryManager:
    """Manages comprehensive app recovery with intelligent strategies."""
    
    def __init__(self, device_manager: EnhancedDeviceManager):
        """Initialize the app recovery manager.
        
        Args:
            device_manager: The device manager instance
        """
        self.device_manager = device_manager
        self.target_package: Optional[str] = None
        self.recovery_config = AppRecoveryConfig()
        self.recovery_history: List[RecoveryAttempt] = []
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_app_state = AppState.UNKNOWN
        self.consecutive_failures = 0
        self.last_successful_recovery = 0.0
        
        # Recovery strategy weights (higher = more likely to be tried first)
        self.strategy_weights = {
            RecoveryStrategy.FOREGROUND_SERVICE: 100,
            RecoveryStrategy.APP_LAUNCH: 80,
            RecoveryStrategy.RECENT_APPS: 60,
            RecoveryStrategy.HOME_AND_LAUNCH: 40,
            RecoveryStrategy.FORCE_STOP_RESTART: 20,
            RecoveryStrategy.CLEAR_RECENTS_LAUNCH: 10
        }
    
    async def start_recovery_monitoring(self, target_package: str) -> bool:
        """Start monitoring and automatic recovery for the target app.
        
        Args:
            target_package: Package name of the target app
            
        Returns:
            bool: True if monitoring started successfully
        """
        if self.is_monitoring:
            log.warning("Recovery monitoring already active")
            return True
        
        self.target_package = target_package
        self.is_monitoring = True
        self.consecutive_failures = 0
        self.last_successful_recovery = time.time()
        
        log.info(f"Starting app recovery monitoring for {target_package}")
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop()
        )
        
        return True
    
    async def stop_recovery_monitoring(self) -> bool:
        """Stop the recovery monitoring.
        
        Returns:
            bool: True if monitoring stopped successfully
        """
        if not self.is_monitoring:
            return True
        
        log.info("Stopping app recovery monitoring")
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.target_package = None
        return True
    
    async def _monitoring_loop(self):
        """Main monitoring loop that continuously checks app state."""
        while self.is_monitoring and self.target_package:
            try:
                # Check current app state
                current_state = await self._get_app_state()
                
                # Detect state changes
                if current_state != self.last_app_state:
                    log.info(f"App state changed: {self.last_app_state.value} -> {current_state.value}")
                    
                    # Handle state changes
                    if current_state in [AppState.BACKGROUND, AppState.STOPPED, AppState.CRASHED]:
                        await self._handle_app_loss()
                    
                    self.last_app_state = current_state
                
                # Wait before next check
                await asyncio.sleep(self.recovery_config.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.recovery_config.check_interval)
    
    async def _get_app_state(self) -> AppState:
        """Get the current state of the target app.
        
        Returns:
            AppState: Current state of the app
        """
        try:
            if not self.target_package:
                return AppState.UNKNOWN
            
            # Check if app is in foreground
            foreground_package = await self.device_manager.get_foreground_package()
            
            if foreground_package == self.target_package:
                return AppState.FOREGROUND
            
            # Check if app is running in background
            running_apps = await self.device_manager.get_running_apps()
            if self.target_package in running_apps:
                return AppState.BACKGROUND
            
            # Check if app is installed but not running
            installed_apps = await self.device_manager.get_installed_apps()
            if self.target_package in installed_apps:
                return AppState.STOPPED
            
            # App might be crashed or uninstalled
            return AppState.CRASHED
            
        except Exception as e:
            log.error(f"Error getting app state: {e}")
            return AppState.UNKNOWN
    
    async def _handle_app_loss(self):
        """Handle the case when the app is lost from foreground."""
        log.warning(f"App {self.target_package} lost from foreground, initiating recovery")
        
        # Attempt recovery
        success = await self.attempt_recovery()
        
        if success:
            self.consecutive_failures = 0
            self.last_successful_recovery = time.time()
            log.success(f"Successfully recovered {self.target_package}")
        else:
            self.consecutive_failures += 1
            log.error(f"Failed to recover {self.target_package} (attempt {self.consecutive_failures})")
            
            # If too many consecutive failures, stop monitoring
            if self.consecutive_failures >= self.recovery_config.max_recovery_attempts:
                log.error(f"Too many consecutive recovery failures, stopping monitoring")
                await self.stop_recovery_monitoring()
    
    async def attempt_recovery(self) -> bool:
        """Attempt to recover the target app using intelligent strategies.
        
        Returns:
            bool: True if recovery successful
        """
        if not self.target_package:
            return False
        
        log.info(f"Attempting to recover {self.target_package}")
        
        # Get prioritized recovery strategies
        strategies = self._get_prioritized_strategies()
        
        for strategy in strategies:
            log.info(f"Trying recovery strategy: {strategy.value}")
            
            start_time = time.time()
            success = False
            error_message = None
            
            try:
                success = await self._execute_recovery_strategy(strategy)
            except Exception as e:
                error_message = str(e)
                success = False
            
            duration = time.time() - start_time
            
            # Record recovery attempt
            attempt = RecoveryAttempt(
                strategy=strategy,
                success=success,
                duration=duration,
                error_message=error_message
            )
            self.recovery_history.append(attempt)
            
            if success:
                log.success(f"Recovery successful with {strategy.value} in {duration:.2f}s")
                return True
            else:
                log.warning(f"Recovery failed with {strategy.value}: {error_message}")
        
        log.error("All recovery strategies failed")
        return False
    
    def _get_prioritized_strategies(self) -> List[RecoveryStrategy]:
        """Get recovery strategies ordered by priority and success rate.
        
        Returns:
            List of recovery strategies in priority order
        """
        # Start with configured strategies
        strategies = self.recovery_config.recovery_strategies.copy()
        
        # Adjust based on recent success rates
        recent_attempts = self.recovery_history[-10:]  # Last 10 attempts
        
        if recent_attempts:
            # Calculate success rates for each strategy
            success_rates = {}
            for strategy in RecoveryStrategy:
                strategy_attempts = [a for a in recent_attempts if a.strategy == strategy]
                if strategy_attempts:
                    success_rate = sum(1 for a in strategy_attempts if a.success) / len(strategy_attempts)
                    success_rates[strategy] = success_rate
            
            # Reorder strategies based on success rates
            strategies.sort(key=lambda s: success_rates.get(s, 0.0), reverse=True)
        
        # Ensure foreground service is tried first if enabled
        if (self.recovery_config.enable_foreground_service and 
            RecoveryStrategy.FOREGROUND_SERVICE in strategies):
            strategies.remove(RecoveryStrategy.FOREGROUND_SERVICE)
            strategies.insert(0, RecoveryStrategy.FOREGROUND_SERVICE)
        
        return strategies
    
    async def _execute_recovery_strategy(self, strategy: RecoveryStrategy) -> bool:
        """Execute a specific recovery strategy.
        
        Args:
            strategy: The recovery strategy to execute
            
        Returns:
            bool: True if strategy executed successfully
        """
        try:
            if strategy == RecoveryStrategy.FOREGROUND_SERVICE:
                return await self._recover_with_foreground_service()
            elif strategy == RecoveryStrategy.APP_LAUNCH:
                return await self._recover_with_app_launch()
            elif strategy == RecoveryStrategy.RECENT_APPS:
                return await self._recover_with_recent_apps()
            elif strategy == RecoveryStrategy.HOME_AND_LAUNCH:
                return await self._recover_with_home_and_launch()
            elif strategy == RecoveryStrategy.FORCE_STOP_RESTART:
                return await self._recover_with_force_stop_restart()
            elif strategy == RecoveryStrategy.CLEAR_RECENTS_LAUNCH:
                return await self._recover_with_clear_recents_launch()
            else:
                log.error(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            log.error(f"Error executing recovery strategy {strategy.value}: {e}")
            return False
    
    async def _recover_with_foreground_service(self) -> bool:
        """Recover using foreground service approach."""
        try:
            # Check if foreground service is running
            is_running = await self.device_manager.is_foreground_service_running(self.target_package)
            
            if not is_running:
                # Try to start foreground service
                success = await self.device_manager.setup_foreground_service(self.target_package)
                if not success:
                    return False
            
            # Wait a moment for service to take effect
            await asyncio.sleep(2.0)
            
            # Check if app is now in foreground
            current_state = await self._get_app_state()
            return current_state == AppState.FOREGROUND
            
        except Exception as e:
            log.error(f"Foreground service recovery failed: {e}")
            return False
    
    async def _recover_with_app_launch(self) -> bool:
        """Recover by launching the app directly."""
        try:
            # Launch the app
            success = await self.device_manager.launch_app(self.target_package)
            if not success:
                return False
            
            # Wait for app to start
            await asyncio.sleep(3.0)
            
            # Check if app is now in foreground
            current_state = await self._get_app_state()
            return current_state == AppState.FOREGROUND
            
        except Exception as e:
            log.error(f"App launch recovery failed: {e}")
            return False
    
    async def _recover_with_recent_apps(self) -> bool:
        """Recover by using recent apps switcher."""
        try:
            # Open recent apps
            await self.device_manager.perform_action({
                "type": "key_event",
                "key_code": "KEYCODE_APP_SWITCH"
            })
            
            await asyncio.sleep(1.0)
            
            # Look for the target app in recent apps and tap it
            # This would require vision analysis to find the app icon
            # For now, we'll use a simpler approach
            
            # Close recent apps and try direct launch
            await self.device_manager.perform_action({
                "type": "key_event",
                "key_code": "KEYCODE_HOME"
            })
            
            await asyncio.sleep(1.0)
            
            # Try launching the app
            return await self._recover_with_app_launch()
            
        except Exception as e:
            log.error(f"Recent apps recovery failed: {e}")
            return False
    
    async def _recover_with_home_and_launch(self) -> bool:
        """Recover by going home and then launching the app."""
        try:
            # Go to home screen
            await self.device_manager.perform_action({
                "type": "key_event",
                "key_code": "KEYCODE_HOME"
            })
            
            await asyncio.sleep(1.0)
            
            # Launch the app
            return await self._recover_with_app_launch()
            
        except Exception as e:
            log.error(f"Home and launch recovery failed: {e}")
            return False
    
    async def _recover_with_force_stop_restart(self) -> bool:
        """Recover by force stopping and restarting the app."""
        try:
            if not self.recovery_config.enable_force_restart:
                return False
            
            # Force stop the app
            await self.device_manager.perform_action({
                "type": "shell_command",
                "command": f"am force-stop {self.target_package}"
            })
            
            await asyncio.sleep(2.0)
            
            # Launch the app
            return await self._recover_with_app_launch()
            
        except Exception as e:
            log.error(f"Force stop restart recovery failed: {e}")
            return False
    
    async def _recover_with_clear_recents_launch(self) -> bool:
        """Recover by clearing recent apps and launching."""
        try:
            if not self.recovery_config.enable_clear_recents:
                return False
            
            # Clear recent apps
            await self.device_manager.perform_action({
                "type": "shell_command",
                "command": "am kill-all"
            })
            
            await asyncio.sleep(2.0)
            
            # Launch the app
            return await self._recover_with_app_launch()
            
        except Exception as e:
            log.error(f"Clear recents launch recovery failed: {e}")
            return False
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics.
        
        Returns:
            Dictionary with recovery statistics
        """
        if not self.recovery_history:
            return {
                "total_attempts": 0,
                "successful_attempts": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "strategy_success_rates": {},
                "is_monitoring": self.is_monitoring,
                "consecutive_failures": self.consecutive_failures
            }
        
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for a in self.recovery_history if a.success)
        success_rate = successful_attempts / total_attempts
        average_duration = sum(a.duration for a in self.recovery_history) / total_attempts
        
        # Calculate strategy success rates
        strategy_success_rates = {}
        for strategy in RecoveryStrategy:
            strategy_attempts = [a for a in self.recovery_history if a.strategy == strategy]
            if strategy_attempts:
                strategy_success_rate = sum(1 for a in strategy_attempts if a.success) / len(strategy_attempts)
                strategy_success_rates[strategy.value] = strategy_success_rate
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": success_rate,
            "average_duration": average_duration,
            "strategy_success_rates": strategy_success_rates,
            "is_monitoring": self.is_monitoring,
            "consecutive_failures": self.consecutive_failures,
            "last_successful_recovery": self.last_successful_recovery,
            "target_package": self.target_package
        }
    
    def update_config(self, config: AppRecoveryConfig) -> None:
        """Update the recovery configuration.
        
        Args:
            config: New recovery configuration
        """
        self.recovery_config = config
        log.info("App recovery configuration updated")
    
    def clear_history(self) -> None:
        """Clear the recovery history."""
        self.recovery_history.clear()
        log.info("App recovery history cleared")


# Global instance
_app_recovery_manager: Optional[AppRecoveryManager] = None


def get_app_recovery_manager(device_manager: Optional[EnhancedDeviceManager] = None) -> AppRecoveryManager:
    """Get the global app recovery manager instance.
    
    Args:
        device_manager: Device manager instance (required for first call)
        
    Returns:
        AppRecoveryManager: The recovery manager instance
    """
    global _app_recovery_manager
    
    if _app_recovery_manager is None:
        if device_manager is None:
            raise ValueError("Device manager is required for first initialization")
        _app_recovery_manager = AppRecoveryManager(device_manager)
    
    return _app_recovery_manager 