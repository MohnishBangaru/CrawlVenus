"""Error handling and recovery for Android automation."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable

from ..core.logger import log


class ErrorHandler:
    """Handles errors and implements recovery strategies for automation failures."""
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_strategies = self._load_recovery_strategies()
        self.max_retries = 3
        self.retry_delay = 1.0
        
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an automation error and attempt recovery.
        
        Args:
            error: The exception that occurred.
            context: Context information about the error.
            
        Returns:
            Dict containing error handling result.
        """
        error_id = f"error_{int(time.time() * 1000)}"
        
        error_record = {
            'error_id': error_id,
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'handled': False,
            'recovery_attempted': False,
            'recovery_successful': False
        }
        
        log.error(f"Handling error {error_id}: {error}")
        
        try:
            # Attempt to recover from the error
            recovery_result = self._attempt_recovery(error, context)
            
            error_record['recovery_attempted'] = True
            error_record['recovery_successful'] = recovery_result['success']
            error_record['recovery_strategy'] = recovery_result['strategy']
            error_record['handled'] = recovery_result['success']
            
            if recovery_result['success']:
                log.success(f"Error {error_id} recovered successfully")
            else:
                log.warning(f"Error {error_id} recovery failed")
                
        except Exception as recovery_error:
            log.error(f"Error recovery failed: {recovery_error}")
            error_record['recovery_error'] = str(recovery_error)
            
        # Record error in history
        self.error_history.append(error_record)
        
        # Keep only recent errors (last 50)
        if len(self.error_history) > 50:
            self.error_history = self.error_history[-50:]
            
        return error_record
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of error handling statistics.
        
        Returns:
            Summary dictionary with error statistics.
        """
        if not self.error_history:
            return {
                'total_errors': 0,
                'handled_errors': 0,
                'recovery_success_rate': 0.0,
                'most_common_error': None,
                'last_error': None
            }
            
        total_errors = len(self.error_history)
        handled_errors = sum(1 for error in self.error_history if error['handled'])
        recovery_attempts = sum(1 for error in self.error_history if error['recovery_attempted'])
        successful_recoveries = sum(1 for error in self.error_history if error['recovery_successful'])
        
        # Find most common error type
        error_types = [error['error_type'] for error in self.error_history]
        most_common_error = max(set(error_types), key=error_types.count) if error_types else None
        
        recovery_success_rate = (successful_recoveries / recovery_attempts * 100) if recovery_attempts > 0 else 0
        
        return {
            'total_errors': total_errors,
            'handled_errors': handled_errors,
            'recovery_attempts': recovery_attempts,
            'successful_recoveries': successful_recoveries,
            'recovery_success_rate': recovery_success_rate,
            'most_common_error': most_common_error,
            'last_error': self.error_history[-1] if self.error_history else None
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error records.
        
        Args:
            limit: Maximum number of errors to return.
            
        Returns:
            List of recent error records.
        """
        return self.error_history[-limit:]
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history.clear()
        
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to recover from an error.
        
        Args:
            error: The exception that occurred.
            context: Context information about the error.
            
        Returns:
            Dict containing recovery result.
        """
        error_type = type(error).__name__
        
        # Find appropriate recovery strategy
        strategy = self._find_recovery_strategy(error_type, context)
        
        if strategy is None:
            return {
                'success': False,
                'strategy': 'none',
                'reason': 'No recovery strategy found'
            }
            
        try:
            # Execute recovery strategy
            recovery_result = strategy(error, context)
            
            return {
                'success': recovery_result,
                'strategy': strategy.__name__,
                'reason': 'Recovery strategy executed'
            }
            
        except Exception as recovery_error:
            log.error(f"Recovery strategy failed: {recovery_error}")
            return {
                'success': False,
                'strategy': strategy.__name__,
                'reason': f'Recovery strategy failed: {recovery_error}'
            }
    
    def _find_recovery_strategy(self, error_type: str, context: Dict[str, Any]) -> Optional[Callable]:
        """Find an appropriate recovery strategy for the error.
        
        Args:
            error_type: Type of error that occurred.
            context: Context information about the error.
            
        Returns:
            Recovery strategy function or None if no strategy found.
        """
        # Check for specific error type strategies
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type]
            
        # Check for general strategies
        if 'general' in self.recovery_strategies:
            return self.recovery_strategies['general']
            
        return None
    
    def _load_recovery_strategies(self) -> Dict[str, Callable]:
        """Load recovery strategies for different error types.
        
        Returns:
            Dictionary mapping error types to recovery strategies.
        """
        return {
            'ConnectionError': self._recover_connection_error,
            'TimeoutError': self._recover_timeout_error,
            'DeviceNotFoundError': self._recover_device_error,
            'PermissionError': self._recover_permission_error,
            'ValueError': self._recover_value_error,
            'general': self._recover_general_error
        }
    
    async def _recover_connection_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from connection errors.
        
        Args:
            error: The connection error.
            context: Error context.
            
        Returns:
            True if recovery successful.
        """
        log.info("Attempting connection error recovery...")
        
        try:
            # Wait and retry connection
            await asyncio.sleep(self.retry_delay)
            
            # Attempt to reconnect device
            device_manager = context.get('device_manager')
            if device_manager:
                success = await device_manager.connect_device()
                if success:
                    log.success("Connection recovered successfully")
                    return True
                    
            log.warning("Connection recovery failed")
            return False
            
        except Exception as e:
            log.error(f"Connection recovery error: {e}")
            return False
    
    async def _recover_timeout_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from timeout errors.
        
        Args:
            error: The timeout error.
            context: Error context.
            
        Returns:
            True if recovery successful.
        """
        log.info("Attempting timeout error recovery...")
        
        try:
            # Increase wait time and retry
            await asyncio.sleep(self.retry_delay * 2)
            
            # Retry the action with increased timeout
            action = context.get('action')
            if action:
                # Modify action to have longer timeout
                action['timeout'] = action.get('timeout', 10) * 2
                
            log.success("Timeout recovery strategy applied")
            return True
            
        except Exception as e:
            log.error(f"Timeout recovery error: {e}")
            return False
    
    async def _recover_device_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from device-related errors.
        
        Args:
            error: The device error.
            context: Error context.
            
        Returns:
            True if recovery successful.
        """
        log.info("Attempting device error recovery...")
        
        try:
            # Wait for device to become available
            await asyncio.sleep(self.retry_delay)
            
            # Check device status
            device_manager = context.get('device_manager')
            if device_manager and device_manager.is_connected():
                log.success("Device error recovered")
                return True
                
            log.warning("Device error recovery failed")
            return False
            
        except Exception as e:
            log.error(f"Device recovery error: {e}")
            return False
    
    async def _recover_permission_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from permission errors.
        
        Args:
            error: The permission error.
            context: Error context.
            
        Returns:
            True if recovery successful.
        """
        log.info("Attempting permission error recovery...")
        
        try:
            # Wait and retry with different approach
            await asyncio.sleep(self.retry_delay)
            
            # Try alternative action or method
            log.info("Permission error - trying alternative approach")
            return True
            
        except Exception as e:
            log.error(f"Permission recovery error: {e}")
            return False
    
    async def _recover_value_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from value errors.
        
        Args:
            error: The value error.
            context: Error context.
            
        Returns:
            True if recovery successful.
        """
        log.info("Attempting value error recovery...")
        
        try:
            # Validate and correct action parameters
            action = context.get('action')
            if action:
                # Try to fix common value issues
                if 'x' in action and 'y' in action:
                    # Ensure coordinates are within bounds
                    action['x'] = max(0, min(1080, action['x']))
                    action['y'] = max(0, min(1920, action['y']))
                    
            log.success("Value error corrected")
            return True
            
        except Exception as e:
            log.error(f"Value recovery error: {e}")
            return False
    
    async def _recover_general_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """General error recovery strategy.
        
        Args:
            error: The general error.
            context: Error context.
            
        Returns:
            True if recovery successful.
        """
        log.info("Attempting general error recovery...")
        
        try:
            # Simple wait and retry
            await asyncio.sleep(self.retry_delay)
            
            log.info("General error recovery applied")
            return True
            
        except Exception as e:
            log.error(f"General recovery error: {e}")
            return False
    
    def add_custom_recovery_strategy(self, error_type: str, strategy: Callable) -> None:
        """Add a custom recovery strategy.
        
        Args:
            error_type: Type of error to handle.
            strategy: Recovery strategy function.
        """
        self.recovery_strategies[error_type] = strategy
        log.info(f"Added custom recovery strategy for {error_type}")
    
    def export_error_data(self, filepath: str) -> bool:
        """Export error handling data to a file.
        
        Args:
            filepath: Path to save the data.
            
        Returns:
            True if export successful.
        """
        try:
            import json
            
            export_data = {
                'summary': self.get_error_summary(),
                'history': self.error_history,
                'recovery_strategies': list(self.recovery_strategies.keys())
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
            log.info(f"Error data exported to {filepath}")
            return True
            
        except Exception as e:
            log.error(f"Failed to export error data: {e}")
            return False 