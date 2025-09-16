"""Action execution engine for Android automation."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from ..core.logger import log
from ..core.device_manager import EnhancedDeviceManager


class ActionExecutor:
    """Executes automation actions with validation and error handling."""
    
    def __init__(self, device_manager: EnhancedDeviceManager):
        """Initialize the action executor.
        
        Args:
            device_manager: The device manager instance for device interactions.
        """
        self.device_manager = device_manager
        self.action_history: List[Dict[str, Any]] = []
        self.retry_count = 0
        self.max_retries = 3
        
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single automation action.
        
        Args:
            action: Action dictionary containing type and parameters.
            
        Returns:
            Dict containing execution result and metadata.
        """
        action_type = action.get("type")
        action_id = f"{action_type}_{int(time.time() * 1000)}"
        
        log.info(f"Executing action {action_id}: {action_type}")
        
        start_time = time.time()
        result = {
            "action_id": action_id,
            "action": action,
            "start_time": start_time,
            "success": False,
            "error": None,
            "retries": 0
        }
        
        try:
            # Validate action before execution
            if not self._validate_action(action):
                raise ValueError(f"Invalid action: {action}")
            
            # Execute action with retry logic
            success = await self._execute_with_retry(action)
            
            result["success"] = success
            result["end_time"] = time.time()
            result["duration"] = result["end_time"] - start_time
            
            if success:
                log.success(f"Action {action_id} completed successfully")
            else:
                log.warning(f"Action {action_id} failed after {self.max_retries} retries")
                
        except Exception as e:
            result["error"] = str(e)
            result["end_time"] = time.time()
            result["duration"] = result["end_time"] - start_time
            log.error(f"Action {action_id} failed: {e}")
            
        # Record action in history
        self.action_history.append(result)
        
        return result
    
    async def execute_sequence(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a sequence of actions.
        
        Args:
            actions: List of action dictionaries to execute in order.
            
        Returns:
            List of execution results for each action.
        """
        results = []
        
        for i, action in enumerate(actions):
            log.info(f"Executing action {i + 1}/{len(actions)}")
            
            result = await self.execute_action(action)
            results.append(result)
            
            # Check if we should continue after failure
            if not result["success"] and action.get("critical", False):
                log.error(f"Critical action failed, stopping sequence")
                break
                
            # Brief pause between actions
            await asyncio.sleep(0.5)
            
        return results
    
    def _validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate action structure and parameters.
        
        Args:
            action: Action dictionary to validate.
            
        Returns:
            True if action is valid, False otherwise.
        """
        required_fields = ["type"]
        action_type = action.get("type")
        
        # Check required fields
        for field in required_fields:
            if field not in action:
                log.error(f"Action missing required field: {field}")
                return False
        
        # Validate action type
        valid_types = ["tap", "swipe", "input_text", "key_event", "wait", "screenshot"]
        if action_type not in valid_types:
            log.error(f"Invalid action type: {action_type}")
            return False
        
        # Type-specific validation
        if action_type == "tap":
            return self._validate_tap_action(action)
        elif action_type == "swipe":
            return self._validate_swipe_action(action)
        elif action_type == "input_text":
            return self._validate_input_action(action)
        elif action_type == "key_event":
            return self._validate_key_action(action)
        elif action_type == "wait":
            return self._validate_wait_action(action)
        
        return True
    
    def _validate_tap_action(self, action: Dict[str, Any]) -> bool:
        """Validate tap action parameters."""
        if "x" not in action or "y" not in action:
            log.error("Tap action missing x or y coordinates")
            return False
        
        x, y = action["x"], action["y"]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            log.error("Tap coordinates must be numbers")
            return False
        
        return True
    
    def _validate_swipe_action(self, action: Dict[str, Any]) -> bool:
        """Validate swipe action parameters."""
        required = ["start_x", "start_y", "end_x", "end_y"]
        for coord in required:
            if coord not in action:
                log.error(f"Swipe action missing {coord}")
                return False
        
        return True
    
    def _validate_input_action(self, action: Dict[str, Any]) -> bool:
        """Validate input text action parameters."""
        if "text" not in action:
            log.error("Input action missing text")
            return False
        
        return True
    
    def _validate_key_action(self, action: Dict[str, Any]) -> bool:
        """Validate key event action parameters."""
        if "key_code" not in action:
            log.error("Key event action missing key_code")
            return False
        
        return True
    
    def _validate_wait_action(self, action: Dict[str, Any]) -> bool:
        """Validate wait action parameters."""
        duration = action.get("duration", 1.0)
        if not isinstance(duration, (int, float)) or duration < 0:
            log.error("Wait duration must be a positive number")
            return False
        
        return True
    
    async def _execute_with_retry(self, action: Dict[str, Any]) -> bool:
        """Execute action with retry logic.
        
        Args:
            action: Action to execute.
            
        Returns:
            True if execution succeeded, False otherwise.
        """
        for attempt in range(self.max_retries + 1):
            try:
                success = await self.device_manager.perform_action(action)
                if success:
                    return True
                    
                if attempt < self.max_retries:
                    log.warning(f"Action failed, retrying ({attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    
            except Exception as e:
                if attempt < self.max_retries:
                    log.warning(f"Action error, retrying ({attempt + 1}/{self.max_retries}): {e}")
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    log.error(f"Action failed after {self.max_retries} retries: {e}")
                    
        return False
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get the history of executed actions.
        
        Returns:
            List of action execution results.
        """
        return self.action_history.copy()
    
    def clear_history(self) -> None:
        """Clear the action execution history."""
        self.action_history.clear() 