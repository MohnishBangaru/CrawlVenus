"""Main DroidBot-GPT framework class - central orchestrator for intelligent Android automation."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# Standard library
from typing import TYPE_CHECKING, Any, cast

# Import VisionEngine only for type checking to avoid heavy runtime dependency
if TYPE_CHECKING:
    from src.vision.engine import VisionEngine

from ..core.device_manager import EnhancedDeviceManager
from ..core.element_tracker import get_element_tracker
from ..core.state_tracker import get_state_tracker
from ..core.app_recovery import get_app_recovery_manager
from ..core.app_foreground_recovery import get_app_foreground_recovery_manager
from ..core.logger import log


class DroidBotGPT:
    """Main DroidBot-GPT framework class.
    
    This class serves as the central orchestrator for intelligent Android automation,
    coordinating between device management, AI decision-making, computer vision,
    and automation execution with element exploration tracking.
    """
    
    def __init__(self) -> None:
        """Initialize a new DroidBotGPT session."""
        self.device_manager = EnhancedDeviceManager()
        self.element_tracker = get_element_tracker()
        self.state_tracker = get_state_tracker()
        self.app_recovery_manager = get_app_recovery_manager(self.device_manager)
        self.app_foreground_recovery_manager = get_app_foreground_recovery_manager(self.device_manager)
        self.session_id = self._generate_session_id()
        self.task_history: list[dict[str, Any]] = []
        self.current_task: dict[str, Any] | None = None
        self.is_running = False
        # Vision engine is created lazily; keep forward reference for typing (no runtime import).
        self._vision_engine: VisionEngine | None = None
        self._setup_directories()
        
        log.info(f"DroidBot-GPT initialized with session ID: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"session_{int(time.time())}"
    
    def _setup_directories(self) -> None:
        """Create necessary directories for the session."""
        try:
            # Create session directory
            session_dir = os.path.join("sessions", self.session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # Create subdirectories
            for subdir in ["screenshots", "logs", "actions", "analysis"]:
                os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)
            
            log.info(f"Session directories created: {session_dir}")
            
        except Exception as e:
            log.error(f"Failed to create session directories: {e}")
            raise
    
    async def connect_device(self, device_serial: str | None = None) -> bool:
        """Connect to Android device or emulator.
        
        Args:
            device_serial: Optional device serial number.
            
        Returns:
            bool: True if connection successful.

        """
        try:
            log.info("Connecting to Android device...")
            success = await self.device_manager.connect_device(device_serial)
            
            if success:
                device_info = self.device_manager.get_device_info()
                if device_info:
                    log.success(
                        f"Connected to {device_info.model} "
                        f"(Android {device_info.android_version})"
                    )
                    
                    # Log device capabilities
                    res_w, res_h = device_info.screen_size
                    log.info(f"Screen resolution: {res_w}x{res_h}")
                    log.info(f"Available memory: {device_info.available_memory}MB")
                    log.info(f"Emulator: {device_info.is_emulator}")
                    
                    return True
                else:
                    log.error("Failed to get device information")
                    return False
            else:
                log.error("Failed to connect to device")
                return False
                
        except Exception as e:
            log.error(f"Device connection failed: {e}")
            return False
    
    async def automate_task(self, task_description: str, max_steps: int = 50, target_package: str | None = None) -> dict[str, Any]:
        """Perform intelligent automation based on natural language task description.
        
        Args:
            task_description: Natural language description of the task.
            max_steps: Maximum number of automation steps to perform.
            target_package: Package name of the target app to automate (optional).
            
        Returns:
            dict: Task execution results with metadata.

        """
        if not self.device_manager.is_connected():
            raise RuntimeError("Device not connected. Call connect_device() first.")
        
        log.info(f"Starting automation task: {task_description}")
        if target_package:
            log.info(f"Target app package: {target_package}")
        
        # Reset element exploration and state tracking for new task
        self.element_tracker.reset_exploration()
        self.state_tracker.reset_state_tracking()
        log.info("Element exploration and state tracking reset for new task")
        
        # Initialize task
        task = {
            "id": f"task_{len(self.task_history) + 1}",
            "description": task_description,
            "start_time": time.time(),
            "steps": [],
            "status": "running",
            "max_steps": max_steps,
            "session_id": self.session_id,
            "target_package": target_package
        }
        
        self.current_task = task
        self.is_running = True
        
        try:
            # Set up foreground service for target app if specified
            if target_package:
                log.info(f"Setting up foreground service for {target_package}")
                foreground_service_setup = await self.setup_foreground_service(target_package)
                if foreground_service_setup:
                    log.success("Foreground service setup successful")
                else:
                    log.warning("Foreground service setup failed, continuing with standard app state validation")
                
                # Start app recovery monitoring
                log.info(f"Starting app recovery monitoring for {target_package}")
                recovery_monitoring_started = await self.start_app_recovery_monitoring(target_package)
                if recovery_monitoring_started:
                    log.success("App recovery monitoring started successfully")
                else:
                    log.warning("App recovery monitoring failed to start, continuing with standard monitoring")
                
                # Start app foreground recovery monitoring
                log.info(f"Starting app foreground recovery monitoring for {target_package}")
                foreground_recovery_started = await self.app_foreground_recovery_manager.start_monitoring(target_package)
                if foreground_recovery_started:
                    log.success("App foreground recovery monitoring started successfully")
                else:
                    log.warning("App foreground recovery monitoring failed to start, continuing with standard monitoring")
            
            # Main automation loop
            step_count = 0
            while self.is_running and step_count < max_steps:
                log.info(f"Automation step {step_count + 1}/{max_steps}")
                
                # APP STATE VALIDATION: Check if target app is running and in foreground
                if target_package:
                    app_status = await self._validate_app_state(target_package)
                    if not app_status["is_valid"]:
                        log.warning(f"App state validation failed: {app_status['reason']}")
                        
                        # Try to recover app state
                        recovery_success = await self._recover_app_state(target_package)
                        if not recovery_success:
                            log.error("Failed to recover app state. Stopping automation.")
                            task["status"] = "failed"
                            task["error"] = f"App state recovery failed: {app_status['reason']}"
                            break
                        
                        # Re-validate after recovery attempt
                        app_status = await self._validate_app_state(target_package)
                        if not app_status["is_valid"]:
                            log.error("App state still invalid after recovery attempt. Stopping automation.")
                            task["status"] = "failed"
                            task["error"] = f"App state recovery failed: {app_status['reason']}"
                            break
                    
                    log.info(f"App state validated: {app_status['status']}")
                
                # Capture current state
                current_state = await self._capture_current_state()
                
                # Analyze state and determine next action
                next_action = await self._determine_next_action(current_state, task_description)
                
                if next_action is None:
                    log.info("No more actions needed. Task appears complete.")
                    break
                
                # Execute action
                action_result = await self._execute_action(next_action)
                
                # Record step
                step = {
                    "step_number": step_count + 1,
                    "action": next_action,
                    "result": action_result,
                    "timestamp": time.time(),
                    "state": current_state,
                    "app_status": app_status if target_package else None
                }
                
                cast(list[dict[str, Any]], task["steps"]).append(step)
                step_count += 1
                
                # Log exploration and state statistics
                exploration_stats = self.element_tracker.get_exploration_stats()
                state_stats = self.state_tracker.get_state_exploration_stats()
                log.info(f"Exploration stats: {exploration_stats['total_explored_elements']} explored, "
                        f"{exploration_stats['recent_explorations']} recent")
                log.info(f"State stats: {state_stats['total_visited_states']} visited states, "
                        f"{state_stats['recent_state_visits']} recent visits")
                
                # Brief pause between actions
                await asyncio.sleep(1)
                
                # Check for completion conditions
                if await self._check_task_completion(task_description, current_state):
                    log.success("Task completion detected!")
                    break
            
            # Finalize task
            task_end = time.time()
            task_start = cast(float, task["start_time"])
            task_duration = task_end - task_start
            task["end_time"] = task_end
            task["duration"] = task_duration
            task["status"] = "completed" if step_count < max_steps else "max_steps_reached"
            task["total_steps"] = step_count
            
            # Clean up foreground service if it was set up
            if target_package:
                log.info(f"Cleaning up foreground service for {target_package}")
                await self.stop_foreground_service(target_package)
                
                # Stop app foreground recovery monitoring
                log.info(f"Stopping app foreground recovery monitoring for {target_package}")
                await self.app_foreground_recovery_manager.stop_monitoring()
            
            # Add exploration and state statistics to task results
            final_exploration_stats = self.element_tracker.get_exploration_stats()
            final_state_stats = self.state_tracker.get_state_exploration_stats()
            task["exploration_stats"] = final_exploration_stats
            task["state_stats"] = final_state_stats
            
            self.task_history.append(task)
            self.current_task = None
            self.is_running = False
            
            log.success(f"Task completed in {task['duration']:.2f}s with {step_count} steps")
            log.info(f"Final exploration stats: {final_exploration_stats['total_explored_elements']} elements explored")
            log.info(f"Final state stats: {final_state_stats['total_visited_states']} states visited")
            
            return task
            
        except Exception as e:
            log.error(f"Task automation failed: {e}")
            task["status"] = "failed"
            task["error"] = str(e)
            task["end_time"] = time.time()
            task["duration"] = cast(float, task["end_time"]) - cast(
                    float, task["start_time"]
                )
            
            self.task_history.append(task)
            self.current_task = None
            self.is_running = False
            
            raise
    
    async def _capture_current_state(self) -> dict[str, Any]:
        """Capture current device state including screenshot and UI elements."""
        try:
            # Capture screenshot
            screenshot_path = await self.device_manager.capture_screenshot()
            
            # Store screenshot path in current task for LLM analysis
            if self.current_task:
                self.current_task["current_screenshot"] = screenshot_path
            
            # Get resource usage
            resource_usage = await self.device_manager.get_resource_usage()
            
            # Vision analysis (Phase 2)
            ui_elements: list[Any] = []  # default empty list
            try:
                # Lazily create and cache VisionEngine to reuse its thread-pool
                if self._vision_engine is None:
                    from src.vision.engine import VisionEngine  # local import only once
                    self._vision_engine = VisionEngine()
                if self._vision_engine is not None:
                    ui_elements = self._vision_engine.analyze(screenshot_path)
            except Exception as e:  # pragma: no cover - vision errors shouldn't crash
                log.warning(f"Vision analysis failed: {e}")

            state = {
                "timestamp": time.time(),
                "screenshot_path": screenshot_path,
                "resource_usage": resource_usage,
                "ui_elements": ui_elements,
                "device_info": self.device_manager.get_device_info()
            }
            
            log.debug(f"Captured device state: {screenshot_path}")
            return state
            
        except Exception as e:
            log.error(f"Failed to capture current state: {e}")
            return {"error": str(e)}
    
    async def _determine_next_action(
        self,
        current_state: dict[str, Any],
        task_description: str,
    ) -> dict[str, Any] | None:
        """Determine the next action based on current state and task description using action prioritization."""
        if "error" in current_state:
            return None

        try:
            # Get UI elements from current state
            ui_elements_raw = current_state.get("ui_elements", [])
            device_info = current_state.get("device_info", {})
            vision_analysis = current_state.get("vision_analysis", {})
            
            # Convert UIElement objects to dictionaries for LLM analysis
            ui_elements = []
            for element in ui_elements_raw:
                if hasattr(element, 'bbox') and hasattr(element, 'text'):
                    # Convert UIElement to dictionary format
                    element_dict = {
                        'text': element.text,
                        'element_type': getattr(element, 'element_type', 'text'),
                        'confidence': getattr(element, 'confidence', 0.0),
                        'bounds': {
                            'x': element.bbox.left,
                            'y': element.bbox.top,
                            'x2': element.bbox.right,
                            'y2': element.bbox.bottom,
                            'width': element.bbox.width(),
                            'height': element.bbox.height()
                        }
                    }
                    ui_elements.append(element_dict)
                elif isinstance(element, dict):
                    # Already a dictionary
                    ui_elements.append(element)
            
            # Get action history
            action_history = (
                [step["action"] for step in self.current_task.get("steps", [])]
                if self.current_task
                else []
            )
            
            # Get current app context for popup classification
            app_context = self.current_task.get("target_package", "") if self.current_task else ""
            
            # Get LLM analysis for action prioritization
            llm_analysis = await self._get_llm_analysis_for_prioritization(
                ui_elements, task_description, action_history
            )
            
            # Use action prioritizer for optimal action selection
            from src.core.action_prioritizer import get_action_prioritizer
            
            prioritizer = get_action_prioritizer()
            
            # Get current screenshot path for Phi Ground
            screenshot_path = None
            if self.current_task and "current_screenshot" in self.current_task:
                screenshot_path = self.current_task["current_screenshot"]
            
            optimal_action = prioritizer.get_optimal_action(
                ui_elements=ui_elements,
                llm_analysis=llm_analysis,
                vision_analysis=vision_analysis,
                task_description=task_description,
                action_history=action_history,
                screenshot_path=screenshot_path
            )
            
            if optimal_action:
                # Convert PrioritizedAction to action dict format
                next_action = self._convert_prioritized_action_to_dict(optimal_action)
                
                log.info(f"Selected optimal action: {next_action.get('type')} - {next_action.get('reasoning')}")
                log.info(f"Action score: {optimal_action.score:.2f}")
                log.info(f"LLM confidence: {optimal_action.llm_confidence:.2f}")
                log.info(f"Vision confidence: {optimal_action.vision_confidence:.2f}")
                log.info(f"Exploration bonus: {optimal_action.exploration_bonus:.2f}")
                
                return next_action
            else:
                log.warning("No optimal action found")
                return {
                    "type": "wait",
                    "duration": 1.0,
                    "reasoning": "No suitable actions available"
                }
            
        except Exception as e:
            log.error(f"Action determination failed: {e}")
            return {
                "type": "wait",
                "duration": 1.0,
                "reasoning": f"Error in action determination: {str(e)}"
            }
    
    async def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute the determined action."""
        try:
            action_type = action.get("type")
            log.log_automation_step(f"Executing action: {action_type}", action)
            
            # Execute action through device manager
            success = await self.device_manager.perform_action(action)
            
            result = {
                "success": success,
                "action": action,
                "timestamp": time.time()
            }
            
            if success:
                log.success(f"Action {action_type} executed successfully")
                
                # Check if this action was a navigation event that requires app recovery
                if self.current_task and self.current_task.get("target_package"):
                    target_package = self.current_task.get("target_package")
                    if target_package:
                        # Start foreground recovery monitoring if not already started
                        if not self.app_foreground_recovery_manager.is_monitoring:
                            await self.app_foreground_recovery_manager.start_monitoring(target_package)
                        
                        # Check if we need to recover the app after this action
                        app_in_foreground = await self.app_foreground_recovery_manager.ensure_app_foreground_after_action(action)
                        
                        if not app_in_foreground:
                            log.warning("App left foreground after action, recovery may be needed")
                            result["app_recovery_needed"] = True
                        else:
                            result["app_recovery_needed"] = False
            else:
                log.warning(f"Action {action_type} may have failed")
            
            return result
            
        except Exception as e:
            log.error(f"Action execution failed: {e}")
            return {
                "success": False,
                "action": action,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _check_task_completion(
        self,
        task_description: str,
        current_state: dict[str, Any],
    ) -> bool:
        """Check if the task has been completed.
        
        TODO: This is a placeholder for Phase 3 (AI Integration).
        For now, return False to continue automation.
        """
        # Placeholder implementation
        # In Phase 3, this will use AI to determine if the task is complete
        return False
    
    async def _validate_app_state(self, target_package: str) -> dict[str, Any]:
        """Validate that the target app is running and in foreground.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            dict: App state validation result with status and reason.
        """
        return await self.device_manager.validate_app_state(target_package)
    
    async def _recover_app_state(self, target_package: str) -> bool:
        """Attempt to recover app state by bringing it to foreground.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            bool: True if recovery successful, False otherwise.
        """
        return await self.device_manager.recover_app_state(target_package)

    async def setup_foreground_service(self, target_package: str) -> bool:
        """Set up a foreground service with persistent notification for the target app.
        
        This creates a foreground service that will help keep the app in the foreground
        during automation by providing a persistent notification.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            bool: True if foreground service setup successful, False otherwise.
        """
        return await self.device_manager.setup_foreground_service(target_package)

    async def stop_foreground_service(self, target_package: str) -> bool:
        """Stop the foreground service and remove the persistent notification.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            bool: True if service stopped successfully, False otherwise.
        """
        return await self.device_manager.stop_foreground_service(target_package)

    async def is_foreground_service_running(self, target_package: str) -> bool:
        """Check if the foreground service is running for the target app.
        
        Args:
            target_package: Package name of the target app.
            
        Returns:
            bool: True if foreground service is running, False otherwise.
        """
        return await self.device_manager.is_foreground_service_running(target_package)
    
    async def stop_automation(self) -> None:
        """Stop the current automation task."""
        if self.is_running:
            log.info("Stopping automation...")
            self.is_running = False
            
            # Stop app recovery monitoring
            await self.stop_app_recovery_monitoring()
            
            if self.current_task:
                self.current_task["status"] = "stopped"
                self.current_task["end_time"] = time.time()
                self.current_task["duration"] = cast(float, self.current_task["end_time"]) - cast(
                    float, self.current_task["start_time"]
                )
                
                self.task_history.append(self.current_task)
                self.current_task = None
            
            log.success("Automation stopped")
    
    async def get_task_history(self) -> list[dict[str, Any]]:
        """Get the history of all executed tasks."""
        return self.task_history.copy()
    
    async def get_current_task(self) -> dict[str, Any] | None:
        """Get the currently running task."""
        return self.current_task.copy() if self.current_task else None
    
    async def get_device_status(self) -> dict[str, Any]:
        """Get current device status and capabilities."""
        if not self.device_manager.is_connected():
            return {"connected": False}
        
        device_info = self.device_manager.get_device_info()
        resource_usage = await self.device_manager.get_resource_usage()
        
        return {
            "connected": True,
            "device_info": device_info.__dict__ if device_info else {},
            "resource_usage": resource_usage,
            "session_id": self.session_id
        }
    
    async def get_exploration_stats(self) -> dict[str, Any]:
        """Get current element exploration statistics."""
        return self.element_tracker.get_exploration_stats()
    
    async def get_state_stats(self) -> dict[str, Any]:
        """Get current state exploration statistics."""
        return self.state_tracker.get_state_exploration_stats()
    
    def set_exploration_strategy(self, strategy: str) -> None:
        """Set the element exploration strategy.
        
        Args:
            strategy: One of "unseen_first", "confidence_based", or "hybrid"
        """
        self.element_tracker.set_exploration_strategy(strategy)
    
    def reset_exploration(self) -> None:
        """Reset element exploration tracking."""
        self.element_tracker.reset_exploration()
        log.info("Element exploration tracking reset")
    
    async def disconnect(self) -> None:
        """Disconnect from device and cleanup."""
        if self.is_running:
            await self.stop_automation()
        
        await self.device_manager.disconnect()
        log.info("DroidBot-GPT disconnected")
    
    async def _get_llm_analysis_for_prioritization(
        self,
        ui_elements: list[dict[str, Any]],
        task_description: str,
        action_history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Get LLM analysis for action prioritization.
        
        Args:
            ui_elements: List of UI elements
            task_description: Current task description
            action_history: History of previous actions
            
        Returns:
            LLM analysis results for prioritization
        """
        try:
            # Import here to avoid circular imports
            from src.ai.openai_client import get_openai_client
            from src.ai.prompt_builder import build_action_prioritization_prompt
            
            openai_client = get_openai_client()
            
            # Generate screenshot with bounding boxes for LLM analysis
            screenshot_with_boxes_base64 = None
            if self.current_task and "current_screenshot" in self.current_task:
                screenshot_path = self.current_task["current_screenshot"]
                if screenshot_path and os.path.exists(screenshot_path):
                    log.info(f"🖼️ Generating base64 screenshot with bounding boxes for LLM analysis...")
                    screenshot_with_boxes_base64 = await self._generate_screenshot_with_boxes_for_llm(
                        screenshot_path, ui_elements
                    )
                    
                    if screenshot_with_boxes_base64:
                        log.success(f"✅ Generated base64 screenshot ({len(screenshot_with_boxes_base64)} chars) for visual LLM analysis")
                    else:
                        log.warning("⚠️ Failed to generate base64 screenshot, proceeding with text-only analysis")
                else:
                    log.debug("No current screenshot available for visual analysis")
            else:
                log.debug("No current task or screenshot path available for visual analysis")
            
            # Build prompt for action prioritization
            log.info(f"🤖 Building LLM prompt with {len(ui_elements)} UI elements")
            if screenshot_with_boxes_base64:
                log.info("📸 Including visual context (base64 screenshot) in LLM analysis")
            
            prompt = build_action_prioritization_prompt(
                ui_elements=ui_elements,
                task_description=task_description,
                action_history=action_history,
                screenshot_with_boxes_base64=screenshot_with_boxes_base64
            )
            
            # Get LLM response with fallback for vision model issues
            log.info("🧠 Sending request to LLM for action prioritization analysis...")
            
            try:
                response = await openai_client.get_completion(prompt)
                vision_analysis_successful = True
            except Exception as vision_error:
                log.warning(f"Vision analysis failed: {vision_error}")
                
                # Fallback to text-only analysis
                if screenshot_with_boxes_base64:
                    log.info("🔄 Falling back to text-only analysis...")
                    text_only_prompt = build_action_prioritization_prompt(
                        ui_elements=ui_elements,
                        task_description=task_description,
                        action_history=action_history,
                        screenshot_with_boxes_base64=None  # No image
                    )
                    
                    try:
                        response = await openai_client.get_completion(text_only_prompt)
                        vision_analysis_successful = False
                        log.info("✅ Text-only analysis completed successfully")
                    except Exception as text_error:
                        log.error(f"Text-only analysis also failed: {text_error}")
                        return {
                            'suggestions': [],
                            'confidence': 0.3,
                            'reasoning': 'LLM analysis unavailable',
                            'element_priorities': {},
                            'screenshot_analyzed': False
                        }
                else:
                    # No image was provided, so this was already text-only
                    raise vision_error
            
            # Parse response
            try:
                analysis = response.get('analysis', {})
                result = {
                    'suggestions': analysis.get('suggestions', []),
                    'confidence': analysis.get('confidence', 0.5),
                    'reasoning': analysis.get('reasoning', ''),
                    'element_priorities': analysis.get('element_priorities', {}),
                    'screenshot_analyzed': screenshot_with_boxes_base64 is not None and vision_analysis_successful
                }
                
                if result['screenshot_analyzed']:
                    log.success(f"🎯 LLM analysis completed with visual context - Confidence: {result['confidence']:.2f}")
                else:
                    log.info(f"📝 LLM analysis completed (text-only) - Confidence: {result['confidence']:.2f}")
                
                return result
                
            except Exception as e:
                log.warning(f"Failed to parse LLM analysis: {e}")
                return {
                    'suggestions': [],
                    'confidence': 0.3,
                    'reasoning': 'LLM analysis parsing failed',
                    'element_priorities': {},
                    'screenshot_analyzed': False
                }
                
        except Exception as e:
            log.warning(f"LLM analysis failed: {e}")
            return {
                'suggestions': [],
                'confidence': 0.3,
                'reasoning': 'LLM analysis unavailable',
                'element_priorities': {},
                'screenshot_analyzed': False
            }
    
    async def _generate_screenshot_with_boxes_for_llm(
        self, 
        screenshot_path: str, 
        ui_elements: list[dict[str, Any]]
    ) -> str | None:
        """Generate a screenshot with bounding boxes for LLM analysis.
        
        Args:
            screenshot_path: Path to the original screenshot
            ui_elements: List of UI elements with bounding box information
            
        Returns:
            Base64 encoded image data, or None if failed
        """
        try:
            import cv2
            import numpy as np
            import base64
            from io import BytesIO
            
            # Load the original screenshot
            img = cv2.imread(screenshot_path)
            if img is None:
                return None
            
            # Draw bounding boxes and labels on the image
            for element in ui_elements:
                # Get bounding box information
                bounds = element.get('bounds', {})
                x = bounds.get('x', 0)
                y = bounds.get('y', 0)
                width = bounds.get('width', 0)
                height = bounds.get('height', 0)
                
                if width <= 0 or height <= 0:
                    continue
                
                # Choose color based on element type
                element_type = element.get('element_type', 'text')
                if element_type == 'button':
                    color = (0, 255, 0)  # Green for buttons
                elif element_type == 'input':
                    color = (255, 0, 0)  # Blue for inputs
                elif element_type == 'template':
                    color = (0, 255, 255)  # Yellow for templates
                else:
                    color = (0, 0, 255)  # Red for text elements
                
                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness=2)
                
                # Prepare text label
                text = element.get('text', '')
                confidence = element.get('confidence', 0.0)
                label = f"{text}:{confidence:.2f}"
                
                # Truncate label if too long
                if len(label) > 30:
                    label = label[:27] + "..."
                
                font_scale = 0.5
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
                
                # Background rectangle behind text for legibility
                text_bg_tl = (x, max(0, y - text_h - 6))
                text_bg_br = (x + text_w + 6, max(0, y))
                cv2.rectangle(img, text_bg_tl, text_bg_br, (255, 255, 255), thickness=cv2.FILLED)
                
                # Draw text
                text_org = (x + 3, max(12, y - 3))
                cv2.putText(
                    img,
                    label,
                    text_org,
                    font,
                    font_scale,
                    color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            
            # Convert to base64
            # Convert BGR to RGB (OpenCV uses BGR, but we want RGB for display)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Encode to JPEG format for smaller size
            success, buffer = cv2.imencode('.jpg', img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                return None
            
            # Optionally save to disk for debugging/analysis
            from pathlib import Path
            if config.save_vision_debug:
                debug_dir = Path(screenshot_path).resolve().parent.parent / config.ocr_images_dir
                debug_dir.mkdir(parents=True, exist_ok=True)
                base = Path(screenshot_path).name.replace(".png", "_llm.jpg")
                cv2.imwrite(str(debug_dir / base), img_rgb)

            # Convert to base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            log.debug(f"Generated LLM analysis screenshot with {len(ui_elements)} elements (base64: {len(img_base64)} chars)")
            return img_base64
            
        except Exception as e:
            log.warning(f"Failed to generate screenshot with boxes for LLM: {e}")
            return None

    def _convert_prioritized_action_to_dict(
        self,
        prioritized_action: 'PrioritizedAction'
    ) -> dict[str, Any]:
        """Convert PrioritizedAction to action dictionary format.
        
        Args:
            prioritized_action: The prioritized action object
            
        Returns:
            Action dictionary in the expected format
        """
        action_type = prioritized_action.action_type
        
        if action_type.name == 'SET_TEXT':
            # Generate appropriate text for the input field
            dummy_text = self._generate_dummy_text_for_category(prioritized_action.category)

            # Derive coordinates from element bounds for precision tap-to-focus
            bounds = prioritized_action.element.get('bounds', {})
            x_c = bounds.get('x', 0) + (bounds.get('width', 0) // 2)
            y_c = bounds.get('y', 0) + (bounds.get('height', 0) // 2)
            
            return {
                'type': 'input_text',
                'element': prioritized_action.element,
                'text': dummy_text,
                'x': x_c,
                'y': y_c,
                'reasoning': prioritized_action.reasoning,
                'score': prioritized_action.score,
                'llm_confidence': prioritized_action.llm_confidence,
                'vision_confidence': prioritized_action.vision_confidence,
                'exploration_bonus': prioritized_action.exploration_bonus
            }
        
        elif action_type.name == 'TAP_PRIMARY':
            bounds = prioritized_action.element.get('bounds', {})
            x_c = bounds.get('x', 0) + (bounds.get('width', 0) // 2)
            y_c = bounds.get('y', 0) + (bounds.get('height', 0) // 2)

            return {
                'type': 'tap',
                'element': prioritized_action.element,
                'x': x_c,
                'y': y_c,
                'reasoning': prioritized_action.reasoning,
                'score': prioritized_action.score,
                'llm_confidence': prioritized_action.llm_confidence,
                'vision_confidence': prioritized_action.vision_confidence,
                'exploration_bonus': prioritized_action.exploration_bonus
            }
        
        elif action_type.name == 'SCROLL':
            return {
                'type': 'scroll',
                'direction': 'down',  # Default to down scroll
                'reasoning': prioritized_action.reasoning,
                'score': prioritized_action.score,
                'llm_confidence': prioritized_action.llm_confidence,
                'vision_confidence': prioritized_action.vision_confidence,
                'exploration_bonus': prioritized_action.exploration_bonus
            }
        
        elif action_type.name == 'TAP_NAVIGATION':
            bounds = prioritized_action.element.get('bounds', {})
            x_c = bounds.get('x', 0) + (bounds.get('width', 0) // 2)
            y_c = bounds.get('y', 0) + (bounds.get('height', 0) // 2)

            return {
                'type': 'tap',
                'element': prioritized_action.element,
                'x': x_c,
                'y': y_c,
                'reasoning': prioritized_action.reasoning,
                'score': prioritized_action.score,
                'llm_confidence': prioritized_action.llm_confidence,
                'vision_confidence': prioritized_action.vision_confidence,
                'exploration_bonus': prioritized_action.exploration_bonus
            }
        
        else:
            # Fallback
            return {
                'type': 'wait',
                'duration': 1.0,
                'reasoning': f'Unknown action type: {action_type.name}'
            }

    def _generate_dummy_text_for_category(self, category: 'ElementCategory') -> str:
        """Generate appropriate dummy text for a given element category.
        
        Args:
            category: The element category
            
        Returns:
            Appropriate dummy text for the category
        """
        dummy_texts = {
            'email_input': 'test@example.com',
            'password_input': 'password123',
            'search_input': 'pizza',
            'name_input': 'John Doe',
            'address_input': '123 Main St',
            'phone_input': '555-123-4567',
            'date_input': '2025-07-18',
            'url_input': 'https://example.com',
            'code_input': '123456',
            'generic_input': 'test input'
        }
        
        return dummy_texts.get(category.value, 'test input')
    
    async def start_foreground_monitoring(self, target_package: str, check_interval: float = 1.0) -> bool:
        """Start foreground monitoring for the target app.
        
        Args:
            target_package: Package name of the target app to monitor
            check_interval: Interval in seconds between foreground checks
            
        Returns:
            True if monitoring started successfully, False otherwise
        """
        try:
            if self._foreground_monitoring_task and not self._foreground_monitoring_task.done():
                log.warning("Foreground monitoring is already running")
                return True
            
            log.info(f"Starting foreground monitoring for {target_package}")
            self._foreground_monitoring_task = await self.device_manager.start_foreground_monitoring(
                target_package, check_interval
            )
            
            log.success(f"Foreground monitoring started for {target_package}")
            return True
            
        except Exception as e:
            log.error(f"Failed to start foreground monitoring: {e}")
            return False

    async def stop_foreground_monitoring(self) -> None:
        """Stop foreground monitoring if it's running."""
        try:
            if self._foreground_monitoring_task and not self._foreground_monitoring_task.done():
                log.info("Stopping foreground monitoring")
                await self.device_manager.stop_foreground_monitoring(self._foreground_monitoring_task)
                self._foreground_monitoring_task = None
                log.success("Foreground monitoring stopped")
            else:
                log.info("No foreground monitoring task to stop")
                
        except Exception as e:
            log.error(f"Error stopping foreground monitoring: {e}")

    async def ensure_app_foreground(self, target_package: str, max_retries: int = 3) -> bool:
        """Ensure the target app is in foreground with retry logic.
        
        Args:
            target_package: Package name of the target app
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if app is in foreground, False otherwise
        """
        return await self.device_manager.ensure_app_foreground(target_package, max_retries)

    async def start_app_recovery_monitoring(self, target_package: str) -> bool:
        """Start app recovery monitoring for the target package.
        
        Args:
            target_package: Package name of the target app
            
        Returns:
            bool: True if monitoring started successfully
        """
        try:
            success = await self.app_recovery_manager.start_recovery_monitoring(target_package)
            if success:
                log.success(f"App recovery monitoring started for {target_package}")
            else:
                log.error(f"Failed to start app recovery monitoring for {target_package}")
            return success
        except Exception as e:
            log.error(f"Error starting app recovery monitoring: {e}")
            return False

    async def stop_app_recovery_monitoring(self) -> bool:
        """Stop app recovery monitoring.
        
        Returns:
            bool: True if monitoring stopped successfully
        """
        try:
            success = await self.app_recovery_manager.stop_recovery_monitoring()
            if success:
                log.success("App recovery monitoring stopped")
            else:
                log.warning("Failed to stop app recovery monitoring")
            return success
        except Exception as e:
            log.error(f"Error stopping app recovery monitoring: {e}")
            return False

    async def get_app_recovery_stats(self) -> dict[str, Any]:
        """Get app recovery statistics.
        
        Returns:
            dict: Recovery statistics and history
        """
        try:
            recovery_stats = self.app_recovery_manager.get_recovery_stats()
            foreground_recovery_stats = self.app_foreground_recovery_manager.get_recovery_stats()
            
            return {
                "app_recovery": recovery_stats,
                "foreground_recovery": foreground_recovery_stats
            }
        except Exception as e:
            log.error(f"Error getting app recovery stats: {e}")
            return {}

    async def attempt_app_recovery(self) -> bool:
        """Manually attempt app recovery.
        
        Returns:
            bool: True if recovery successful
        """
        try:
            success = await self.app_recovery_manager.attempt_recovery()
            if success:
                log.success("Manual app recovery successful")
            else:
                log.warning("Manual app recovery failed")
            return success
        except Exception as e:
            log.error(f"Error during manual app recovery: {e}")
            return False

    def update_app_recovery_config(self, config: dict[str, Any]) -> None:
        """Update app recovery configuration.
        
        Args:
            config: New recovery configuration
        """
        try:
            from ..core.app_recovery import AppRecoveryConfig
            
            # Convert dict to AppRecoveryConfig
            recovery_config = AppRecoveryConfig(
                max_recovery_attempts=config.get('max_recovery_attempts', 5),
                recovery_timeout=config.get('recovery_timeout', 30.0),
                check_interval=config.get('check_interval', 2.0),
                enable_foreground_service=config.get('enable_foreground_service', True),
                enable_force_restart=config.get('enable_force_restart', True),
                enable_clear_recents=config.get('enable_clear_recents', False)
            )
            
            self.app_recovery_manager.update_config(recovery_config)
            log.info("App recovery configuration updated")
        except Exception as e:
            log.error(f"Error updating app recovery config: {e}")

    @asynccontextmanager
    async def automation_session(
        self, device_serial: str | None = None,
    ) -> AsyncIterator[DroidBotGPT]:
        """Context manager for automation sessions."""
        try:
            # Connect to device
            connected = await self.connect_device(device_serial)
            if not connected:
                raise RuntimeError("Failed to connect to device")
            
            yield self
            
        finally:
            # Cleanup
            await self.disconnect()
    
    def get_session_info(self) -> dict[str, Any]:
        """Get information about the current session."""
        exploration_stats = self.element_tracker.get_exploration_stats()
        
        return {
            "session_id": self.session_id,
            "total_tasks": len(self.task_history),
            "current_task": self.current_task["id"] if self.current_task else None,
            "is_running": self.is_running,
            "connected": self.device_manager.is_connected(),
            "exploration_stats": exploration_stats
        } 