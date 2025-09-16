"""Task planning and decomposition for Android automation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..core.logger import log


class TaskPlanner:
    """Plans and decomposes automation tasks into executable actions."""
    
    def __init__(self):
        """Initialize the task planner."""
        self.common_patterns = self._load_common_patterns()
        self.action_templates = self._load_action_templates()
        
    def plan_task(self, task_description: str, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan a task by decomposing it into executable actions.
        
        Args:
            task_description: Natural language description of the task.
            current_state: Current device state information.
            
        Returns:
            List of planned actions to execute.
        """
        log.info(f"Planning task: {task_description}")
        
        # Normalize task description
        normalized_task = self._normalize_task_description(task_description)
        
        # Identify task type and extract parameters
        task_type, parameters = self._analyze_task(normalized_task)
        
        # Generate action sequence
        actions = self._generate_action_sequence(task_type, parameters, current_state)
        
        log.info(f"Generated {len(actions)} actions for task")
        return actions
    
    def _normalize_task_description(self, description: str) -> str:
        """Normalize task description for consistent processing.
        
        Args:
            description: Raw task description.
            
        Returns:
            Normalized task description.
        """
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', description.lower().strip())
        
        # Common abbreviations and synonyms
        replacements = {
            'click': 'tap',
            'press': 'tap',
            'touch': 'tap',
            'type': 'input',
            'enter': 'input',
            'scroll': 'swipe',
            'swipe up': 'swipe_up',
            'swipe down': 'swipe_down',
            'swipe left': 'swipe_left',
            'swipe right': 'swipe_right',
            'go back': 'back',
            'return': 'back',
            'home': 'home',
            'menu': 'menu',
            'settings': 'settings',
            'app': 'application'
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
            
        return normalized
    
    def _analyze_task(self, task: str) -> tuple[str, Dict[str, Any]]:
        """Analyze task to identify type and extract parameters.
        
        Args:
            task: Normalized task description.
            
        Returns:
            Tuple of (task_type, parameters).
        """
        parameters = {}
        
        # Navigation tasks
        if any(word in task for word in ['navigate', 'go to', 'open', 'launch']):
            task_type = 'navigation'
            # Extract app/activity name
            app_match = re.search(r'(?:to|open|launch)\s+([a-zA-Z0-9\s]+)', task)
            if app_match:
                parameters['target'] = app_match.group(1).strip()
                
        # Input tasks
        elif any(word in task for word in ['input', 'type', 'enter', 'write']):
            task_type = 'input'
            # Extract text to input
            text_match = re.search(r'(?:input|type|enter|write)\s+["\']([^"\']+)["\']', task)
            if text_match:
                parameters['text'] = text_match.group(1)
                
        # Tap/Click tasks
        elif any(word in task for word in ['tap', 'click', 'press']):
            task_type = 'tap'
            # Extract coordinates or element description
            coord_match = re.search(r'at\s+\((\d+),\s*(\d+)\)', task)
            if coord_match:
                parameters['x'] = int(coord_match.group(1))
                parameters['y'] = int(coord_match.group(2))
            else:
                # Extract element description
                element_match = re.search(r'(?:tap|click|press)\s+([a-zA-Z0-9\s]+)', task)
                if element_match:
                    parameters['element'] = element_match.group(1).strip()
                    
        # Swipe tasks
        elif any(word in task for word in ['swipe', 'scroll']):
            task_type = 'swipe'
            # Extract direction
            if 'up' in task:
                parameters['direction'] = 'up'
            elif 'down' in task:
                parameters['direction'] = 'down'
            elif 'left' in task:
                parameters['direction'] = 'left'
            elif 'right' in task:
                parameters['direction'] = 'right'
                
        # Wait tasks
        elif any(word in task for word in ['wait', 'pause', 'sleep']):
            task_type = 'wait'
            # Extract duration
            duration_match = re.search(r'(\d+)\s*(?:seconds?|s)', task)
            if duration_match:
                parameters['duration'] = float(duration_match.group(1))
            else:
                parameters['duration'] = 1.0
                
        # Screenshot tasks
        elif any(word in task for word in ['screenshot', 'capture', 'photo']):
            task_type = 'screenshot'
            
        # Default to generic task
        else:
            task_type = 'generic'
            parameters['description'] = task
            
        return task_type, parameters
    
    def _generate_action_sequence(self, task_type: str, parameters: Dict[str, Any], 
                                current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate action sequence for a given task type.
        
        Args:
            task_type: Type of task to generate actions for.
            parameters: Task parameters.
            current_state: Current device state.
            
        Returns:
            List of actions to execute.
        """
        actions = []
        
        if task_type == 'navigation':
            actions.extend(self._generate_navigation_actions(parameters))
        elif task_type == 'input':
            actions.extend(self._generate_input_actions(parameters))
        elif task_type == 'tap':
            actions.extend(self._generate_tap_actions(parameters, current_state))
        elif task_type == 'swipe':
            actions.extend(self._generate_swipe_actions(parameters))
        elif task_type == 'wait':
            actions.extend(self._generate_wait_actions(parameters))
        elif task_type == 'screenshot':
            actions.extend(self._generate_screenshot_actions())
        else:
            # Generic task - try to infer actions
            actions.extend(self._generate_generic_actions(parameters))
            
        return actions
    
    def _generate_navigation_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions for navigation tasks."""
        actions = []
        target = parameters.get('target', '')
        
        if target:
            # For now, generate a placeholder action
            # In a full implementation, this would use app detection and launch logic
            actions.append({
                'type': 'wait',
                'duration': 1.0,
                'reasoning': f'Preparing to navigate to {target}'
            })
            
        return actions
    
    def _generate_input_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions for input tasks."""
        actions = []
        text = parameters.get('text', '')
        
        if text:
            actions.append({
                'type': 'input_text',
                'text': text,
                'reasoning': f'Inputting text: {text}'
            })
            
        return actions
    
    def _generate_tap_actions(self, parameters: Dict[str, Any], 
                            current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions for tap tasks."""
        actions = []
        
        # If coordinates are provided, use them directly
        if 'x' in parameters and 'y' in parameters:
            actions.append({
                'type': 'tap',
                'x': parameters['x'],
                'y': parameters['y'],
                'reasoning': f'Tapping at coordinates ({parameters["x"]}, {parameters["y"]})'
            })
        elif 'element' in parameters:
            # For element-based taps, we'd need element detection
            # This is a placeholder for future implementation
            actions.append({
                'type': 'wait',
                'duration': 1.0,
                'reasoning': f'Preparing to tap element: {parameters["element"]}'
            })
            
        return actions
    
    def _generate_swipe_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions for swipe tasks."""
        actions = []
        direction = parameters.get('direction', 'down')
        
        # Default swipe coordinates (center of screen)
        # In a real implementation, these would be calculated based on screen size
        if direction == 'up':
            actions.append({
                'type': 'swipe',
                'start_x': 500,
                'start_y': 1000,
                'end_x': 500,
                'end_y': 200,
                'duration': 500,
                'reasoning': f'Swiping {direction}'
            })
        elif direction == 'down':
            actions.append({
                'type': 'swipe',
                'start_x': 500,
                'start_y': 200,
                'end_x': 500,
                'end_y': 1000,
                'duration': 500,
                'reasoning': f'Swiping {direction}'
            })
        elif direction == 'left':
            actions.append({
                'type': 'swipe',
                'start_x': 800,
                'start_y': 500,
                'end_x': 200,
                'end_y': 500,
                'duration': 500,
                'reasoning': f'Swiping {direction}'
            })
        elif direction == 'right':
            actions.append({
                'type': 'swipe',
                'start_x': 200,
                'start_y': 500,
                'end_x': 800,
                'end_y': 500,
                'duration': 500,
                'reasoning': f'Swiping {direction}'
            })
            
        return actions
    
    def _generate_wait_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions for wait tasks."""
        duration = parameters.get('duration', 1.0)
        
        return [{
            'type': 'wait',
            'duration': duration,
            'reasoning': f'Waiting for {duration} seconds'
        }]
    
    def _generate_screenshot_actions(self) -> List[Dict[str, Any]]:
        """Generate actions for screenshot tasks."""
        return [{
            'type': 'screenshot',
            'reasoning': 'Capturing device screenshot'
        }]
    
    def _generate_generic_actions(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions for generic tasks."""
        description = parameters.get('description', '')
        
        # For generic tasks, we'll add a wait action as a placeholder
        # In a full implementation, this would use AI to determine appropriate actions
        return [{
            'type': 'wait',
            'duration': 2.0,
            'reasoning': f'Processing generic task: {description}'
        }]
    
    def _load_common_patterns(self) -> Dict[str, Any]:
        """Load common task patterns for recognition."""
        return {
            'navigation': ['navigate', 'go to', 'open', 'launch', 'start'],
            'input': ['input', 'type', 'enter', 'write', 'fill'],
            'tap': ['tap', 'click', 'press', 'touch'],
            'swipe': ['swipe', 'scroll', 'slide'],
            'wait': ['wait', 'pause', 'sleep', 'delay'],
            'screenshot': ['screenshot', 'capture', 'photo', 'picture']
        }
    
    def _load_action_templates(self) -> Dict[str, Any]:
        """Load action templates for common operations."""
        return {
            'back': {'type': 'key_event', 'key_code': 4},
            'home': {'type': 'key_event', 'key_code': 3},
            'menu': {'type': 'key_event', 'key_code': 82},
            'enter': {'type': 'key_event', 'key_code': 66}
        } 