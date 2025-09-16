"""Validation utility functions for DroidBot-GPT framework."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

from ..core.logger import log


def validate_coordinates(x: Union[int, float], y: Union[int, float], 
                        max_width: int = 1080, max_height: int = 1920) -> bool:
    """Validate screen coordinates.
    
    Args:
        x: X coordinate.
        y: Y coordinate.
        max_width: Maximum screen width.
        max_height: Maximum screen height.
        
    Returns:
        True if coordinates are valid, False otherwise.
    """
    try:
        x_val = float(x)
        y_val = float(y)
        
        if x_val < 0 or x_val > max_width:
            log.warning(f"X coordinate {x_val} out of bounds [0, {max_width}]")
            return False
            
        if y_val < 0 or y_val > max_height:
            log.warning(f"Y coordinate {y_val} out of bounds [0, {max_height}]")
            return False
            
        return True
        
    except (ValueError, TypeError):
        log.error(f"Invalid coordinate values: x={x}, y={y}")
        return False


def validate_action(action: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate automation action structure.
    
    Args:
        action: Action dictionary to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not isinstance(action, dict):
        return False, "Action must be a dictionary"
        
    if 'type' not in action:
        return False, "Action missing 'type' field"
        
    action_type = action['type']
    
    # Validate action type
    valid_types = ['tap', 'swipe', 'input_text', 'key_event', 'wait', 'screenshot']
    if action_type not in valid_types:
        return False, f"Invalid action type: {action_type}"
        
    # Type-specific validation
    if action_type == 'tap':
        return _validate_tap_action(action)
    elif action_type == 'swipe':
        return _validate_swipe_action(action)
    elif action_type == 'input_text':
        return _validate_input_action(action)
    elif action_type == 'key_event':
        return _validate_key_action(action)
    elif action_type == 'wait':
        return _validate_wait_action(action)
    elif action_type == 'screenshot':
        return True, ""
        
    return False, f"Unknown action type: {action_type}"


def _validate_tap_action(action: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate tap action parameters."""
    if 'x' not in action or 'y' not in action:
        return False, "Tap action missing x or y coordinates"
        
    x, y = action['x'], action['y']
    
    if not validate_coordinates(x, y):
        return False, "Invalid tap coordinates"
        
    return True, ""


def _validate_swipe_action(action: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate swipe action parameters."""
    required_coords = ['start_x', 'start_y', 'end_x', 'end_y']
    
    for coord in required_coords:
        if coord not in action:
            return False, f"Swipe action missing {coord}"
            
    start_x, start_y = action['start_x'], action['start_y']
    end_x, end_y = action['end_x'], action['end_y']
    
    if not validate_coordinates(start_x, start_y):
        return False, "Invalid swipe start coordinates"
        
    if not validate_coordinates(end_x, end_y):
        return False, "Invalid swipe end coordinates"
        
    # Validate duration if provided
    if 'duration' in action:
        duration = action['duration']
        if not isinstance(duration, (int, float)) or duration <= 0:
            return False, "Swipe duration must be a positive number"
            
    return True, ""


def _validate_input_action(action: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate input text action parameters."""
    if 'text' not in action:
        return False, "Input action missing text"
        
    text = action['text']
    if not isinstance(text, str):
        return False, "Input text must be a string"
        
    if len(text) == 0:
        return False, "Input text cannot be empty"
        
    return True, ""


def _validate_key_action(action: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate key event action parameters."""
    if 'key_code' not in action:
        return False, "Key event action missing key_code"
        
    key_code = action['key_code']
    if not isinstance(key_code, int):
        return False, "Key code must be an integer"
        
    # Common Android key codes
    valid_key_codes = [
        3, 4, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
        106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
        134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
        148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
        162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
        190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
        204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
        218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
        232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
        246, 247, 248, 249, 250, 251, 252, 253, 254, 255
    ]
    
    if key_code not in valid_key_codes:
        log.warning(f"Key code {key_code} not in standard Android key codes")
        
    return True, ""


def _validate_wait_action(action: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate wait action parameters."""
    duration = action.get('duration', 1.0)
    
    if not isinstance(duration, (int, float)):
        return False, "Wait duration must be a number"
        
    if duration < 0:
        return False, "Wait duration cannot be negative"
        
    if duration > 3600:  # Max 1 hour
        return False, "Wait duration cannot exceed 1 hour"
        
    return True, ""


def validate_device_info(device_info: Dict[str, Any]) -> bool:
    """Validate device information structure.
    
    Args:
        device_info: Device information dictionary.
        
    Returns:
        True if device info is valid, False otherwise.
    """
    required_fields = ['serial', 'model', 'android_version', 'screen_size']
    
    for field in required_fields:
        if field not in device_info:
            log.error(f"Device info missing required field: {field}")
            return False
            
    # Validate screen size
    screen_size = device_info['screen_size']
    if not isinstance(screen_size, (list, tuple)) or len(screen_size) != 2:
        log.error("Device screen_size must be a tuple of (width, height)")
        return False
        
    width, height = screen_size
    if not isinstance(width, int) or not isinstance(height, int):
        log.error("Screen dimensions must be integers")
        return False
        
    if width <= 0 or height <= 0:
        log.error("Screen dimensions must be positive")
        return False
        
    return True


def validate_task_description(description: str) -> bool:
    """Validate task description.
    
    Args:
        description: Task description string.
        
    Returns:
        True if description is valid, False otherwise.
    """
    if not isinstance(description, str):
        log.error("Task description must be a string")
        return False
        
    if len(description.strip()) == 0:
        log.error("Task description cannot be empty")
        return False
        
    if len(description) > 1000:
        log.error("Task description too long (max 1000 characters)")
        return False
        
    return True


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe file system usage.
    
    Args:
        filename: Original filename.
        
    Returns:
        Sanitized filename.
    """
    import re
    
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed_file"
        
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
        
    return sanitized 