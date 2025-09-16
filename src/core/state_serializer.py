"""Enhanced serialize device state into a deterministic compact string for LLM prompts with improved element detection."""

from __future__ import annotations

from typing import Any

from src.vision.models import UIElement  # type: ignore

__all__ = ["serialize_state"]


def _sort_elements_reading_order(e: UIElement) -> tuple[int, int]:
    """Sort helper - top-to-bottom, left-to-right."""
    return e.bbox.top, e.bbox.left


def _classify_element_type(element: UIElement) -> str:
    """Classify element type based on text content and element properties."""
    text_lower = element.text.lower()
    
    # Button detection
    button_keywords = [
        "button", "tap", "click", "press", "select", "choose", "continue", "next",
        "submit", "save", "confirm", "ok", "yes", "no", "cancel", "back", "login",
        "sign in", "register", "sign up", "search", "menu", "settings", "profile",
        "cart", "checkout", "order", "add", "remove", "edit", "delete", "close"
    ]
    
    # Input field detection
    input_keywords = [
        "email", "password", "username", "name", "phone", "address", "search",
        "query", "input", "field", "text", "enter", "type", "fill", "write"
    ]
    
    # Navigation detection
    nav_keywords = [
        "back", "forward", "home", "menu", "settings", "profile", "help",
        "tab", "page", "section", "navigation"
    ]
    
    # Content detection
    content_keywords = [
        "title", "header", "label", "description", "text", "info", "message",
        "status", "loading", "error", "success", "complete"
    ]
    
    # Check element type from vision engine
    if element.element_type in ["button", "colored_button"]:
        return "BUTTON"
    elif element.element_type == "input":
        return "INPUT"
    
    # Classify based on text content
    if any(keyword in text_lower for keyword in button_keywords):
        return "BUTTON"
    elif any(keyword in text_lower for keyword in input_keywords):
        return "INPUT"
    elif any(keyword in text_lower for keyword in nav_keywords):
        return "NAVIGATION"
    elif any(keyword in text_lower for keyword in content_keywords):
        return "CONTENT"
    else:
        return "UNKNOWN"


def _get_element_confidence_level(confidence: float) -> str:
    """Convert confidence score to descriptive level."""
    if confidence >= 0.9:
        return "VERY_HIGH"
    elif confidence >= 0.7:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    elif confidence >= 0.3:
        return "LOW"
    else:
        return "VERY_LOW"


def _analyze_element_relationships(elements: list[UIElement]) -> dict[str, Any]:
    """Analyze spatial relationships between elements."""
    relationships = {
        "form_groups": [],
        "navigation_groups": [],
        "button_groups": [],
        "input_sequences": []
    }
    
    # Group elements by proximity and type
    for i, element in enumerate(elements):
        x1, y1, x2, y2 = element.bbox.as_tuple()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Find nearby elements
        nearby = []
        for j, other in enumerate(elements):
            if i != j:
                ox1, oy1, ox2, oy2 = other.bbox.as_tuple()
                ocenter_x = (ox1 + ox2) / 2
                ocenter_y = (oy1 + oy2) / 2
                
                # Calculate distance
                distance = ((center_x - ocenter_x) ** 2 + (center_y - ocenter_y) ** 2) ** 0.5
                if distance < 100:  # Within 100 pixels
                    nearby.append((j, distance, other))
        
        # Analyze relationships
        if _classify_element_type(element) == "INPUT":
            # Look for labels above or to the left
            for _, _, other in nearby:
                if _classify_element_type(other) == "CONTENT":
                    relationships["form_groups"].append({
                        "input": element.text,
                        "label": other.text,
                        "distance": distance
                    })
        
        elif _classify_element_type(element) == "BUTTON":
            # Look for button groups
            button_group = [element.text]
            for _, _, other in nearby:
                if _classify_element_type(other) == "BUTTON":
                    button_group.append(other.text)
            if len(button_group) > 1:
                relationships["button_groups"].append(button_group)
    
    return relationships


def serialize_state(
    state: dict[str, Any],
    task_description: str,
    action_history: list[str],
    *,
    max_elements: int = 25,
    include_relationships: bool = True,
) -> str:
    """Return deterministic multi-line string describing current device state with enhanced element detection.

    Args:
        state: Dict produced by DroidBotGPT._capture_current_state().
        task_description: Current high-level task.
        action_history: List of string descriptions of previous actions.
        max_elements: Trim UI element list for prompt budget.
        include_relationships: Whether to include element relationship analysis.

    """
    device_info = state.get("device_info")
    if device_info is None:
        raise ValueError("state missing device_info")

    elements: list[UIElement] = state.get("ui_elements", [])  # type: ignore[assignment]
    elements_sorted = sorted(elements, key=_sort_elements_reading_order)[:max_elements]

    lines: list[str] = []
    lines.append(
        f"Device: {device_info.model} - Android {device_info.android_version} "
        f"(API {device_info.api_level})"
    )
    w, h = device_info.screen_size
    lines.append(f"Screen: {w}x{h} px, density {device_info.screen_density}")
    lines.append(f'Task: "{task_description}"')
    lines.append("")
    
    # Enhanced element analysis
    lines.append("UI ELEMENT ANALYSIS")
    lines.append("===================")
    
    # Element relationships
    if include_relationships and elements_sorted:
        relationships = _analyze_element_relationships(elements_sorted)
        if relationships["form_groups"]:
            lines.append("FORM GROUPS:")
            for group in relationships["form_groups"][:3]:  # Limit to 3
                lines.append(f"  - {group['label']} → {group['input']}")
            lines.append("")
        
        if relationships["button_groups"]:
            lines.append("BUTTON GROUPS:")
            for group in relationships["button_groups"][:3]:  # Limit to 3
                lines.append(f"  - {' | '.join(group)}")
            lines.append("")
    
    # Detailed element listing
    lines.append("DETECTED ELEMENTS:")
    lines.append("------------------")
    
    interactive_count = 0
    input_count = 0
    button_count = 0
    
    for idx, el in enumerate(elements_sorted, 1):
        x1, y1, x2, y2 = el.bbox.as_tuple()
        element_type = _classify_element_type(el)
        confidence_level = _get_element_confidence_level(el.confidence)
        
        # Count element types
        if element_type == "INPUT":
            input_count += 1
        elif element_type == "BUTTON":
            button_count += 1
        
        if element_type in ["BUTTON", "INPUT", "NAVIGATION"]:
            interactive_count += 1
        
        element_desc = (
            f"- [{idx}] \"{el.text}\" "
            f"(type={element_type}, vision_type={el.element_type}, "
            f"confidence={confidence_level}, bbox=({x1},{y1},{x2},{y2}))"
        )
        lines.append(element_desc)
    
    # Summary statistics
    lines.append("")
    lines.append("ELEMENT SUMMARY:")
    lines.append("----------------")
    lines.append(f"Total elements: {len(elements_sorted)}")
    lines.append(f"Interactive elements: {interactive_count}")
    lines.append(f"Input fields: {input_count}")
    lines.append(f"Buttons: {button_count}")
    lines.append(f"Content elements: {len(elements_sorted) - interactive_count}")
    
    # Screen context analysis
    lines.append("")
    lines.append("SCREEN CONTEXT:")
    lines.append("---------------")
    
    # Analyze screen type based on element distribution
    if input_count > 2:
        lines.append("Screen type: FORM/INPUT (multiple input fields detected)")
    elif button_count > 3:
        lines.append("Screen type: NAVIGATION/MENU (multiple buttons detected)")
    elif interactive_count == 0:
        lines.append("Screen type: CONTENT/READING (no interactive elements)")
    else:
        lines.append("Screen type: MIXED (combination of elements)")
    
    # Detect potential actions
    if input_count > 0:
        lines.append("Potential actions: Fill forms, enter data")
    if button_count > 0:
        lines.append("Potential actions: Navigate, submit, confirm")
    
    lines.append("")
    lines.append("Action history (last 5)")
    lines.append("-----------------------")
    for act in action_history[-5:]:
        lines.append(f"- {act}")

    return "\n".join(lines) 