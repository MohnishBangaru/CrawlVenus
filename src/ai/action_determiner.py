"""Enhanced intelligent action determination for Explorer automation.

This module provides smart action selection that:
1. Uses improved prompts for better element detection and classification
2. Filters interactions to app elements only (excludes system UI)
3. Uses LLM to generate appropriate text for input fields
4. Handles keyboard triggering for text input
5. Provides context-aware action selection with enhanced analysis
6. Tracks explored elements to prioritize unseen elements
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

from loguru import logger

from ..core.config import config
from ..core.element_tracker import get_element_tracker
from ..core.state_tracker import get_state_tracker
from ..vision.models import UIElement
from .openai_client import get_openai_client
from .prompt_builder import build_element_detection_prompt, build_context_analysis_prompt
from .phi_ground import get_phi_ground_generator
from .ui_venus_ground import get_ui_venus_generator


class EnhancedActionDeterminer:
    """Enhanced intelligent action determination with improved element detection and LLM text generation."""
    
    def __init__(self) -> None:
        """Initialize the enhanced action determiner."""
        self._openai_client = None
        self._phi_ground_generator = None
        self._ui_venus_generator = None
        self._element_tracker = get_element_tracker()
        self._state_tracker = get_state_tracker()
        self._text_input_patterns = [
            r"email", r"password", r"username", r"name", r"phone", r"address",
            r"search", r"query", r"input", r"field", r"text", r"enter", r"type",
            r"fill", r"write", r"add", r"message", r"comment", r"note"
        ]
        self._system_ui_indicators = [
            "status bar", "navigation bar", "home", "back", "recent apps",
            "settings", "notifications", "quick settings", "system ui",
            "battery", "signal", "wifi", "time", "date"
        ]
        self._interactive_keywords = [
            "button", "tap", "click", "press", "select", "choose", "continue", "next",
            "submit", "save", "confirm", "ok", "yes", "no", "cancel", "back",
            "login", "sign in", "register", "sign up", "search", "menu", "settings",
            "profile", "cart", "checkout", "order", "add", "remove", "edit", "delete",
            "close", "exit", "help", "info", "more", "details", "view", "open"
        ]
        
        # Critical precedence rules for action selection
        self._text_input_keywords = [
            "email", "password", "search", "name", "address", "phone", "date", "url", "code"
        ]
        self._primary_action_keywords = [
            "next", "continue", "sign in", "submit", "add to cart", "save", "confirm"
        ]
        self._important_link_keywords = [
            "view details", "learn more", "read more", "see more", "expand"
        ]
        self._interactive_control_keywords = [
            "checkbox", "radio", "switch", "toggle", "select", "choose"
        ]
        self._general_action_keywords = [
            "add", "edit", "share", "download", "upload", "send", "create"
        ]
        self._navigation_keywords = [
            "back", "dismiss", "close", "cancel", "exit", "menu", "tab", "home"
        ]
    
    async def determine_next_action(
        self,
        ui_elements: list[UIElement],
        task_description: str,
        action_history: list[dict[str, Any]],
        device_info: dict[str, Any],
        screenshot_path: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Determine the next action based on current UI state and task with enhanced analysis.
        
        Args:
            ui_elements: List of detected UI elements
            task_description: Current automation task
            action_history: Previous actions performed
            device_info: Device information
            screenshot_path: Path to current screenshot for Phi Ground
            
        Returns:
            Action dictionary or None if no action needed
        """
        try:
            # Try UI-Venus first if enabled and screenshot is available
            if config.use_ui_venus and screenshot_path:
                venus_action = await self._try_ui_venus_action(
                    screenshot_path, task_description, action_history, ui_elements
                )
                if venus_action:
                    logger.info("Using UI-Venus generated action")
                    return venus_action

            # Fallback to Phi Ground if enabled
            if config.use_phi_ground and screenshot_path:
                phi_ground_action = await self._try_phi_ground_action(
                    screenshot_path, task_description, action_history, ui_elements
                )
                if phi_ground_action:
                    logger.info("Using Phi Ground generated action")
                    return phi_ground_action
            
            # Enhanced element analysis using LLM
            element_analysis = await self._analyze_elements_with_llm(
                ui_elements, task_description, action_history
            )
            
            # Filter to app-only elements with enhanced filtering
            app_elements = self._filter_app_elements_enhanced(ui_elements, element_analysis)
            
            if not app_elements:
                logger.warning("No app elements found for interaction")
                return {"type": "wait", "duration": 1.0, "reasoning": "No app elements available"}
            
            # Get exploration context
            exploration_context = self._get_exploration_context(task_description, action_history)
            
            # Check if current state is visited
            is_state_visited = self._state_tracker.is_state_visited(ui_elements, exploration_context)
            state_priority = self._state_tracker.get_state_exploration_priority(ui_elements, exploration_context)
            
            if is_state_visited:
                logger.warning(f"Current state has been visited before (priority: {state_priority:.2f})")
                # Force exploration of new elements in visited state
                logger.info("Forcing exploration of new elements to avoid state loops")
            
            # Prioritize elements based on exploration strategy
            prioritized_elements = self._element_tracker.get_exploration_priority(
                app_elements, exploration_context
            )
            
            # Get unexplored elements for logging
            unexplored_elements = self._element_tracker.get_unexplored_elements(app_elements, exploration_context)
            logger.info(f"Found {len(unexplored_elements)} unexplored elements out of {len(app_elements)} total")
            logger.info(f"State exploration priority: {state_priority:.2f} (visited: {is_state_visited})")
            
            if not prioritized_elements:
                logger.warning("No prioritized elements found")
                return {"type": "wait", "duration": 1.0, "reasoning": "No suitable elements found"}
            
            # Analyze elements for potential actions with exploration and state awareness
            action_candidates = self._analyze_action_candidates_with_exploration(
                prioritized_elements, task_description, element_analysis, exploration_context, state_priority
            )
            
            if not action_candidates:
                logger.warning("No action candidates found")
                return {"type": "wait", "duration": 1.0, "reasoning": "No suitable actions found"}
            
            # Apply critical precedence rules
            action_candidates = self._apply_critical_precedence_rules(
                action_candidates, ui_elements
            )
            
            # Select best action with enhanced reasoning
            best_action = await self._select_best_action_enhanced(
                action_candidates, task_description, action_history, element_analysis
            )
            
            # Mark the selected element as explored
            if best_action and "element_text" in best_action:
                # Find the corresponding element
                for element, _ in prioritized_elements:
                    if element.text == best_action["element_text"]:
                        self._element_tracker.mark_element_explored(
                            element, best_action, exploration_context
                        )
                        break
            
            # Mark current state as visited after action selection
            self._state_tracker.mark_state_visited(ui_elements, exploration_context, best_action)
            
            return best_action
            
        except Exception as e:
            logger.error(f"Enhanced action determination failed: {e}")
            return {"type": "wait", "duration": 1.0, "reasoning": f"Error: {str(e)}"}
    
    def _get_exploration_context(self, task_description: str, action_history: list[dict[str, Any]]) -> str:
        """Generate exploration context based on task and history."""
        # Create context from task description and recent actions
        context_parts = [task_description]
        
        # Add recent actions to context
        recent_actions = action_history[-3:]  # Last 3 actions
        for action in recent_actions:
            action_type = action.get("type", "")
            element_text = action.get("element_text", "")
            if element_text:
                context_parts.append(f"{action_type}:{element_text}")
        
        return "|".join(context_parts)
    
    async def _analyze_elements_with_llm(
        self,
        ui_elements: list[UIElement],
        task_description: str,
        action_history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze elements using LLM for better understanding and classification."""
        try:
            if not config.openai_api_key:
                return {"interactive_elements": [], "suggested_actions": []}
            
            if self._openai_client is None:
                self._openai_client = get_openai_client()
            
            # Create state for LLM analysis
            state = {
                "ui_elements": ui_elements,
                "device_info": {"model": "Android Device", "screen_size": (1080, 1920)}
            }
            
            # Use enhanced prompt for element detection
            messages = build_element_detection_prompt(
                state, task_description, [str(action) for action in action_history[-5:]]
            )
            
            response = await self._openai_client.chat(messages=messages)
            
            # Parse LLM response
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM element analysis response")
                return {"interactive_elements": [], "suggested_actions": []}
                
        except Exception as e:
            logger.warning(f"LLM element analysis failed: {e}")
            return {"interactive_elements": [], "suggested_actions": []}
    
    def _filter_app_elements_enhanced(
        self, 
        elements: list[UIElement], 
        element_analysis: dict[str, Any]
    ) -> list[UIElement]:
        """Enhanced filtering of elements to only include app-specific interactive elements."""
        app_elements = []
        
        # Get LLM-suggested interactive elements if available
        llm_interactive = element_analysis.get("interactive_elements", [])
        llm_element_texts = [item.get("text", "").lower() for item in llm_interactive]
        
        for element in elements:
            # Skip system UI elements
            if self._is_system_ui_enhanced(element):
                continue
            
            # Skip non-interactive elements
            if not self._is_interactive_enhanced(element, llm_element_texts):
                continue
            
            # Skip elements that are too small or too large
            if not self._is_appropriate_size_enhanced(element):
                continue
            
            app_elements.append(element)
        
        logger.debug(f"Enhanced filtering: {len(elements)} → {len(app_elements)} app elements")
        return app_elements
    
    def _is_system_ui_enhanced(self, element: UIElement) -> bool:
        """Enhanced check if element is part of system UI."""
        text_lower = element.text.lower()
        
        # Check for system UI indicators
        for indicator in self._system_ui_indicators:
            if indicator in text_lower:
                return True
        
        # Check for common system UI patterns
        system_patterns = [
            r"^\d{1,2}:\d{2}$",  # Time format
            r"^\d+%$",  # Battery percentage
            r"^[A-Z]{2,3}$",  # Network indicators
            r"^\d{1,2}/\d{1,2}$",  # Date format
            r"^[A-Z]{1,2}\d{1,2}$",  # Signal strength
        ]
        
        for pattern in system_patterns:
            if re.match(pattern, element.text):
                return True
        
        # Check element position (top/bottom edges often contain system UI)
        x1, y1, x2, y2 = element.bbox.as_tuple()
        screen_height = 1920  # Default, should be passed from device_info
        
        if y1 < 50 or y2 > screen_height - 100:  # Top or bottom 100px
            if len(element.text) < 10:  # Short text in edges is likely system UI
                return True
        
        return False
    
    def _is_interactive_enhanced(self, element: UIElement, llm_suggestions: list[str]) -> bool:
        """Enhanced check if element is likely interactive."""
        text_lower = element.text.lower()
        
        # Check if LLM suggested this element as interactive
        if text_lower in llm_suggestions:
            return True
        
        # Check for interactive keywords
        for keyword in self._interactive_keywords:
            if keyword in text_lower:
                return True
        
        # Check element type from vision engine
        if element.element_type in ["button", "colored_button", "input"]:
            return True
        
        # Check for clickable patterns (text with action words)
        action_patterns = [
            r"^\w+\s+\w+$",  # Two word phrases often indicate buttons
            r"^[A-Z][a-z]+$",  # Title case often indicate buttons
            r"^[A-Z][A-Z\s]+$",  # All caps often indicate buttons
        ]
        
        for pattern in action_patterns:
            if re.match(pattern, element.text):
                return True
        
        # Check for input field indicators
        for pattern in self._text_input_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False

    def _apply_critical_precedence_rules(
        self,
        candidates: list[dict[str, Any]],
        ui_elements: list[UIElement]
    ) -> list[dict[str, Any]]:
        """Apply critical precedence rules for action selection.
        
        Order of Precedence:
        1. set_text Action (Highest Priority) - Fill empty/relevant text input fields
        2. tap on Primary/Forwarding Elements (High Priority) - Advance user flow
        3. scroll Action (Medium Priority) - Only if no higher priority actions available
        4. tap on Navigation/Secondary/Dismiss/Back Elements (Lowest Priority) - Exit/navigate
        
        Args:
            candidates: List of action candidates
            ui_elements: All UI elements for scroll detection
            
        Returns:
            Reordered candidates following precedence rules
        """
        if not candidates:
            return candidates
        
        # Separate candidates by type and priority
        text_input_candidates = []
        primary_tap_candidates = []
        general_tap_candidates = []
        navigation_tap_candidates = []
        
        for candidate in candidates:
            if candidate["type"] == "text_input":
                text_input_candidates.append(candidate)
            elif candidate["type"] == "tap":
                element_text = candidate.get("element_text", "").lower()
                
                # Check precedence levels for tap actions
                if any(keyword in element_text for keyword in self._primary_action_keywords):
                    primary_tap_candidates.append(candidate)
                elif any(keyword in element_text for keyword in self._important_link_keywords):
                    primary_tap_candidates.append(candidate)
                elif any(keyword in element_text for keyword in self._interactive_control_keywords):
                    primary_tap_candidates.append(candidate)
                elif any(keyword in element_text for keyword in self._general_action_keywords):
                    general_tap_candidates.append(candidate)
                elif any(keyword in element_text for keyword in self._navigation_keywords):
                    navigation_tap_candidates.append(candidate)
                else:
                    # Default to general tap for unknown elements
                    general_tap_candidates.append(candidate)
        
        # Apply precedence rules
        reordered_candidates = []
        
        # 1. Text input actions (Highest Priority)
        if text_input_candidates:
            # Sort by exploration priority within text input
            text_input_candidates.sort(key=lambda x: x.get("exploration_priority", 0), reverse=True)
            reordered_candidates.extend(text_input_candidates)
            logger.info(f"CRITICAL RULE: Prioritizing {len(text_input_candidates)} text input actions")
        
        # 2. Primary/Forwarding tap actions (High Priority)
        if primary_tap_candidates:
            # Sort by exploration priority within primary actions
            primary_tap_candidates.sort(key=lambda x: x.get("exploration_priority", 0), reverse=True)
            reordered_candidates.extend(primary_tap_candidates)
            logger.info(f"CRITICAL RULE: Prioritizing {len(primary_tap_candidates)} primary tap actions")
        
        # 3. General tap actions (Medium Priority)
        if general_tap_candidates:
            # Sort by exploration priority within general actions
            general_tap_candidates.sort(key=lambda x: x.get("exploration_priority", 0), reverse=True)
            reordered_candidates.extend(general_tap_candidates)
            logger.info(f"CRITICAL RULE: Adding {len(general_tap_candidates)} general tap actions")
        
        # 4. Navigation/Secondary tap actions (Lowest Priority)
        if navigation_tap_candidates:
            # Sort by exploration priority within navigation actions
            navigation_tap_candidates.sort(key=lambda x: x.get("exploration_priority", 0), reverse=True)
            reordered_candidates.extend(navigation_tap_candidates)
            logger.info(f"CRITICAL RULE: Adding {len(navigation_tap_candidates)} navigation tap actions")
        
        # 5. Scroll action (Only if no other actions available)
        if not reordered_candidates and self._should_add_scroll_action(ui_elements):
            scroll_candidate = self._create_scroll_candidate(ui_elements)
            if scroll_candidate:
                reordered_candidates.append(scroll_candidate)
                logger.info("CRITICAL RULE: Adding scroll action as last resort")
        
        return reordered_candidates

    def _should_add_scroll_action(self, ui_elements: list[UIElement]) -> bool:
        """Determine if a scroll action should be added."""
        # Check if screen appears scrollable (has elements near edges)
        screen_height = 1920  # Default, should be dynamic
        screen_width = 1080   # Default, should be dynamic
        
        elements_near_bottom = 0
        elements_near_right = 0
        
        for element in ui_elements:
            x1, y1, x2, y2 = element.bbox.as_tuple()
            
            # Check if element is near bottom edge (indicating more content below)
            if y2 > screen_height * 0.8:
                elements_near_bottom += 1
            
            # Check if element is near right edge (indicating more content to the right)
            if x2 > screen_width * 0.8:
                elements_near_right += 1
        
        # Add scroll if we have elements near edges
        return elements_near_bottom > 2 or elements_near_right > 2

    def _create_scroll_candidate(self, ui_elements: list[UIElement]) -> dict[str, Any]:
        """Create a scroll action candidate."""
        # Determine scroll direction based on element positions
        screen_height = 1920  # Default, should be dynamic
        screen_width = 1080   # Default, should be dynamic
        
        elements_near_bottom = 0
        elements_near_right = 0
        
        for element in ui_elements:
            x1, y1, x2, y2 = element.bbox.as_tuple()
            if y2 > screen_height * 0.8:
                elements_near_bottom += 1
            if x2 > screen_width * 0.8:
                elements_near_right += 1
        
        # Prioritize vertical scroll over horizontal
        if elements_near_bottom > elements_near_right:
            return {
                "type": "scroll",
                "scroll_direction": "down",
                "reasoning": "CRITICAL RULE: Scrolling down to reveal more content",
                "priority": 1.0,
                "exploration_priority": 1.0,
                "is_explored": False
            }
        else:
            return {
                "type": "scroll",
                "scroll_direction": "right",
                "reasoning": "CRITICAL RULE: Scrolling right to reveal more content",
                "priority": 1.0,
                "exploration_priority": 1.0,
                "is_explored": False
            }
    
    def _is_appropriate_size_enhanced(self, element: UIElement) -> bool:
        """Enhanced check if element has appropriate size for interaction."""
        x1, y1, x2, y2 = element.bbox.as_tuple()
        width = x2 - x1
        height = y2 - y1
        
        # Minimum size for reliable tapping
        min_size = 25
        # Maximum size (avoid tapping entire screen areas)
        max_size = 400
        
        # Check aspect ratio (avoid very long/thin elements)
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            return False
        
        return min_size <= width <= max_size and min_size <= height <= max_size
    
    def _analyze_action_candidates_with_exploration(
        self, 
        prioritized_elements: list[tuple[UIElement, float]], 
        task_description: str,
        element_analysis: dict[str, Any],
        exploration_context: str,
        state_priority: float
    ) -> list[dict[str, Any]]:
        """Enhanced analysis of elements to generate action candidates with exploration awareness."""
        candidates = []
        
        # Get LLM suggestions if available
        llm_suggestions = element_analysis.get("suggested_actions", [])
        
        for element, exploration_priority in prioritized_elements:
            # Check if element is unexplored
            is_explored = self._element_tracker.is_element_explored(element, exploration_context)
            
            # Check for text input fields with enhanced detection
            if self._is_text_input_field_enhanced(element):
                # Higher priority for unexplored input fields
                base_priority = 15 if not is_explored else 10
                # Add state priority bonus for new states
                state_bonus = state_priority * 5 if state_priority > 0.5 else 0
                candidates.append({
                    "type": "text_input",
                    "element": element,
                    "priority": base_priority + exploration_priority + state_bonus,
                    "reasoning": f"Text input field detected: {element.text} (explored: {is_explored}, state_priority: {state_priority:.2f})",
                    "llm_suggested": element.text.lower() in [s.get("text", "").lower() for s in llm_suggestions],
                    "exploration_priority": exploration_priority,
                    "state_priority": state_priority,
                    "is_explored": is_explored
                })
            
            # Check for clickable elements with enhanced detection
            elif self._is_clickable_enhanced(element):
                # Higher priority for unexplored elements
                base_priority = 10 if not is_explored else 5
                if element.text.lower() in [s.get("text", "").lower() for s in llm_suggestions]:
                    base_priority += 2
                
                # Add state priority bonus for new states
                state_bonus = state_priority * 3 if state_priority > 0.5 else 0
                
                candidates.append({
                    "type": "tap",
                    "element": element,
                    "priority": base_priority + exploration_priority + state_bonus,
                    "reasoning": f"Clickable element: {element.text} (explored: {is_explored}, state_priority: {state_priority:.2f})",
                    "llm_suggested": element.text.lower() in [s.get("text", "").lower() for s in llm_suggestions],
                    "exploration_priority": exploration_priority,
                    "state_priority": state_priority,
                    "is_explored": is_explored
                })
        
        # Sort by priority (highest first)
        candidates.sort(key=lambda x: x["priority"], reverse=True)
        
        # Log exploration statistics
        unexplored_candidates = [c for c in candidates if not c["is_explored"]]
        logger.info(f"Action candidates: {len(unexplored_candidates)} unexplored, {len(candidates) - len(unexplored_candidates)} explored")
        
        return candidates
    
    def _is_text_input_field_enhanced(self, element: UIElement) -> bool:
        """Enhanced check if element is a text input field."""
        text_lower = element.text.lower()
        
        # Check for input field indicators
        for pattern in self._text_input_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for placeholder text patterns
        placeholder_patterns = [
            r"enter", r"type", r"input", r"fill", r"write", r"add", r"search",
            r"email", r"password", r"username", r"name", r"phone", r"address"
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check element type from vision engine
        if element.element_type == "input":
            return True
        
        return False
    
    def _is_clickable_enhanced(self, element: UIElement) -> bool:
        """Enhanced check if element is clickable."""
        text_lower = element.text.lower()
        
        # Check for clickable keywords
        for keyword in self._interactive_keywords:
            if keyword in text_lower:
                return True
        
        # Check element type from vision engine
        if element.element_type in ["button", "colored_button"]:
            return True
        
        return False
    
    async def _select_best_action_enhanced(
        self,
        candidates: list[dict[str, Any]],
        task_description: str,
        action_history: list[dict[str, Any]],
        element_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Enhanced selection of the best action from candidates using LLM analysis."""
        if not candidates:
            return {"type": "wait", "duration": 1.0, "reasoning": "No candidates available"}
        
        # For text input, use LLM to generate appropriate text
        if candidates[0]["type"] == "text_input":
            return await self._handle_text_input_action_enhanced(
                candidates[0], task_description, element_analysis
            )
        
        # For tap actions, select the highest priority candidate
        best_candidate = candidates[0]
        element = best_candidate["element"]
        x1, y1, x2, y2 = element.bbox.as_tuple()
        
        # Add exploration information to action
        action = {
            "type": "tap",
            "x": (x1 + x2) // 2,
            "y": (y1 + y2) // 2,
            "element_text": element.text,
            "reasoning": best_candidate["reasoning"],
            "llm_suggested": best_candidate.get("llm_suggested", False),
            "exploration_priority": best_candidate.get("exploration_priority", 0.0),
            "is_explored": best_candidate.get("is_explored", False)
        }
        
        # Log exploration decision
        if not best_candidate.get("is_explored", False):
            logger.info(f"Selected unexplored element: {element.text} (priority: {best_candidate['priority']:.2f})")
        else:
            logger.info(f"Selected explored element: {element.text} (priority: {best_candidate['priority']:.2f})")
        
        return action
    
    async def _handle_text_input_action_enhanced(
        self,
        candidate: dict[str, Any],
        task_description: str,
        element_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Enhanced handling of text input action with LLM-generated text."""
        element = candidate["element"]
        x1, y1, x2, y2 = element.bbox.as_tuple()
        
        # Generate appropriate text using LLM with enhanced context
        generated_text = await self._generate_input_text_enhanced(
            element.text, task_description, element_analysis
        )
        
        action = {
            "type": "text_input",
            "x": (x1 + x2) // 2,
            "y": (y1 + y2) // 2,
            "text": generated_text,
            "field_hint": element.text,
            "reasoning": f"Generated text for {element.text}: {generated_text}",
            "llm_suggested": candidate.get("llm_suggested", False),
            "exploration_priority": candidate.get("exploration_priority", 0.0),
            "is_explored": candidate.get("is_explored", False)
        }
        
        # Log exploration decision
        if not candidate.get("is_explored", False):
            logger.info(f"Selected unexplored input field: {element.text} (priority: {candidate['priority']:.2f})")
        else:
            logger.info(f"Selected explored input field: {element.text} (priority: {candidate['priority']:.2f})")
        
        return action
    
    async def _generate_input_text_enhanced(
        self, 
        field_hint: str, 
        task_description: str,
        element_analysis: dict[str, Any]
    ) -> str:
        """Enhanced generation of appropriate text for input field using LLM."""
        try:
            if not config.openai_api_key:
                return self._generate_fallback_text_enhanced(field_hint)
            
            if self._openai_client is None:
                self._openai_client = get_openai_client()
            
            # Enhanced prompt with context analysis
            context_insights = element_analysis.get("context_insights", {})
            screen_type = context_insights.get("screen_type", "unknown")
            
            prompt = f"""
            Generate appropriate text for an Android app input field.
            
            Field hint: "{field_hint}"
            Task context: "{task_description}"
            Screen type: "{screen_type}"
            
            Generate realistic, contextually appropriate text that a user would enter.
            Consider the app context and screen type when generating text.
            Return only the text, no explanations.
            
            Examples:
            - Email field: "user@example.com"
            - Name field: "John Doe"
            - Search field: "pizza delivery"
            - Phone field: "555-123-4567"
            - Address field: "123 Main St, City, State"
            """
            
            response = await self._openai_client.chat(messages=[
                {"role": "system", "content": "You are a helpful assistant that generates realistic text for mobile app input fields based on context."},
                {"role": "user", "content": prompt}
            ])
            
            # Clean up response
            text = response.strip().strip('"').strip("'")
            if len(text) > 50:  # Limit length
                text = text[:50]
            
            return text if text else self._generate_fallback_text_enhanced(field_hint)
            
        except Exception as e:
            logger.warning(f"Enhanced LLM text generation failed: {e}")
            return self._generate_fallback_text_enhanced(field_hint)
    
    def _generate_fallback_text_enhanced(self, field_hint: str) -> str:
        """Enhanced fallback text generation when LLM is not available."""
        field_lower = field_hint.lower()
        
        # Enhanced fallback logic
        if "email" in field_lower:
            return "test@example.com"
        elif "password" in field_lower:
            return "password123"
        elif "name" in field_lower:
            return "Test User"
        elif "phone" in field_lower:
            return "555-123-4567"
        elif "search" in field_lower:
            return "test search"
        elif "address" in field_lower:
            return "123 Test St"
        elif "username" in field_lower:
            return "testuser"
        elif "message" in field_lower or "comment" in field_lower:
            return "Test message"
        else:
            return "test input"
    
    def get_exploration_stats(self) -> dict[str, Any]:
        """Get exploration statistics."""
        return self._element_tracker.get_exploration_stats()
    
    def get_state_stats(self) -> dict[str, Any]:
        """Get state exploration statistics."""
        return self._state_tracker.get_state_exploration_stats()
    
    def reset_exploration(self) -> None:
        """Reset exploration tracking."""
        self._element_tracker.reset_exploration()
    
    def reset_state_tracking(self) -> None:
        """Reset state tracking."""
        self._state_tracker.reset_state_tracking()
    
    def set_exploration_strategy(self, strategy: str) -> None:
        """Set exploration strategy."""
        self._element_tracker.set_exploration_strategy(strategy)
    
    def set_state_similarity_threshold(self, threshold: float) -> None:
        """Set state similarity threshold."""
        self._state_tracker.set_similarity_threshold(threshold)
    
    async def _try_phi_ground_action(
        self,
        screenshot_path: str,
        task_description: str,
        action_history: list[dict[str, Any]],
        ui_elements: list[UIElement]
    ) -> Optional[dict[str, Any]]:
        """Try to generate action using Phi Ground.
        
        Args:
            screenshot_path: Path to the screenshot
            task_description: Current automation task
            action_history: Previous actions performed
            ui_elements: Detected UI elements for validation
            
        Returns:
            Phi Ground generated action or None
        """
        try:
            if self._phi_ground_generator is None:
                self._phi_ground_generator = get_phi_ground_generator()
            
            # Generate action using Phi Ground
            action = await self._phi_ground_generator.generate_touch_action(
                screenshot_path, task_description, action_history, ui_elements
            )
            
            if action:
                # Validate action coordinates
                if not self._phi_ground_generator.validate_action_coordinates(action):
                    logger.warning("Phi Ground generated invalid coordinates, falling back to traditional method")
                    return None
                
                # Check confidence threshold
                confidence = action.get("confidence", 0.5)
                if confidence < config.phi_ground_confidence_threshold:
                    logger.warning(f"Phi Ground confidence too low ({confidence:.2f}), falling back to traditional method")
                    return None
                
                logger.info(f"Phi Ground generated action: {action['type']} with confidence {confidence:.2f}")
                return action
            
            return None
            
        except Exception as e:
            logger.warning(f"Phi Ground action generation failed: {e}")
            return None

    async def _try_ui_venus_action(
        self,
        screenshot_path: str,
        task_description: str,
        action_history: list[dict[str, Any]],
        ui_elements: list[UIElement],
    ) -> Optional[dict[str, Any]]:
        """Generate action using UI-Venus model.

        Mirrors the logic of ``_try_phi_ground_action`` for seamless fallback handling.
        """
        try:
            if self._ui_venus_generator is None:
                self._ui_venus_generator = get_ui_venus_generator()

            action = await self._ui_venus_generator.generate_touch_action(
                screenshot_path, task_description, action_history, ui_elements
            )

            if action and not self._ui_venus_generator.validate_action_coordinates(action):
                logger.warning("UI-Venus generated invalid coordinates, falling back")
                return None

            confidence = action.get("confidence", 0.5) if action else 0
            if action and confidence < config.ui_venus_confidence_threshold:
                logger.warning(
                    f"UI-Venus confidence too low ({confidence:.2f}), falling back"
                )
                return None

            return action
        except Exception as exc:
            logger.warning(f"UI-Venus action generation failed: {exc}")
            return None


# Global instance for reuse
_enhanced_action_determiner = None


def get_enhanced_action_determiner() -> EnhancedActionDeterminer:
    """Get or create the global enhanced action determiner instance."""
    global _enhanced_action_determiner
    if _enhanced_action_determiner is None:
        _enhanced_action_determiner = EnhancedActionDeterminer()
    return _enhanced_action_determiner


# Backward compatibility
class ActionDeterminer(EnhancedActionDeterminer):
    """Backward compatibility wrapper for the original ActionDeterminer."""
    pass


def get_action_determiner() -> EnhancedActionDeterminer:
    """Get or create the global action determiner instance (backward compatibility)."""
    return get_enhanced_action_determiner() 