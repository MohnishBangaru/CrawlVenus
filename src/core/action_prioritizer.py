"""Action Prioritizer for DroidBot-GPT framework.

This module implements a comprehensive action prioritization system that consolidates
order of precedence rules with LLM analysis and vision analysis to make optimal
action decisions for exploration and interaction.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.logger import log
from ..core.popup_classifier import get_popup_classifier, PopupAction
from ..ai.phi_ground import get_phi_ground_generator


class ActionType(Enum):
    """Action types with their priority levels."""
    SET_TEXT = 1      # Highest priority
    TAP_PRIMARY = 2   # High priority
    SCROLL = 3        # Medium priority
    TAP_NAVIGATION = 4  # Lowest priority


class ElementCategory(Enum):
    """Element categories for classification."""
    # Text input fields
    EMAIL_INPUT = "email_input"
    PASSWORD_INPUT = "password_input"
    SEARCH_INPUT = "search_input"
    NAME_INPUT = "name_input"
    ADDRESS_INPUT = "address_input"
    PHONE_INPUT = "phone_input"
    DATE_INPUT = "date_input"
    URL_INPUT = "url_input"
    CODE_INPUT = "code_input"
    GENERIC_INPUT = "generic_input"
    
    # Primary action buttons
    PRIMARY_BUTTON = "primary_button"
    FORWARDING_BUTTON = "forwarding_button"
    
    # Navigation elements
    NAVIGATION_BUTTON = "navigation_button"
    BACK_BUTTON = "back_button"
    DISMISS_BUTTON = "dismiss_button"
    
    # Interactive controls
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    SWITCH = "switch"
    
    # General elements
    GENERAL_BUTTON = "general_button"
    LINK = "link"
    OTHER = "other"


@dataclass
class PrioritizedAction:
    """Represents a prioritized action with scoring."""
    action_type: ActionType
    element: Dict[str, Any]
    score: float
    reasoning: str
    category: ElementCategory
    llm_confidence: float = 0.0
    vision_confidence: float = 0.0
    exploration_bonus: float = 0.0


class ActionPrioritizer:
    """Main action prioritizer that consolidates rules, LLM, and vision analysis."""
    
    def __init__(self):
        """Initialize the action prioritizer."""
        self._phi_ground_generator = None
        
        # Text input field keywords
        self.text_input_keywords = {
            "email": ElementCategory.EMAIL_INPUT,
            "password": ElementCategory.PASSWORD_INPUT,
            "search": ElementCategory.SEARCH_INPUT,
            "name": ElementCategory.NAME_INPUT,
            "address": ElementCategory.ADDRESS_INPUT,
            "phone": ElementCategory.PHONE_INPUT,
            "date": ElementCategory.DATE_INPUT,
            "url": ElementCategory.URL_INPUT,
            "code": ElementCategory.CODE_INPUT,
        }
        
        # Primary action button keywords
        self.primary_action_keywords = [
            "next", "continue", "sign in", "submit", "add to cart", "save", "confirm",
            "proceed", "finish", "complete", "done", "ok", "yes", "accept"
        ]
        
        # Forwarding action keywords
        self.forwarding_keywords = [
            "view details", "learn more", "read more", "see more", "expand",
            "open", "launch", "start", "begin"
        ]
        
        # Navigation keywords
        self.navigation_keywords = [
            "back", "previous", "cancel", "close", "dismiss", "exit",
            "menu", "home", "settings", "profile", "account"
        ]
        
        # Social login providers we want to avoid tapping automatically
        self.social_login_keywords = [
            "google", "facebook", "apple", "twitter", "linkedin", "microsoft", "sign in with google"
        ]

        # Guest / skip-login keywords that should be promoted
        self.guest_keywords = [
            "continue as guest", "guest checkout", "skip login", "continue without signing in",
            "browse as guest"
        ]
        
        # Base scores for different action types
        self.base_scores = {
            ActionType.SET_TEXT: 100.0,
            ActionType.TAP_PRIMARY: 80.0,
            ActionType.SCROLL: 60.0,
            ActionType.TAP_NAVIGATION: 40.0
        }
        
        # Exploration bonus for unexplored elements
        self.exploration_bonus = 20.0
        
        # LLM and vision confidence weights
        self.llm_weight = 0.4
        self.vision_weight = 0.3
        self.rule_weight = 0.3

    def prioritize_actions(
        self,
        ui_elements: List[Dict[str, Any]],
        llm_analysis: Optional[Dict[str, Any]] = None,
        vision_analysis: Optional[Dict[str, Any]] = None,
        task_description: str = "",
        action_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[PrioritizedAction]:
        """Prioritize actions based on order of precedence, LLM, and vision analysis.
        
        Args:
            ui_elements: List of UI elements from vision analysis
            llm_analysis: LLM analysis results
            vision_analysis: Vision analysis results
            task_description: Current task description
            action_history: History of previous actions
            
        Returns:
            List of prioritized actions sorted by score
        """
        if action_history is None:
            action_history = []
        
        prioritized_actions = []
        
        # Step 1: Apply order of precedence rules
        rule_based_actions = self._apply_order_of_precedence(ui_elements, action_history)
        
        # Step 2: Integrate LLM analysis
        if llm_analysis:
            llm_enhanced_actions = self._integrate_llm_analysis(
                rule_based_actions, llm_analysis, task_description
            )
        else:
            llm_enhanced_actions = rule_based_actions
        
        # Step 3: Integrate vision analysis
        if vision_analysis:
            vision_enhanced_actions = self._integrate_vision_analysis(
                llm_enhanced_actions, vision_analysis
            )
        else:
            vision_enhanced_actions = llm_enhanced_actions
        
        # Step 4: Calculate final scores and sort
        for action in vision_enhanced_actions:
            final_score = self._calculate_final_score(action)
            action.score = final_score
            prioritized_actions.append(action)
        
        # Sort by score (highest first)
        prioritized_actions.sort(key=lambda x: x.score, reverse=True)
        
        log.info(f"Prioritized {len(prioritized_actions)} actions")
        if prioritized_actions:
            top_action = prioritized_actions[0]
            log.info(f"Top action: {top_action.action_type.name} - {top_action.reasoning} (Score: {top_action.score:.2f})")
        
        return prioritized_actions

    def _apply_order_of_precedence(
        self,
        ui_elements: List[Dict[str, Any]],
        action_history: Optional[List[Dict[str, Any]]]
    ) -> List[PrioritizedAction]:
        """Apply the order of precedence rules to UI elements.
        
        Args:
            ui_elements: List of UI elements
            action_history: History of previous actions
            
        Returns:
            List of prioritized actions based on rules
        """
        actions = []
        explored_elements = self._get_explored_elements(action_history or [])
        
        # Get popup classifier to filter and classify popups
        popup_classifier = get_popup_classifier()
        
        # Get screen size from first element (assuming all elements have bounds)
        screen_size = (1080, 2340)  # Default fallback
        if ui_elements and 'bounds' in ui_elements[0]:
            bounds = ui_elements[0]['bounds']
            if 'x2' in bounds and 'y2' in bounds:
                screen_size = (bounds['x2'], bounds['y2'])
        
        # Filter popup elements and get dismiss actions
        non_popup_elements, popup_analyses = popup_classifier.filter_popup_elements(
            ui_elements, screen_size
        )
        
        # Add dismiss actions for external popups (highest priority)
        dismiss_actions = popup_classifier.get_dismiss_actions(popup_analyses)
        for dismiss_action in dismiss_actions:
            action = PrioritizedAction(
                action_type=ActionType.TAP_PRIMARY,
                element=dismiss_action,
                score=self.base_scores[ActionType.TAP_PRIMARY] + 50.0,  # Higher priority for dismiss
                reasoning=dismiss_action.get('reasoning', 'Dismiss external popup'),
                category=ElementCategory.DISMISS_BUTTON,
                exploration_bonus=0.0  # No exploration bonus for dismiss actions
            )
            actions.append(action)
        
        # Process internal popups (menu cards) with high priority for decision-making
        internal_popup_elements = []
        for analysis in popup_analyses:
            if analysis.action == PopupAction.HANDLE_THROUGH_DECISION:
                internal_popup_elements.append(analysis.element)
        
        # Process non-popup elements and internal popups through normal precedence
        all_elements_to_process = non_popup_elements + internal_popup_elements
        for element in all_elements_to_process:
            # Skip social login / third-party auth buttons completely
            if any(k in element.get('text', '').lower() for k in self.social_login_keywords):
                continue

            # Check if element has been explored
            is_explored = self._is_element_explored(element, explored_elements)
            exploration_bonus = 0.0 if is_explored else self.exploration_bonus
            
            # Check if this is a menu card (internal popup) and give it higher priority
            is_menu_card = element in internal_popup_elements
            if is_menu_card:
                exploration_bonus += 30.0  # Extra bonus for menu cards
            
            # Step 1: SET_TEXT Action (Highest Priority)
            if self._is_text_input_field(element):
                category = self._classify_text_input_field(element)
                dummy_text = self._generate_dummy_text(category)
                
                action = PrioritizedAction(
                    action_type=ActionType.SET_TEXT,
                    element=element,
                    score=self.base_scores[ActionType.SET_TEXT] + exploration_bonus,
                    reasoning=f"Fill {category.value} field with appropriate data",
                    category=category,
                    exploration_bonus=exploration_bonus
                )
                actions.append(action)
                continue
            
            # Step 2: TAP_PRIMARY Action (High Priority)
            if self._is_primary_action_element(element):
                category = self._classify_primary_action(element)
                # Extra boost for guest continuation actions
                guest_boost = 20.0 if any(gk in element.get('text', '').lower() for gk in self.guest_keywords) else 0.0
                score = self.base_scores[ActionType.TAP_PRIMARY] + exploration_bonus + guest_boost
                
                action = PrioritizedAction(
                    action_type=ActionType.TAP_PRIMARY,
                    element=element,
                    score=score,
                    reasoning=f"Tap primary action: {element.get('text', 'Unknown')}",
                    category=category,
                    exploration_bonus=exploration_bonus
                )
                actions.append(action)
                continue
            
            # Step 3: Interactive Controls
            if self._is_interactive_control(element):
                category = self._classify_interactive_control(element)
                
                action = PrioritizedAction(
                    action_type=ActionType.TAP_PRIMARY,
                    element=element,
                    score=self.base_scores[ActionType.TAP_PRIMARY] + exploration_bonus - 10.0,
                    reasoning=f"Interact with {category.value} to change state",
                    category=category,
                    exploration_bonus=exploration_bonus
                )
                actions.append(action)
                continue
            
            # Step 4: TAP_NAVIGATION Action (Lowest Priority)
            if self._is_navigation_element(element):
                category = self._classify_navigation_element(element)
                
                action = PrioritizedAction(
                    action_type=ActionType.TAP_NAVIGATION,
                    element=element,
                    score=self.base_scores[ActionType.TAP_NAVIGATION] + exploration_bonus,
                    reasoning=f"Navigate using: {element.get('text', 'Unknown')}",
                    category=category,
                    exploration_bonus=exploration_bonus
                )
                actions.append(action)
        
        # ------------------------------------------------------------------
        # Fallback: if no TAP_PRIMARY actions were collected, but colored buttons
        # exist on screen, promote them so we attempt to dismiss pop-ups such
        # as invalid-mobile-number dialogs that use colored buttons without
        # recognizable text.
        # ------------------------------------------------------------------
        has_primary = any(a.action_type == ActionType.TAP_PRIMARY for a in actions)
        if not has_primary:
            for element in all_elements_to_process:
                if element.get("element_type", "").lower() == "colored_button":
                    # skip social login buttons
                    txt_lower = element.get("text", "").lower()
                    if any(sk in txt_lower for sk in self.social_login_keywords):
                        continue
                    bounds = element.get("bounds", {})
                    action = PrioritizedAction(
                        action_type=ActionType.TAP_PRIMARY,
                        element=element,
                        score=self.base_scores[ActionType.TAP_PRIMARY] + 10.0,  # small boost
                        reasoning="Colored button fallback (no other primary buttons found)",
                        category=ElementCategory.PRIMARY_BUTTON,
                        exploration_bonus=self.exploration_bonus if not self._is_element_explored(element, explored_elements) else 0.0,
                        # treat guest buttons as primary even in fallback
                    )
                    actions.append(action)
            if actions:
                log.debug("Added colored_button fallback actions (no primary buttons detected)")

        # Always add a swipe/scroll action as a lower-priority fallback so that the
        # system will attempt scrolling after exhausting higher-priority actions
        # (inputs and button taps). This ensures the desired flow: interact with
        # input elements → tap buttons → try swipe actions.

        scroll_action = PrioritizedAction(
            action_type=ActionType.SCROLL,
            element={"type": "scroll", "direction": "down"},
            score=self.base_scores[ActionType.SCROLL],  # Lower than TAP_PRIMARY but higher than navigation
            reasoning="Swipe/scroll to reveal more content",
            category=ElementCategory.OTHER
        )
        actions.append(scroll_action)
        
        return actions

    def _is_text_input_field(self, element: Dict[str, Any]) -> bool:
        """Check if element is a text input field."""
        text = element.get('text', '').lower()
        element_type = element.get('element_type', '').lower()
        
        # Check for input keywords
        for keyword in self.text_input_keywords:
            if keyword in text or keyword in element_type:
                return True
        
        # Check for generic input types
        if element_type in ['input', 'edittext', 'textfield']:
            return True
        
        return False

    def _classify_text_input_field(self, element: Dict[str, Any]) -> ElementCategory:
        """Classify the type of text input field."""
        text = element.get('text', '').lower()
        element_type = element.get('element_type', '').lower()
        
        for keyword, category in self.text_input_keywords.items():
            if keyword in text or keyword in element_type:
                return category
        
        return ElementCategory.GENERIC_INPUT

    def _generate_dummy_text(self, category: ElementCategory) -> str:
        """Generate contextually appropriate dummy text."""
        dummy_texts = {
            ElementCategory.EMAIL_INPUT: "test@example.com",
            ElementCategory.PASSWORD_INPUT: "password123",
            ElementCategory.SEARCH_INPUT: "pizza",
            ElementCategory.NAME_INPUT: "John Doe",
            ElementCategory.ADDRESS_INPUT: "123 Main St",
            ElementCategory.PHONE_INPUT: "555-123-4567",
            ElementCategory.DATE_INPUT: "2025-07-18",
            ElementCategory.URL_INPUT: "https://example.com",
            ElementCategory.CODE_INPUT: "123456",
            ElementCategory.GENERIC_INPUT: "test input"
        }
        return dummy_texts.get(category, "test")

    def _is_primary_action_element(self, element: Dict[str, Any]) -> bool:
        """Check if element is a primary action button."""
        text = element.get('text', '').lower()
        element_type = element.get('element_type', '').lower()
        
        # Skip social login & third-party auth buttons
        if any(provider in text for provider in self.social_login_keywords):
            return False
        
        # If element offers guest continuation, treat as primary regardless of text input presence
        if any(gk in text for gk in self.guest_keywords):
            return True
        
        # Check for primary action keywords
        for keyword in self.primary_action_keywords:
            if keyword in text:
                return True
        
        # Check for forwarding keywords
        for keyword in self.forwarding_keywords + self.guest_keywords:
            if keyword in text:
                return True
        
        # Check for button types
        if element_type in ['button', 'colored_button']:
            return True
        
        return False

    def _classify_primary_action(self, element: Dict[str, Any]) -> ElementCategory:
        """Classify the type of primary action."""
        text = element.get('text', '').lower()
        
        for keyword in self.primary_action_keywords + self.guest_keywords:
            if keyword in text:
                return ElementCategory.PRIMARY_BUTTON
        
        for keyword in self.forwarding_keywords:
            if keyword in text:
                return ElementCategory.FORWARDING_BUTTON
        
        return ElementCategory.GENERAL_BUTTON

    def _is_interactive_control(self, element: Dict[str, Any]) -> bool:
        """Check if element is an interactive control."""
        element_type = element.get('element_type', '').lower()
        text = element.get('text', '').lower()
        
        # Check for checkbox, radio, switch
        if element_type in ['checkbox', 'radio', 'switch']:
            return True
        
        # Check for unchecked state
        if 'unchecked' in text or 'off' in text:
            return True
        
        return False

    def _classify_interactive_control(self, element: Dict[str, Any]) -> ElementCategory:
        """Classify the type of interactive control."""
        element_type = element.get('element_type', '').lower()
        
        if element_type == 'checkbox':
            return ElementCategory.CHECKBOX
        elif element_type == 'radio':
            return ElementCategory.RADIO_BUTTON
        elif element_type == 'switch':
            return ElementCategory.SWITCH
        
        return ElementCategory.OTHER

    def _is_navigation_element(self, element: Dict[str, Any]) -> bool:
        """Check if element is a navigation element."""
        text = element.get('text', '').lower()
        element_type = element.get('element_type', '').lower()
        
        # Check for navigation keywords
        for keyword in self.navigation_keywords:
            if keyword in text:
                return True
        
        # Check for back/dismiss types
        if element_type in ['back', 'close', 'dismiss']:
            return True
        
        return False

    def _classify_navigation_element(self, element: Dict[str, Any]) -> ElementCategory:
        """Classify the type of navigation element."""
        text = element.get('text', '').lower()
        
        if text in ['back', 'previous', 'cancel', 'close', 'dismiss', 'exit']:
            return ElementCategory.BACK_BUTTON
        elif text in ['menu', 'home', 'settings', 'profile', 'account']:
            return ElementCategory.NAVIGATION_BUTTON
        
        return ElementCategory.GENERAL_BUTTON

    def _is_element_explored(self, element: Dict[str, Any], explored_elements: List[str]) -> bool:
        """Check if element has been explored before."""
        element_id = self._generate_element_id(element)
        return element_id in explored_elements

    def _generate_element_id(self, element: Dict[str, Any]) -> str:
        """Generate a unique ID for an element."""
        text = element.get('text', '')
        element_type = element.get('element_type', '')
        bounds = element.get('bounds', {})
        x = bounds.get('x', 0)
        y = bounds.get('y', 0)
        
        return f"{text}_{element_type}_{x}_{y}"

    def _get_explored_elements(self, action_history: List[Dict[str, Any]]) -> List[str]:
        """Get list of explored element IDs from action history."""
        explored = []
        for action in action_history:
            if 'element' in action:
                element_id = self._generate_element_id(action['element'])
                explored.append(element_id)
        return explored

    def _integrate_llm_analysis(
        self,
        actions: List[PrioritizedAction],
        llm_analysis: Dict[str, Any],
        task_description: str
    ) -> List[PrioritizedAction]:
        """Integrate LLM analysis with rule-based actions."""
        llm_suggestions = llm_analysis.get('suggestions', [])
        llm_confidence = llm_analysis.get('confidence', 0.5)
        
        for action in actions:
            # Find matching LLM suggestion
            matching_suggestion = self._find_matching_llm_suggestion(
                action, llm_suggestions, task_description
            )
            
            if matching_suggestion:
                action.llm_confidence = matching_suggestion.get('confidence', llm_confidence)
                action.reasoning += f" | LLM: {matching_suggestion.get('reasoning', '')}"
            else:
                action.llm_confidence = 0.3  # Lower confidence if no LLM match
        
        return actions

    def _find_matching_llm_suggestion(
        self,
        action: PrioritizedAction,
        llm_suggestions: List[Dict[str, Any]],
        task_description: str
    ) -> Optional[Dict[str, Any]]:
        """Find matching LLM suggestion for an action."""
        element_text = action.element.get('text', '').lower()
        action_type = action.action_type.name.lower()
        
        for suggestion in llm_suggestions:
            suggestion_text = suggestion.get('element_text', '').lower()
            suggestion_type = suggestion.get('action_type', '').lower()
            
            # Check for text similarity
            if element_text and suggestion_text:
                similarity = self._calculate_text_similarity(element_text, suggestion_text)
                if similarity > 0.7:
                    return suggestion
            
            # Check for action type match
            if action_type in suggestion_type or suggestion_type in action_type:
                return suggestion
        
        return None

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def _integrate_vision_analysis(
        self,
        actions: List[PrioritizedAction],
        vision_analysis: Dict[str, Any]
    ) -> List[PrioritizedAction]:
        """Integrate vision analysis with actions."""
        element_confidence = vision_analysis.get('element_confidence', {})
        screen_analysis = vision_analysis.get('screen_analysis', {})
        
        for action in actions:
            element_id = self._generate_element_id(action.element)
            
            # Get vision confidence for this element
            action.vision_confidence = element_confidence.get(element_id, 0.5)
            
            # Adjust score based on screen analysis
            if screen_analysis.get('is_scrollable', False):
                if action.action_type == ActionType.SCROLL:
                    action.score += 10.0
            
            # Adjust based on element visibility
            if action.vision_confidence > 0.8:
                action.score += 5.0
            elif action.vision_confidence < 0.3:
                action.score -= 10.0
        
        return actions

    def _calculate_final_score(self, action: PrioritizedAction) -> float:
        """Calculate final score by consolidating all inputs."""
        # Base score from rules
        base_score = action.score
        
        # Weighted contribution from LLM and vision
        llm_contribution = action.llm_confidence * self.llm_weight * 100
        vision_contribution = action.vision_confidence * self.vision_weight * 100
        
        # Exploration bonus
        exploration_contribution = action.exploration_bonus
        
        # Calculate final score
        final_score = (
            base_score * self.rule_weight +
            llm_contribution +
            vision_contribution +
            exploration_contribution
        )
        
        return final_score

    def _try_phi_ground_action(
        self,
        screenshot_path: str,
        task_description: str,
        action_history: List[Dict[str, Any]],
        ui_elements: List[Dict[str, Any]]
    ) -> Optional[PrioritizedAction]:
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
            
            # Convert UI elements to UIElement objects for Phi Ground
            from ..vision.models import UIElement, BoundingBox
            ui_element_objects = []
            for element in ui_elements:
                bounds = element.get('bounds', {})
                bbox = BoundingBox(
                    left=bounds.get('x', 0),
                    top=bounds.get('y', 0),
                    right=bounds.get('x2', 0),
                    bottom=bounds.get('y2', 0)
                )
                ui_element_objects.append(UIElement(
                    bbox=bbox,
                    text=element.get('text', ''),
                    confidence=element.get('confidence', 0.5),
                    element_type=element.get('element_type', 'text')
                ))
            
            # Generate action using Phi Ground
            import asyncio
            action = asyncio.run(self._phi_ground_generator.generate_touch_action(
                screenshot_path, task_description, action_history, ui_element_objects
            ))
            
            if action:
                # Validate action coordinates
                if not self._phi_ground_generator.validate_action_coordinates(action):
                    log.warning("Phi Ground generated invalid coordinates, falling back to traditional method")
                    return None
                
                # Check confidence threshold
                confidence = action.get("confidence", 0.5)
                if confidence < 0.5:  # Default threshold
                    log.warning(f"Phi Ground confidence too low ({confidence:.2f}), falling back to traditional method")
                    return None
                
                # Convert to PrioritizedAction
                action_type = self._convert_action_type(action.get("type", ""))
                category = self._classify_element_category(action)
                
                prioritized_action = PrioritizedAction(
                    action_type=action_type,
                    element=action,
                    score=confidence * 100,  # Use confidence as base score
                    reasoning=action.get("reasoning", "Phi Ground generated"),
                    category=category,
                    llm_confidence=confidence,
                    vision_confidence=confidence,
                    exploration_bonus=0.0
                )
                
                log.info(f"Phi Ground generated action: {action['type']} with confidence {confidence:.2f}")
                return prioritized_action
            
            return None
            
        except Exception as e:
            log.warning(f"Phi Ground action generation failed: {e}")
            return None
    
    def _convert_action_type(self, action_type: str) -> ActionType:
        """Convert action type string to ActionType enum."""
        if action_type == "text_input":
            return ActionType.SET_TEXT
        elif action_type == "tap":
            return ActionType.TAP_PRIMARY
        elif action_type == "swipe":
            return ActionType.SCROLL
        else:
            return ActionType.TAP_NAVIGATION
    
    def _classify_element_category(self, action: Dict[str, Any]) -> ElementCategory:
        """Classify element category based on action."""
        element_text = action.get("element_text", "").lower()
        
        # Check for input fields
        for keyword, category in self.text_input_keywords.items():
            if keyword in element_text:
                return category
        
        # Check for primary actions
        if any(keyword in element_text for keyword in self.primary_action_keywords):
            return ElementCategory.PRIMARY_BUTTON
        
        # Check for forwarding actions
        if any(keyword in element_text for keyword in self.forwarding_keywords):
            return ElementCategory.FORWARDING_BUTTON
        
        # Default
        return ElementCategory.GENERAL_BUTTON

    def get_optimal_action(
        self,
        ui_elements: List[Dict[str, Any]],
        llm_analysis: Optional[Dict[str, Any]] = None,
        vision_analysis: Optional[Dict[str, Any]] = None,
        task_description: str = "",
        action_history: Optional[List[Dict[str, Any]]] = None,
        screenshot_path: Optional[str] = None
    ) -> Optional[PrioritizedAction]:
        """Get the optimal action based on all inputs.
        
        Args:
            ui_elements: List of UI elements
            llm_analysis: LLM analysis results
            vision_analysis: Vision analysis results
            task_description: Current task description
            action_history: History of previous actions
            screenshot_path: Path to current screenshot for Phi Ground
            
        Returns:
            Optimal action or None if no actions available
        """
        # Try Phi Ground first if enabled and screenshot is available
        if screenshot_path:
            phi_ground_action = self._try_phi_ground_action(
                screenshot_path, task_description, action_history or [], ui_elements
            )
            if phi_ground_action:
                log.info("Using Phi Ground generated action")
                return phi_ground_action
        
        prioritized_actions = self.prioritize_actions(
            ui_elements, llm_analysis, vision_analysis, task_description, action_history
        )
        
        if not prioritized_actions:
            return None
        
        optimal_action = prioritized_actions[0]
        
        log.info(f"Selected optimal action: {optimal_action.action_type.name}")
        log.info(f"Reasoning: {optimal_action.reasoning}")
        log.info(f"Final score: {optimal_action.score:.2f}")
        
        return optimal_action


# Global instance
_action_prioritizer: Optional[ActionPrioritizer] = None


def get_action_prioritizer() -> ActionPrioritizer:
    """Get the global action prioritizer instance.
    
    Returns:
        ActionPrioritizer: The prioritizer instance.
    """
    global _action_prioritizer
    if _action_prioritizer is None:
        _action_prioritizer = ActionPrioritizer()
    return _action_prioritizer 