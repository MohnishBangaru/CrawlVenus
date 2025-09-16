"""Popup classification system for DroidBot-GPT framework.

This module provides intelligent popup detection and classification to differentiate
between internal app popups (menu cards, navigation elements) that should be handled
through the decision-making process, and external system popups that should be dismissed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.logger import log


class PopupType(Enum):
    """Types of popups that can be detected."""
    INTERNAL_MENU = "internal_menu"
    INTERNAL_NAVIGATION = "internal_navigation"
    INTERNAL_SELECTION = "internal_selection"
    EXTERNAL_SYSTEM = "external_system"
    EXTERNAL_PERMISSION = "external_permission"
    EXTERNAL_NOTIFICATION = "external_notification"
    UNKNOWN = "unknown"


class PopupAction(Enum):
    """Actions to take for different popup types."""
    HANDLE_THROUGH_DECISION = "handle_through_decision"
    DISMISS = "dismiss"
    IGNORE = "ignore"


@dataclass
class PopupAnalysis:
    """Analysis result for a popup element."""
    element: Dict[str, Any]
    popup_type: PopupType
    action: PopupAction
    confidence: float
    reasoning: str
    is_menu_card: bool = False
    is_navigation_element: bool = False
    is_system_popup: bool = False


class PopupClassifier:
    """Intelligent popup classifier that differentiates internal vs external popups."""
    
    def __init__(self):
        """Initialize the popup classifier."""
        # Internal popup indicators (menu cards, navigation)
        self.internal_indicators = {
            "menu": ["menu", "options", "more", "settings", "profile", "account"],
            "navigation": ["back", "forward", "home", "tab", "page", "section"],
            "selection": ["select", "choose", "pick", "option", "item", "card"],
            "app_specific": ["order", "cart", "checkout", "payment", "delivery", "tracking"]
        }
        
        # External popup indicators (system dialogs, permissions)
        self.external_indicators = {
            "system": ["google play", "android", "system", "settings", "permission"],
            "permission": ["allow", "deny", "permission", "access", "grant"],
            "notification": ["notification", "alert", "warning", "error", "update"],
            "external_app": ["chrome", "browser", "gmail", "maps", "play store"]
        }
        
        # Menu card specific patterns
        self.menu_card_patterns = [
            r"menu.*card",
            r"card.*menu", 
            r"option.*card",
            r"item.*card",
            r"selection.*card",
            r"choice.*card"
        ]
        
        # Navigation element patterns
        self.navigation_patterns = [
            r"back.*button",
            r"forward.*button", 
            r"home.*button",
            r"tab.*bar",
            r"navigation.*menu",
            r"breadcrumb"
        ]
        
        # System popup patterns
        self.system_popup_patterns = [
            r"google.*play",
            r"android.*system",
            r"permission.*dialog",
            r"system.*alert",
            r"external.*app"
        ]
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        
        # Element position analysis
        self.screen_center_threshold = 0.3  # 30% from center
        self.overlay_threshold = 0.8  # 80% overlay indicates popup

    def classify_popup(
        self, 
        element: Dict[str, Any], 
        ui_elements: List[Dict[str, Any]],
        screen_size: Tuple[int, int],
        app_context: str = ""
    ) -> PopupAnalysis:
        """Classify a popup element as internal or external.
        
        Args:
            element: The popup element to classify
            ui_elements: All UI elements on the screen
            screen_size: Screen dimensions (width, height)
            app_context: Current app context/package name
            
        Returns:
            PopupAnalysis with classification and recommended action
        """
        text = element.get('text', '').lower()
        element_type = element.get('element_type', '').lower()
        bounds = element.get('bounds', {})
        
        # Step 1: Check if it's a menu card
        is_menu_card = self._is_menu_card(text, element_type)
        
        # Step 2: Check if it's a navigation element
        is_navigation = self._is_navigation_element(text, element_type)
        
        # Step 3: Check if it's a system popup
        is_system_popup = self._is_system_popup(text, element_type, app_context)
        
        # Step 4: Analyze position and overlay characteristics
        position_analysis = self._analyze_position(bounds, screen_size, ui_elements)
        
        # Step 5: Determine popup type and action
        popup_type, action, confidence, reasoning = self._determine_type_and_action(
            text, element_type, is_menu_card, is_navigation, is_system_popup, 
            position_analysis, app_context
        )
        
        return PopupAnalysis(
            element=element,
            popup_type=popup_type,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            is_menu_card=is_menu_card,
            is_navigation_element=is_navigation,
            is_system_popup=is_system_popup
        )

    def _is_menu_card(self, text: str, element_type: str) -> bool:
        """Check if element is a menu card."""
        # Check for menu card patterns
        for pattern in self.menu_card_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for menu-related keywords
        menu_keywords = self.internal_indicators["menu"] + self.internal_indicators["selection"]
        for keyword in menu_keywords:
            if keyword in text:
                return True
        
        # Check element type
        if element_type in ['card', 'menu_item', 'option']:
            return True
        
        return False

    def _is_navigation_element(self, text: str, element_type: str) -> bool:
        """Check if element is a navigation element."""
        # Check for navigation patterns
        for pattern in self.navigation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for navigation keywords
        nav_keywords = self.internal_indicators["navigation"]
        for keyword in nav_keywords:
            if keyword in text:
                return True
        
        # Check element type
        if element_type in ['back_button', 'navigation', 'tab']:
            return True
        
        return False

    def _is_system_popup(self, text: str, element_type: str, app_context: str) -> bool:
        """Check if element is a system popup."""
        # Check for system popup patterns
        for pattern in self.system_popup_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for external indicators
        for category, keywords in self.external_indicators.items():
            for keyword in keywords:
                if keyword in text:
                    return True
        
        # Check if it's from a different app context
        if app_context and element_type == 'external_dialog':
            return True
        
        return False

    def _analyze_position(
        self, 
        bounds: Dict[str, Any], 
        screen_size: Tuple[int, int],
        ui_elements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze element position and overlay characteristics."""
        screen_width, screen_height = screen_size
        
        # Get element bounds
        x1 = bounds.get('x', 0)
        y1 = bounds.get('y', 0)
        x2 = bounds.get('x2', x1)
        y2 = bounds.get('y2', y1)
        
        # Calculate center position
        element_center_x = (x1 + x2) / 2
        element_center_y = (y1 + y2) / 2
        screen_center_x = screen_width / 2
        screen_center_y = screen_height / 2
        
        # Check if element is centered (indicates popup)
        center_distance = ((element_center_x - screen_center_x) ** 2 + 
                          (element_center_y - screen_center_y) ** 2) ** 0.5
        max_center_distance = min(screen_width, screen_height) * self.screen_center_threshold
        is_centered = center_distance < max_center_distance
        
        # Check overlay with other elements
        overlay_count = 0
        for other_element in ui_elements:
            if other_element != bounds:
                other_bounds = other_element.get('bounds', {})
                if self._elements_overlap(bounds, other_bounds):
                    overlay_count += 1
        
        overlay_ratio = overlay_count / len(ui_elements) if ui_elements else 0
        is_overlay = overlay_ratio > self.overlay_threshold
        
        return {
            'is_centered': is_centered,
            'is_overlay': is_overlay,
            'overlay_ratio': overlay_ratio,
            'center_distance': center_distance
        }

    def _elements_overlap(self, bounds1: Dict[str, Any], bounds2: Dict[str, Any]) -> bool:
        """Check if two elements overlap."""
        x1_1, y1_1 = bounds1.get('x', 0), bounds1.get('y', 0)
        x2_1, y2_1 = bounds1.get('x2', x1_1), bounds1.get('y2', y1_1)
        x1_2, y1_2 = bounds2.get('x', 0), bounds2.get('y', 0)
        x2_2, y2_2 = bounds2.get('x2', x1_2), bounds2.get('y2', y1_2)
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

    def _determine_type_and_action(
        self,
        text: str,
        element_type: str,
        is_menu_card: bool,
        is_navigation: bool,
        is_system_popup: bool,
        position_analysis: Dict[str, Any],
        app_context: str
    ) -> Tuple[PopupType, PopupAction, float, str]:
        """Determine popup type and recommended action."""
        confidence = 0.0
        reasoning = ""
        
        # High confidence cases
        if is_menu_card:
            return (
                PopupType.INTERNAL_MENU,
                PopupAction.HANDLE_THROUGH_DECISION,
                0.9,
                "Menu card detected - should be handled through decision-making process"
            )
        
        if is_navigation:
            return (
                PopupType.INTERNAL_NAVIGATION,
                PopupAction.HANDLE_THROUGH_DECISION,
                0.85,
                "Navigation element detected - should be handled through decision-making process"
            )
        
        if is_system_popup:
            return (
                PopupType.EXTERNAL_SYSTEM,
                PopupAction.DISMISS,
                0.9,
                "System popup detected - should be dismissed"
            )
        
        # Medium confidence cases based on position
        if position_analysis['is_centered'] and position_analysis['is_overlay']:
            # Centered overlay suggests popup, but need more context
            if any(keyword in text for keyword in self.internal_indicators["app_specific"]):
                return (
                    PopupType.INTERNAL_SELECTION,
                    PopupAction.HANDLE_THROUGH_DECISION,
                    0.7,
                    "App-specific popup detected - likely internal selection"
                )
            else:
                return (
                    PopupType.EXTERNAL_SYSTEM,
                    PopupAction.DISMISS,
                    0.7,
                    "Centered overlay popup detected - likely external system dialog"
                )
        
        # Low confidence cases
        if position_analysis['is_centered']:
            return (
                PopupType.UNKNOWN,
                PopupAction.HANDLE_THROUGH_DECISION,
                0.5,
                "Centered element - default to decision-making process"
            )
        
        # Default case
        return (
            PopupType.UNKNOWN,
            PopupAction.IGNORE,
            0.3,
            "Unknown element type - ignoring"
        )

    def filter_popup_elements(
        self,
        ui_elements: List[Dict[str, Any]],
        screen_size: Tuple[int, int],
        app_context: str = ""
    ) -> Tuple[List[Dict[str, Any]], List[PopupAnalysis]]:
        """Filter UI elements and classify popups.
        
        Args:
            ui_elements: All UI elements on screen
            screen_size: Screen dimensions
            app_context: Current app context
            
        Returns:
            Tuple of (non-popup elements, popup analyses)
        """
        popup_analyses = []
        non_popup_elements = []
        
        for element in ui_elements:
            analysis = self.classify_popup(element, ui_elements, screen_size, app_context)
            
            if analysis.popup_type != PopupType.UNKNOWN and analysis.confidence > 0.5:
                popup_analyses.append(analysis)
                log.debug(f"Popup detected: {analysis.popup_type.value} - {analysis.reasoning}")
            else:
                non_popup_elements.append(element)
        
        return non_popup_elements, popup_analyses

    def get_dismiss_actions(self, popup_analyses: List[PopupAnalysis]) -> List[Dict[str, Any]]:
        """Get actions to dismiss external popups.
        
        Args:
            popup_analyses: List of popup analyses
            
        Returns:
            List of dismiss actions
        """
        dismiss_actions = []
        
        for analysis in popup_analyses:
            if analysis.action == PopupAction.DISMISS:
                element = analysis.element
                bounds = element.get('bounds', {})
                
                # Create dismiss action
                dismiss_action = {
                    "type": "tap",
                    "x": (bounds.get('x', 0) + bounds.get('x2', 0)) // 2,
                    "y": (bounds.get('y', 0) + bounds.get('y2', 0)) // 2,
                    "reasoning": f"Dismiss {analysis.popup_type.value}: {analysis.reasoning}",
                    "popup_type": analysis.popup_type.value,
                    "confidence": analysis.confidence
                }
                dismiss_actions.append(dismiss_action)
        
        return dismiss_actions


# Global instance
_popup_classifier: Optional[PopupClassifier] = None


def get_popup_classifier() -> PopupClassifier:
    """Get the global popup classifier instance."""
    global _popup_classifier
    if _popup_classifier is None:
        _popup_classifier = PopupClassifier()
    return _popup_classifier 