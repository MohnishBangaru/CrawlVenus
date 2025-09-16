"""Element tracking system for DroidBot-GPT automation.

This module tracks which UI elements have been explored and interacted with,
ensuring the automation focuses on unseen elements to avoid redundant interactions.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from ..vision.models import UIElement


class ElementTracker:
    """Tracks explored UI elements and manages element exploration strategy."""
    
    def __init__(self) -> None:
        """Initialize the element tracker."""
        self.explored_elements: Set[str] = set()
        self.element_history: List[Dict[str, Any]] = []
        self.interaction_counts: Dict[str, int] = {}
        self.last_exploration_time: float = 0.0
        self.exploration_strategy = "unseen_first"  # unseen_first, confidence_based, hybrid
        
    def _generate_element_hash(self, element: UIElement, context: str = "") -> str:
        """Generate a unique hash for an element based on its properties and context."""
        # Create a hash based on element text, position, and context
        element_data = {
            "text": element.text.lower().strip(),
            "bbox": element.bbox.as_tuple(),
            "element_type": element.element_type,
            "context": context
        }
        
        # Convert to JSON string and hash
        json_str = json.dumps(element_data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _calculate_element_similarity(self, element1: UIElement, element2: UIElement) -> float:
        """Calculate similarity between two elements (0.0 to 1.0)."""
        # Text similarity
        text1 = element1.text.lower().strip()
        text2 = element2.text.lower().strip()
        text_similarity = 1.0 if text1 == text2 else 0.0
        
        # Position similarity (normalized distance)
        bbox1 = element1.bbox.as_tuple()
        bbox2 = element2.bbox.as_tuple()
        
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        # Calculate Euclidean distance and normalize
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        max_distance = 1000  # Maximum reasonable distance
        position_similarity = max(0.0, 1.0 - (distance / max_distance))
        
        # Type similarity
        type_similarity = 1.0 if element1.element_type == element2.element_type else 0.0
        
        # Weighted average
        return (text_similarity * 0.5 + position_similarity * 0.3 + type_similarity * 0.2)
    
    def mark_element_explored(self, element: UIElement, action: Dict[str, Any], context: str = "") -> None:
        """Mark an element as explored after interaction."""
        element_hash = self._generate_element_hash(element, context)
        
        # Add to explored set
        self.explored_elements.add(element_hash)
        
        # Update interaction count
        self.interaction_counts[element_hash] = self.interaction_counts.get(element_hash, 0) + 1
        
        # Record in history
        history_entry = {
            "timestamp": time.time(),
            "element_hash": element_hash,
            "element_text": element.text,
            "element_type": element.element_type,
            "bbox": element.bbox.as_tuple(),
            "action": action,
            "context": context,
            "confidence": element.confidence
        }
        
        self.element_history.append(history_entry)
        self.last_exploration_time = time.time()
        
        logger.debug(f"Marked element as explored: {element.text} (hash: {element_hash[:8]})")
    
    def is_element_explored(self, element: UIElement, context: str = "", similarity_threshold: float = 0.8) -> bool:
        """Check if an element has been explored (including similar elements)."""
        current_hash = self._generate_element_hash(element, context)
        
        # Direct hash match
        if current_hash in self.explored_elements:
            return True
        
        # Check for similar elements in history
        for history_entry in self.element_history:
            # Reconstruct the historical element
            historical_element = UIElement(
                bbox=element.bbox,  # Use current bbox as placeholder
                text=history_entry["element_text"],
                confidence=history_entry["confidence"],
                element_type=history_entry["element_type"]
            )
            
            # Calculate similarity
            similarity = self._calculate_element_similarity(element, historical_element)
            if similarity >= similarity_threshold:
                logger.debug(f"Element {element.text} similar to explored element {history_entry['element_text']} (similarity: {similarity:.2f})")
                return True
        
        return False
    
    def get_exploration_priority(self, elements: List[UIElement], context: str = "") -> List[Tuple[UIElement, float]]:
        """Get elements prioritized by exploration strategy."""
        if not elements:
            return []
        
        element_priorities = []
        
        for element in elements:
            priority = 0.0
            
            # Check if element is unexplored
            is_explored = self.is_element_explored(element, context)
            
            if self.exploration_strategy == "unseen_first":
                # Prioritize unseen elements
                priority = 0.0 if is_explored else 1.0
                
            elif self.exploration_strategy == "confidence_based":
                # Prioritize by confidence, but still prefer unseen
                priority = element.confidence
                if not is_explored:
                    priority += 0.5  # Bonus for unseen elements
                    
            elif self.exploration_strategy == "hybrid":
                # Hybrid approach: unseen + confidence + recency
                base_priority = element.confidence
                
                if not is_explored:
                    base_priority += 0.8  # High bonus for unseen
                else:
                    # Reduce priority for explored elements based on interaction count
                    element_hash = self._generate_element_hash(element, context)
                    interaction_count = self.interaction_counts.get(element_hash, 0)
                    base_priority -= (interaction_count * 0.2)  # Penalty for multiple interactions
                
                # Time-based decay for explored elements
                if is_explored:
                    time_since_exploration = time.time() - self.last_exploration_time
                    if time_since_exploration > 300:  # 5 minutes
                        base_priority += 0.1  # Small bonus for old explored elements
                
                priority = max(0.0, base_priority)
            
            element_priorities.append((element, priority))
        
        # Sort by priority (highest first)
        element_priorities.sort(key=lambda x: x[1], reverse=True)
        
        return element_priorities
    
    def get_unexplored_elements(self, elements: List[UIElement], context: str = "") -> List[UIElement]:
        """Get only unexplored elements from the list."""
        unexplored = []
        
        for element in elements:
            if not self.is_element_explored(element, context):
                unexplored.append(element)
        
        return unexplored
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get statistics about element exploration."""
        total_explored = len(self.explored_elements)
        total_interactions = sum(self.interaction_counts.values())
        
        # Analyze recent activity
        recent_time = time.time() - 300  # Last 5 minutes
        recent_explorations = [
            entry for entry in self.element_history 
            if entry["timestamp"] > recent_time
        ]
        
        return {
            "total_explored_elements": total_explored,
            "total_interactions": total_interactions,
            "recent_explorations": len(recent_explorations),
            "exploration_strategy": self.exploration_strategy,
            "last_exploration_time": self.last_exploration_time,
            "element_history_length": len(self.element_history)
        }
    
    def reset_exploration(self) -> None:
        """Reset all exploration tracking."""
        self.explored_elements.clear()
        self.element_history.clear()
        self.interaction_counts.clear()
        self.last_exploration_time = 0.0
        logger.info("Element exploration tracking reset")
    
    def set_exploration_strategy(self, strategy: str) -> None:
        """Set the exploration strategy."""
        valid_strategies = ["unseen_first", "confidence_based", "hybrid"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        
        self.exploration_strategy = strategy
        logger.info(f"Exploration strategy set to: {strategy}")
    
    def export_exploration_data(self) -> Dict[str, Any]:
        """Export exploration data for analysis."""
        return {
            "explored_elements": list(self.explored_elements),
            "element_history": self.element_history,
            "interaction_counts": self.interaction_counts,
            "exploration_strategy": self.exploration_strategy,
            "last_exploration_time": self.last_exploration_time,
            "stats": self.get_exploration_stats()
        }
    
    def import_exploration_data(self, data: Dict[str, Any]) -> None:
        """Import exploration data."""
        self.explored_elements = set(data.get("explored_elements", []))
        self.element_history = data.get("element_history", [])
        self.interaction_counts = data.get("interaction_counts", {})
        self.exploration_strategy = data.get("exploration_strategy", "unseen_first")
        self.last_exploration_time = data.get("last_exploration_time", 0.0)
        
        logger.info(f"Imported exploration data: {len(self.explored_elements)} explored elements")


# Global instance for reuse
_element_tracker = None


def get_element_tracker() -> ElementTracker:
    """Get or create the global element tracker instance."""
    global _element_tracker
    if _element_tracker is None:
        _element_tracker = ElementTracker()
    return _element_tracker 