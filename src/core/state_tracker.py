"""State tracking system for DroidBot-GPT automation.

This module tracks visited UI states and helps avoid revisiting them,
ensuring the automation explores new states and doesn't get stuck in loops.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from ..vision.models import UIElement


class StateTracker:
    """Tracks visited UI states and manages state exploration strategy."""
    
    def __init__(self) -> None:
        """Initialize the state tracker."""
        self.visited_states: Set[str] = set()
        self.state_history: List[Dict[str, Any]] = []
        self.state_visit_counts: Dict[str, int] = {}
        self.last_state_change_time: float = 0.0
        self.current_state_hash: str = ""
        self.state_similarity_threshold: float = 0.85
        self.max_state_history: int = 100
        
    def _generate_state_hash(self, ui_elements: List[UIElement], context: str = "") -> str:
        """Generate a unique hash for the current UI state."""
        # Create a state signature based on element properties
        state_data = {
            "element_count": len(ui_elements),
            "element_signatures": [],
            "context": context
        }
        
        # Create signatures for each element (text, type, relative position)
        for element in ui_elements:
            x1, y1, x2, y2 = element.bbox.as_tuple()
            # Normalize position to relative coordinates (0-100)
            rel_x = (x1 + x2) / 2 / 1080 * 100  # Assuming 1080px width
            rel_y = (y1 + y2) / 2 / 1920 * 100  # Assuming 1920px height
            
            element_signature = {
                "text": element.text.lower().strip(),
                "type": element.element_type,
                "rel_x": round(rel_x, 1),
                "rel_y": round(rel_y, 1),
                "confidence": round(element.confidence, 2)
            }
            state_data["element_signatures"].append(element_signature)
        
        # Sort element signatures for consistent hashing
        state_data["element_signatures"].sort(key=lambda x: (x["rel_y"], x["rel_x"]))
        
        # Convert to JSON string and hash
        json_str = json.dumps(state_data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _calculate_state_similarity(self, state1_elements: List[UIElement], state2_elements: List[UIElement]) -> float:
        """Calculate similarity between two states (0.0 to 1.0)."""
        if not state1_elements or not state2_elements:
            return 0.0
        
        # Count similar elements
        similar_count = 0
        total_elements = max(len(state1_elements), len(state2_elements))
        
        for elem1 in state1_elements:
            for elem2 in state2_elements:
                # Check if elements are similar (same text and type)
                if (elem1.text.lower().strip() == elem2.text.lower().strip() and 
                    elem1.element_type == elem2.element_type):
                    similar_count += 1
                    break
        
        return similar_count / total_elements if total_elements > 0 else 0.0
    
    def is_state_visited(self, ui_elements: List[UIElement], context: str = "") -> bool:
        """Check if the current state has been visited before."""
        current_hash = self._generate_state_hash(ui_elements, context)
        
        # Direct hash match
        if current_hash in self.visited_states:
            return True
        
        # Check for similar states in history
        for history_entry in self.state_history:
            if history_entry.get("context") == context:
                # Reconstruct historical elements (simplified)
                historical_elements = history_entry.get("elements", [])
                if historical_elements:
                    similarity = self._calculate_state_similarity(ui_elements, historical_elements)
                    if similarity >= self.state_similarity_threshold:
                        logger.debug(f"State similar to visited state (similarity: {similarity:.2f})")
                        return True
        
        return False
    
    def mark_state_visited(self, ui_elements: List[UIElement], context: str = "", action: Optional[Dict[str, Any]] = None) -> None:
        """Mark the current state as visited."""
        state_hash = self._generate_state_hash(ui_elements, context)
        
        # Add to visited set
        self.visited_states.add(state_hash)
        
        # Update visit count
        self.state_visit_counts[state_hash] = self.state_visit_counts.get(state_hash, 0) + 1
        
        # Record in history
        history_entry = {
            "timestamp": time.time(),
            "state_hash": state_hash,
            "context": context,
            "element_count": len(ui_elements),
            "elements": [
                {
                    "text": elem.text,
                    "type": elem.element_type,
                    "bbox": elem.bbox.as_tuple(),
                    "confidence": elem.confidence
                }
                for elem in ui_elements
            ],
            "action": action,
            "visit_count": self.state_visit_counts[state_hash]
        }
        
        self.state_history.append(history_entry)
        self.last_state_change_time = time.time()
        self.current_state_hash = state_hash
        
        # Limit history size
        if len(self.state_history) > self.max_state_history:
            self.state_history.pop(0)
        
        logger.debug(f"Marked state as visited: {state_hash[:8]} (elements: {len(ui_elements)})")
    
    def get_state_exploration_priority(self, ui_elements: List[UIElement], context: str = "") -> float:
        """Get exploration priority for the current state (higher = more novel)."""
        if not ui_elements:
            return 0.0
        
        # Check if state is visited
        is_visited = self.is_state_visited(ui_elements, context)
        
        if not is_visited:
            # New state gets high priority
            return 1.0
        
        # Visited state gets lower priority based on visit count
        state_hash = self._generate_state_hash(ui_elements, context)
        visit_count = self.state_visit_counts.get(state_hash, 1)
        
        # Exponential decay based on visit count
        priority = max(0.0, 1.0 - (visit_count * 0.3))
        
        # Time-based bonus for old visited states
        time_since_visit = time.time() - self.last_state_change_time
        if time_since_visit > 300:  # 5 minutes
            priority += 0.1
        
        return priority
    
    def get_state_transition_analysis(self, previous_elements: List[UIElement], current_elements: List[UIElement]) -> Dict[str, Any]:
        """Analyze the transition from previous state to current state."""
        if not previous_elements or not current_elements:
            return {"type": "unknown", "confidence": 0.0}
        
        # Calculate similarity
        similarity = self._calculate_state_similarity(previous_elements, current_elements)
        
        # Analyze transition type
        if similarity > 0.9:
            transition_type = "no_change"
        elif similarity > 0.7:
            transition_type = "minor_change"
        elif similarity > 0.3:
            transition_type = "major_change"
        else:
            transition_type = "new_screen"
        
        # Count new elements
        new_elements = []
        for current_elem in current_elements:
            is_new = True
            for prev_elem in previous_elements:
                if (current_elem.text.lower().strip() == prev_elem.text.lower().strip() and 
                    current_elem.element_type == prev_elem.element_type):
                    is_new = False
                    break
            if is_new:
                new_elements.append(current_elem.text)
        
        return {
            "type": transition_type,
            "similarity": similarity,
            "new_elements": new_elements,
            "new_element_count": len(new_elements),
            "confidence": 1.0 - similarity  # Higher confidence for more novel states
        }
    
    def get_state_exploration_stats(self) -> Dict[str, Any]:
        """Get statistics about state exploration."""
        total_visited = len(self.visited_states)
        total_visits = sum(self.state_visit_counts.values())
        
        # Analyze recent activity
        recent_time = time.time() - 300  # Last 5 minutes
        recent_states = [
            entry for entry in self.state_history 
            if entry["timestamp"] > recent_time
        ]
        
        # Calculate average visits per state
        avg_visits = total_visits / total_visited if total_visited > 0 else 0
        
        # Find most visited state
        most_visited_state = None
        max_visits = 0
        for state_hash, visits in self.state_visit_counts.items():
            if visits > max_visits:
                max_visits = visits
                most_visited_state = state_hash
        
        return {
            "total_visited_states": total_visited,
            "total_state_visits": total_visits,
            "average_visits_per_state": round(avg_visits, 2),
            "recent_state_visits": len(recent_states),
            "most_visited_state": most_visited_state[:8] if most_visited_state else None,
            "max_visits_to_state": max_visits,
            "current_state_hash": self.current_state_hash[:8] if self.current_state_hash else None,
            "last_state_change_time": self.last_state_change_time,
            "state_history_length": len(self.state_history)
        }
    
    def reset_state_tracking(self) -> None:
        """Reset all state tracking."""
        self.visited_states.clear()
        self.state_history.clear()
        self.state_visit_counts.clear()
        self.last_state_change_time = 0.0
        self.current_state_hash = ""
        logger.info("State tracking reset")
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """Set the state similarity threshold."""
        if 0.0 <= threshold <= 1.0:
            self.state_similarity_threshold = threshold
            logger.info(f"State similarity threshold set to: {threshold}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def export_state_data(self) -> Dict[str, Any]:
        """Export state tracking data for analysis."""
        return {
            "visited_states": list(self.visited_states),
            "state_history": self.state_history,
            "state_visit_counts": self.state_visit_counts,
            "similarity_threshold": self.state_similarity_threshold,
            "last_state_change_time": self.last_state_change_time,
            "current_state_hash": self.current_state_hash,
            "stats": self.get_state_exploration_stats()
        }
    
    def import_state_data(self, data: Dict[str, Any]) -> None:
        """Import state tracking data."""
        self.visited_states = set(data.get("visited_states", []))
        self.state_history = data.get("state_history", [])
        self.state_visit_counts = data.get("state_visit_counts", {})
        self.state_similarity_threshold = data.get("similarity_threshold", 0.85)
        self.last_state_change_time = data.get("last_state_change_time", 0.0)
        self.current_state_hash = data.get("current_state_hash", "")
        
        logger.info(f"Imported state data: {len(self.visited_states)} visited states")


# Global instance for reuse
_state_tracker = None


def get_state_tracker() -> StateTracker:
    """Get or create the global state tracker instance."""
    global _state_tracker
    if _state_tracker is None:
        _state_tracker = StateTracker()
    return _state_tracker 