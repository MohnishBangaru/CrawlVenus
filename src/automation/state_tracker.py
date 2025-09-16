"""State tracking and monitoring for Android automation."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..core.logger import log


class StateTracker:
    """Tracks and monitors device state changes during automation."""
    
    def __init__(self):
        """Initialize the state tracker."""
        self.state_history: List[Dict[str, Any]] = []
        self.current_state: Optional[Dict[str, Any]] = None
        self.state_change_threshold = 0.1  # Minimum change to consider significant
        
    def update_state(self, new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update the current state and track changes.
        
        Args:
            new_state: New device state information.
            
        Returns:
            Dict containing state change information.
        """
        timestamp = time.time()
        
        # Add timestamp to state
        new_state['timestamp'] = timestamp
        
        # Calculate state changes
        state_changes = self._calculate_state_changes(new_state)
        
        # Update current state
        self.current_state = new_state.copy()
        
        # Record in history
        state_record = {
            'timestamp': timestamp,
            'state': new_state,
            'changes': state_changes,
            'significant_changes': self._identify_significant_changes(state_changes)
        }
        
        self.state_history.append(state_record)
        
        # Keep only recent history (last 100 states)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
            
        log.debug(f"State updated at {timestamp}, {len(state_changes)} changes detected")
        
        return state_record
    
    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get the current device state.
        
        Returns:
            Current state dictionary or None if no state recorded.
        """
        return self.current_state
    
    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get state history.
        
        Args:
            limit: Maximum number of states to return.
            
        Returns:
            List of state records.
        """
        if limit is None:
            return self.state_history.copy()
        else:
            return self.state_history[-limit:]
    
    def get_recent_changes(self, time_window: float = 60.0) -> List[Dict[str, Any]]:
        """Get state changes within a time window.
        
        Args:
            time_window: Time window in seconds.
            
        Returns:
            List of state changes within the window.
        """
        current_time = time.time()
        recent_changes = []
        
        for record in reversed(self.state_history):
            if current_time - record['timestamp'] <= time_window:
                if record['significant_changes']:
                    recent_changes.append(record)
            else:
                break
                
        return recent_changes
    
    def has_state_changed_significantly(self, new_state: Dict[str, Any]) -> bool:
        """Check if the new state represents a significant change.
        
        Args:
            new_state: New state to compare against current state.
            
        Returns:
            True if significant changes detected.
        """
        if self.current_state is None:
            return True
            
        changes = self._calculate_state_changes(new_state)
        significant_changes = self._identify_significant_changes(changes)
        
        return len(significant_changes) > 0
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state tracking.
        
        Returns:
            Summary dictionary with statistics.
        """
        if not self.state_history:
            return {
                'total_states': 0,
                'total_changes': 0,
                'significant_changes': 0,
                'last_update': None,
                'tracking_duration': 0.0
            }
            
        total_states = len(self.state_history)
        total_changes = sum(len(record['changes']) for record in self.state_history)
        significant_changes = sum(len(record['significant_changes']) for record in self.state_history)
        
        first_timestamp = self.state_history[0]['timestamp']
        last_timestamp = self.state_history[-1]['timestamp']
        tracking_duration = last_timestamp - first_timestamp
        
        return {
            'total_states': total_states,
            'total_changes': total_changes,
            'significant_changes': significant_changes,
            'last_update': last_timestamp,
            'tracking_duration': tracking_duration,
            'average_changes_per_state': total_changes / total_states if total_states > 0 else 0
        }
    
    def _calculate_state_changes(self, new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate changes between current and new state.
        
        Args:
            new_state: New state to compare.
            
        Returns:
            Dictionary of detected changes.
        """
        if self.current_state is None:
            return {'initial_state': True}
            
        changes = {}
        
        # Compare basic state fields
        for key in new_state:
            if key == 'timestamp':
                continue
                
            if key not in self.current_state:
                changes[f'added_{key}'] = new_state[key]
            elif new_state[key] != self.current_state[key]:
                changes[f'changed_{key}'] = {
                    'old': self.current_state[key],
                    'new': new_state[key]
                }
        
        # Check for removed fields
        for key in self.current_state:
            if key == 'timestamp':
                continue
                
            if key not in new_state:
                changes[f'removed_{key}'] = self.current_state[key]
                
        return changes
    
    def _identify_significant_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Identify which changes are significant.
        
        Args:
            changes: Dictionary of all changes.
            
        Returns:
            Dictionary of significant changes only.
        """
        significant_changes = {}
        
        for change_key, change_value in changes.items():
            if self._is_significant_change(change_key, change_value):
                significant_changes[change_key] = change_value
                
        return significant_changes
    
    def _is_significant_change(self, change_key: str, change_value: Any) -> bool:
        """Determine if a change is significant.
        
        Args:
            change_key: Key identifying the change.
            change_value: Value of the change.
            
        Returns:
            True if the change is considered significant.
        """
        # Always consider structural changes significant
        if change_key.startswith(('added_', 'removed_')):
            return True
            
        # For numeric changes, check against threshold
        if change_key.startswith('changed_'):
            if isinstance(change_value, dict) and 'old' in change_value and 'new' in change_value:
                old_val = change_value['old']
                new_val = change_value['new']
                
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    if old_val != 0:
                        change_ratio = abs(new_val - old_val) / abs(old_val)
                        return change_ratio > self.state_change_threshold
                    else:
                        return new_val != 0
                        
        # For other changes, consider them significant
        return True
    
    def clear_history(self) -> None:
        """Clear the state history."""
        self.state_history.clear()
        self.current_state = None
        
    def export_state_data(self, filepath: str) -> bool:
        """Export state tracking data to a file.
        
        Args:
            filepath: Path to save the data.
            
        Returns:
            True if export successful.
        """
        try:
            import json
            
            export_data = {
                'summary': self.get_state_summary(),
                'history': self.state_history,
                'current_state': self.current_state
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
            log.info(f"State data exported to {filepath}")
            return True
            
        except Exception as e:
            log.error(f"Failed to export state data: {e}")
            return False 