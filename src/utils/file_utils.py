"""File utility functions for DroidBot-GPT framework."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.logger import log


def ensure_directory(directory_path: str) -> str:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory.
        
    Returns:
        Absolute path to the directory.
    """
    path = Path(directory_path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_timestamp() -> str:
    """Get current timestamp as a string.
    
    Returns:
        Timestamp string in format YYYY-MM-DD_HH-MM-SS.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def get_timestamp_ms() -> int:
    """Get current timestamp in milliseconds.
    
    Returns:
        Current timestamp in milliseconds.
    """
    return int(time.time() * 1000)


def save_json(data: Any, filepath: str, indent: int = 2) -> bool:
    """Save data to a JSON file.
    
    Args:
        data: Data to save.
        filepath: Path to the JSON file.
        indent: JSON indentation level.
        
    Returns:
        True if save successful, False otherwise.
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            ensure_directory(directory)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            
        log.debug(f"Data saved to {filepath}")
        return True
        
    except Exception as e:
        log.error(f"Failed to save JSON to {filepath}: {e}")
        return False


def load_json(filepath: str) -> Optional[Any]:
    """Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file.
        
    Returns:
        Loaded data or None if failed.
    """
    try:
        if not os.path.exists(filepath):
            log.warning(f"JSON file not found: {filepath}")
            return None
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        log.debug(f"Data loaded from {filepath}")
        return data
        
    except Exception as e:
        log.error(f"Failed to load JSON from {filepath}: {e}")
        return None


def save_screenshot(image_data: bytes, filepath: str) -> bool:
    """Save screenshot data to a file.
    
    Args:
        image_data: Raw image data.
        filepath: Path to save the image.
        
    Returns:
        True if save successful, False otherwise.
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            ensure_directory(directory)
            
        with open(filepath, 'wb') as f:
            f.write(image_data)
            
        log.debug(f"Screenshot saved to {filepath}")
        return True
        
    except Exception as e:
        log.error(f"Failed to save screenshot to {filepath}: {e}")
        return False


def get_file_size(filepath: str) -> Optional[int]:
    """Get file size in bytes.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        File size in bytes or None if file doesn't exist.
    """
    try:
        if os.path.exists(filepath):
            return os.path.getsize(filepath)
        return None
    except Exception as e:
        log.error(f"Failed to get file size for {filepath}: {e}")
        return None


def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
    """Clean up old files in a directory.
    
    Args:
        directory: Directory to clean up.
        max_age_hours: Maximum age of files to keep in hours.
        
    Returns:
        Number of files removed.
    """
    try:
        if not os.path.exists(directory):
            return 0
            
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        removed_count = 0
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                
                if file_age > max_age_seconds:
                    try:
                        os.remove(filepath)
                        removed_count += 1
                        log.debug(f"Removed old file: {filepath}")
                    except Exception as e:
                        log.warning(f"Failed to remove old file {filepath}: {e}")
                        
        if removed_count > 0:
            log.info(f"Cleaned up {removed_count} old files from {directory}")
            
        return removed_count
        
    except Exception as e:
        log.error(f"Failed to cleanup old files in {directory}: {e}")
        return 0


def create_session_directory(session_id: str) -> str:
    """Create a session directory structure.
    
    Args:
        session_id: Unique session identifier.
        
    Returns:
        Path to the created session directory.
    """
    session_dir = os.path.join("sessions", session_id)
    ensure_directory(session_dir)
    
    # Create subdirectories
    subdirs = ["screenshots", "logs", "actions", "analysis", "exports"]
    for subdir in subdirs:
        ensure_directory(os.path.join(session_dir, subdir))
        
    log.info(f"Session directory created: {session_dir}")
    return session_dir


def export_session_data(session_id: str, data: Dict[str, Any], filename: str) -> bool:
    """Export session data to a file.
    
    Args:
        session_id: Session identifier.
        data: Data to export.
        filename: Name of the export file.
        
    Returns:
        True if export successful, False otherwise.
    """
    try:
        session_dir = os.path.join("sessions", session_id, "exports")
        ensure_directory(session_dir)
        
        filepath = os.path.join(session_dir, filename)
        return save_json(data, filepath)
        
    except Exception as e:
        log.error(f"Failed to export session data: {e}")
        return False 