#!/usr/bin/env python3
"""
Fix Test Directories Script

This script creates the necessary directories for testing.
"""

import os
from pathlib import Path

def main():
    """Create necessary test directories."""
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    
    # Directories to create
    directories = [
        "test_reports",
        "distributed_test_reports", 
        "screenshots",
        "logs",
        "vision_debug",
        "ocr_images"
    ]
    
    print("🔧 Creating test directories...")
    
    for directory in directories:
        dir_path = project_root / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        else:
            print(f"📁 Exists: {directory}")
    
    print("\n🎉 All test directories are ready!")

if __name__ == "__main__":
    main()
