#!/usr/bin/env python3
"""
Quick Fix for .env file issues

This script creates a clean .env file that won't cause validation errors.
"""

import os
from pathlib import Path

def main():
    """Create a clean .env file."""
    
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    
    # Clean .env content that won't cause validation errors
    clean_env_content = """# AA_VA-Phi Clean Configuration
# This file contains only the essential settings

# Distributed Setup
LOCAL_ADB_HOST=127.0.0.1
LOCAL_ADB_PORT=8000

# Framework Configuration
LOG_LEVEL=INFO
DEBUG_MODE=false
SAVE_SCREENSHOTS=true

# Computer Vision
CV_CONFIDENCE_THRESHOLD=0.8
USE_OCR=true

# Android Configuration
ANDROID_DEVICE_ID=emulator-5554
"""
    
    # Write clean .env file
    with open(env_file, 'w') as f:
        f.write(clean_env_content)
    
    print(f"✅ Clean .env file created: {env_file}")
    print("🚀 You can now run your distributed scripts!")

if __name__ == "__main__":
    main()
