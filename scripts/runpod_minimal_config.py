#!/usr/bin/env python3
"""
Minimal Configuration for RunPod

This script creates a minimal configuration for RunPod that doesn't require API keys.
"""

import os
import sys
from pathlib import Path

def create_minimal_env():
    """Create a minimal .env file for RunPod."""
    
    print("🔧 Creating minimal configuration for RunPod...")
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    
    # Minimal configuration for RunPod
    env_content = [
        "# AA_VA-Phi Minimal Configuration for RunPod",
        "# This configuration allows basic functionality without API keys",
        "",
        "# OpenAI Configuration (optional)",
        "OPENAI_API_KEY=",
        "OPENAI_MODEL=gpt-4",
        "OPENAI_MAX_TOKENS=2000",
        "OPENAI_TEMPERATURE=0.7",
        "",
        "# Phi Ground Configuration (free, local)",
        "USE_PHI_GROUND=true",
        "PHI_GROUND_MODEL=microsoft/Phi-3-vision-128k-instruct",
        "PHI_GROUND_TEMPERATURE=0.7",
        "PHI_GROUND_MAX_TOKENS=256",
        "",
        "# Distributed Setup",
        "LOCAL_ADB_HOST=127.0.0.1",
        "LOCAL_ADB_PORT=8000",
        "",
        "# Framework Configuration",
        "LOG_LEVEL=INFO",
        "DEBUG_MODE=false",
        "SAVE_SCREENSHOTS=true",
        "SCREENSHOT_DIR=screenshots",
        "",
        "# Computer Vision Configuration",
        "CV_CONFIDENCE_THRESHOLD=0.8",
        "CV_TEMPLATE_MATCHING_THRESHOLD=0.9",
        "USE_OCR=true",
        "",
        "# Android Configuration",
        "ANDROID_DEVICE_ID=emulator-5554",
        "ANDROID_PLATFORM_VERSION=30",
        "",
        "# Resource Management",
        "MAX_WAIT_TIME=10",
        "RETRY_ATTEMPTS=3",
        "BATCH_SIZE=5"
    ]
    
    # Write to file
    with open(env_file, 'w') as f:
        f.write('\n'.join(env_content))
    
    print(f"✅ Minimal configuration created: {env_file}")
    print()
    print("📋 This configuration includes:")
    print("   ✅ Basic framework settings")
    print("   ✅ Phi Ground (free, local AI)")
    print("   ✅ Computer vision settings")
    print("   ✅ Distributed setup defaults")
    print("   ⚠️  No OpenAI API key (optional)")
    print()
    print("🚀 You can now run:")
    print("   python scripts/quick_start_distributed.py --local-ip YOUR_LAPTOP_IP")
    print()
    print("💡 To add OpenAI API key later:")
    print("   python scripts/setup_api_keys.py")

def main():
    """Main function."""
    
    try:
        create_minimal_env()
        return 0
    except Exception as e:
        print(f"❌ Error creating minimal config: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
