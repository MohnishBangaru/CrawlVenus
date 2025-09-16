#!/usr/bin/env python3
"""
Fix RunPod Dependencies Script

This script installs missing dependencies commonly needed on RunPod.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description}: SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}: FAILED")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Main function to fix dependencies."""
    
    print("🚀 RunPod Dependency Fixer")
    print("="*40)
    
    # List of packages to install
    packages = [
        ("python-dotenv", "Python dotenv for environment variables"),
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("aiohttp", "Async HTTP client/server"),
        ("requests", "HTTP library"),
        ("pydantic", "Data validation"),
        ("pydantic-settings", "Settings management"),
        ("loguru", "Advanced logging"),
        ("rich", "Rich text formatting"),
        ("typer", "CLI framework"),
        ("numpy", "Numerical computing"),
        ("Pillow", "Image processing"),
        ("opencv-python", "Computer vision"),
        ("transformers", "Hugging Face transformers"),
        ("torch", "PyTorch"),
        ("accelerate", "Accelerated training"),
        ("openai", "OpenAI API client"),
        ("adbutils", "Android Debug Bridge utilities"),
        ("psutil", "System monitoring"),
        ("aiofiles", "Async file operations"),
        ("redis", "Redis client"),
        ("pytest", "Testing framework"),
        ("GitPython", "Git integration")
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package, description in packages:
        if run_command(f"pip install {package}", f"Installing {package}"):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{total_count} packages installed successfully")
    
    if success_count == total_count:
        print("🎉 All dependencies installed successfully!")
        return 0
    else:
        print("⚠️  Some packages failed to install. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
