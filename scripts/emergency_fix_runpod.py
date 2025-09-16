#!/usr/bin/env python3
"""
Emergency Fix for RunPod Dependencies

This script installs all critical dependencies needed to run AA_VA-Phi on RunPod.
"""

import subprocess
import sys
import os

def install_package(package, description=""):
    """Install a single package."""
    try:
        print(f"🔧 Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], 
                      check=True, capture_output=True, text=True)
        print(f"✅ {package} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e.stderr}")
        return False

def main():
    """Install all critical dependencies."""
    
    print("🚨 EMERGENCY RUNPOD DEPENDENCY FIX")
    print("="*50)
    
    # Critical dependencies for AA_VA-Phi
    critical_packages = [
        "python-dotenv",
        "pydantic",
        "pydantic-settings", 
        "fastapi",
        "uvicorn",
        "aiohttp",
        "requests",
        "loguru",
        "rich",
        "typer",
        "numpy",
        "Pillow",
        "opencv-python",
        "transformers",
        "torch",
        "accelerate",
        "openai",
        "adbutils",
        "psutil",
        "aiofiles",
        "redis",
        "pytest",
        "GitPython"
    ]
    
    success_count = 0
    failed_packages = []
    
    for package in critical_packages:
        if install_package(package):
            success_count += 1
        else:
            failed_packages.append(package)
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{len(critical_packages)}")
    
    if failed_packages:
        print(f"❌ Failed packages: {', '.join(failed_packages)}")
        print(f"\n💡 Try installing failed packages manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
    
    if success_count == len(critical_packages):
        print("\n🎉 All dependencies installed successfully!")
        print("🚀 You can now run your distributed scripts!")
    else:
        print(f"\n⚠️  {len(failed_packages)} packages failed to install.")
        print("   The core functionality should still work.")
    
    return 0 if success_count >= len(critical_packages) * 0.8 else 1

if __name__ == "__main__":
    exit(main())
