#!/usr/bin/env python3
"""
Quick Fix for dotenv Module Error

This script quickly installs the missing dotenv module.
"""

import subprocess
import sys

def main():
    """Install python-dotenv quickly."""
    
    print("🔧 Quick Fix: Installing python-dotenv...")
    
    try:
        # Install python-dotenv
        subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"], check=True)
        print("✅ python-dotenv installed successfully!")
        
        # Also install other common missing packages
        packages = ["fastapi", "uvicorn", "aiohttp", "requests", "pydantic", "pydantic-settings"]
        for package in packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                print(f"✅ {package} installed successfully!")
            except subprocess.CalledProcessError:
                print(f"⚠️  {package} installation failed (may already be installed)")
        
        print("\n🎉 Quick fix completed! Try running your script again.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("\n💡 Alternative: Try running manually:")
        print("pip install python-dotenv fastapi uvicorn aiohttp requests pydantic-settings")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
