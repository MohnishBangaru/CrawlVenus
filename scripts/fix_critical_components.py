#!/usr/bin/env python3
"""Fix critical AI components for AA_VA-Phi.

This script fixes the three critical components:
1. Phi Ground Model: transformers.cache_utils issue
2. OpenAI Client: API key configuration
3. Tesseract OCR: Installation and setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command: str, description: str, silent: bool = False) -> bool:
    """Run a command and return success status."""
    if not silent:
        print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            if not silent:
                print(f"✅ {description} - Success")
            return True
        else:
            if not silent:
                print(f"❌ {description} - Failed")
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        if not silent:
            print(f"❌ {description} - Exception: {e}")
        return False

def fix_transformers_issue():
    """Fix transformers.cache_utils issue by updating transformers."""
    print("\n🚀 Fixing Phi Ground Transformers Issue...")
    
    # Update transformers to latest version
    success = run_command(
        "pip install --upgrade transformers>=4.36.0",
        "Updating transformers to latest version"
    )
    
    if success:
        # Also update related packages
        run_command("pip install --upgrade accelerate", "Updating accelerate")
        run_command("pip install --upgrade safetensors", "Updating safetensors")
        run_command("pip install --upgrade huggingface-hub", "Updating huggingface-hub")
        
        # Try to install FlashAttention2 using specialized script
        print("🔄 Attempting to install FlashAttention2 for GPU acceleration...")
        flash_result = run_command("python scripts/install_flash_attention.py", "Installing FlashAttention2 using specialized script", silent=True)
        if flash_result:
            print("✅ FlashAttention2 installation completed")
        else:
            print("⚠️ FlashAttention2 installation failed - will use standard attention")
            print("   This is normal and the system will work without GPU acceleration")
    
    return success

def setup_openai_api_key():
    """Setup OpenAI API key interactively."""
    print("\n🤖 Setting up OpenAI API Key...")
    
    # Check if API key already exists
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if "OPENAI_API_KEY=" in content and "OPENAI_API_KEY=your_openai_api_key_here" not in content:
                print("✅ OpenAI API key already configured")
                return True
    
    # Get API key from user
    print("📝 Please enter your OpenAI API key (or press Enter to skip):")
    api_key = input("OpenAI API Key: ").strip()
    
    if not api_key:
        print("⚠️  Skipping OpenAI API key setup")
        return False
    
    # Update .env file
    if env_file.exists():
        # Read existing content
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Update or add OPENAI_API_KEY
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("OPENAI_API_KEY="):
                lines[i] = f"OPENAI_API_KEY={api_key}\n"
                updated = True
                break
        
        if not updated:
            lines.append(f"OPENAI_API_KEY={api_key}\n")
        
        # Write back
        with open(env_file, 'w') as f:
            f.writelines(lines)
    else:
        # Create new .env file
        with open(env_file, 'w') as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
    
    print("✅ OpenAI API key configured successfully")
    return True

def install_tesseract():
    """Install Tesseract OCR based on the operating system."""
    print("\n👁️  Installing Tesseract OCR...")
    
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        return run_command("brew install tesseract", "Installing Tesseract via Homebrew")
    
    elif system == "linux":
        # Try different package managers
        if run_command("which apt-get", "Checking for apt-get"):
            return run_command("sudo apt-get update && sudo apt-get install -y tesseract-ocr", "Installing Tesseract via apt-get")
        elif run_command("which yum", "Checking for yum"):
            return run_command("sudo yum install -y tesseract", "Installing Tesseract via yum")
        elif run_command("which dnf", "Checking for dnf"):
            return run_command("sudo dnf install -y tesseract", "Installing Tesseract via dnf")
        else:
            print("❌ Could not determine package manager for Linux")
            return False
    
    elif system == "windows":
        print("⚠️  For Windows, please install Tesseract manually:")
        print("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Install and add to PATH")
        return False
    
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False

def install_python_tesseract():
    """Install Python Tesseract wrapper."""
    print("\n🐍 Installing Python Tesseract wrapper...")
    return run_command("pip install pytesseract", "Installing pytesseract")

def test_components():
    """Test if all components are working."""
    print("\n🧪 Testing Components...")
    
    # Test transformers
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        # Test cache_utils specifically
        from transformers import cache_utils
        print("✅ transformers.cache_utils available")
    except ImportError as e:
        print(f"❌ Transformers issue: {e}")
        return False
    
    # Test OpenAI
    try:
        import openai
        print("✅ OpenAI package available")
        
        # Test if API key is configured (without importing src)
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            print("✅ OpenAI API key configured")
        else:
            print("⚠️  OpenAI API key not configured (will use fallback)")
    except Exception as e:
        print(f"⚠️  OpenAI test failed: {e}")
    
    # Test Tesseract
    try:
        import pytesseract
        print("✅ pytesseract available")
        
        # Test if tesseract is installed
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        return False
    
    return True

def main():
    """Main function to fix all critical components."""
    print("🔧 AA_VA-Phi Critical Components Fix")
    print("=" * 50)
    
    # Fix transformers issue
    transformers_fixed = fix_transformers_issue()
    
    # Setup OpenAI API key
    openai_setup = setup_openai_api_key()
    
    # Install Tesseract
    tesseract_installed = install_tesseract()
    if tesseract_installed:
        install_python_tesseract()
    
    # Test all components
    print("\n" + "=" * 50)
    print("🧪 Testing All Components...")
    all_working = test_components()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Fix Summary:")
    print(f"   Transformers: {'✅ Fixed' if transformers_fixed else '❌ Failed'}")
    print(f"   OpenAI Setup: {'✅ Configured' if openai_setup else '⚠️  Skipped'}")
    print(f"   Tesseract: {'✅ Installed' if tesseract_installed else '❌ Failed'}")
    print(f"   Overall: {'✅ All Working' if all_working else '⚠️  Some Issues'}")
    
    if all_working:
        print("\n🎉 All critical components are now working!")
        print("You can now run the distributed APK tester with full AI capabilities.")
    else:
        print("\n⚠️  Some components still need attention.")
        print("The system will work with fallback methods, but AI features may be limited.")

if __name__ == "__main__":
    main()
