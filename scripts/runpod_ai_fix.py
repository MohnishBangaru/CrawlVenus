#!/usr/bin/env python3
"""RunPod AI Components Fix - Non-interactive version.

This script fixes the three critical AI components for RunPod environment:
1. Phi Ground Model: transformers.cache_utils issue
2. OpenAI Client: Create minimal config
3. Tesseract OCR: Install system dependencies
"""

import os
import sys
import subprocess
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

def create_minimal_env():
    """Create minimal .env file for RunPod."""
    print("\n🤖 Creating minimal .env file for RunPod...")
    
    env_content = """# AA_VA-Phi Minimal Configuration for RunPod
# Add your API keys here if needed

# OpenAI Configuration (optional)
OPENAI_API_KEY=

# Phi Ground Configuration
USE_PHI_GROUND=true
PHI_GROUND_MODEL=microsoft/Phi-3-vision-128k-instruct
PHI_GROUND_TEMPERATURE=0.7
PHI_GROUND_MAX_TOKENS=256
PHI_GROUND_CONFIDENCE_THRESHOLD=0.5

# Distributed Configuration
LOCAL_ADB_SERVER_URL=http://localhost:8000
OUTPUT_DIR=/workspace/test_reports
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Minimal .env file created")
    return True

def install_tesseract_linux():
    """Install Tesseract OCR on Linux (RunPod)."""
    print("\n👁️  Installing Tesseract OCR on Linux...")
    
    # Update package list
    run_command("apt-get update", "Updating package list")
    
    # Install Tesseract
    success = run_command(
        "apt-get install -y tesseract-ocr tesseract-ocr-eng",
        "Installing Tesseract OCR"
    )
    
    if success:
        # Install Python wrapper
        run_command("pip install pytesseract", "Installing pytesseract")
    
    return success

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
    """Main function to fix all critical components for RunPod."""
    print("🔧 AA_VA-Phi Critical Components Fix (RunPod)")
    print("=" * 50)
    
    # Fix transformers issue
    transformers_fixed = fix_transformers_issue()
    
    # Create minimal .env file
    env_created = create_minimal_env()
    
    # Install Tesseract
    tesseract_installed = install_tesseract_linux()
    
    # Test all components
    print("\n" + "=" * 50)
    print("🧪 Testing All Components...")
    all_working = test_components()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Fix Summary:")
    print(f"   Transformers: {'✅ Fixed' if transformers_fixed else '❌ Failed'}")
    print(f"   Environment: {'✅ Created' if env_created else '❌ Failed'}")
    print(f"   Tesseract: {'✅ Installed' if tesseract_installed else '❌ Failed'}")
    print(f"   Overall: {'✅ All Working' if all_working else '⚠️  Some Issues'}")
    
    if all_working:
        print("\n🎉 All critical components are now working!")
        print("You can now run the distributed APK tester with full AI capabilities.")
        print("\n📝 Note: Add your OpenAI API key to .env file if you want to use OpenAI features.")
    else:
        print("\n⚠️  Some components still need attention.")
        print("The system will work with fallback methods, but AI features may be limited.")

if __name__ == "__main__":
    main()
