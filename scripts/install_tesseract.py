#!/usr/bin/env python3
"""
Tesseract OCR Installation and Configuration Script
==================================================

This script helps install and configure Tesseract OCR for the vision engine.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def check_tesseract_installation():
    """Check if Tesseract is installed and working."""
    print("🔍 Checking Tesseract installation...")
    
    # Common Tesseract paths
    tesseract_paths = [
        "/opt/homebrew/bin/tesseract",  # macOS Homebrew
        "/usr/local/bin/tesseract",     # macOS/Linux
        "/usr/bin/tesseract",           # Linux
        "tesseract"                     # PATH
    ]
    
    for path in tesseract_paths:
        try:
            result = subprocess.run([path, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Tesseract found at: {path}")
                print(f"   Version: {result.stdout.strip()}")
                return path
        except Exception:
            continue
    
    print("❌ Tesseract not found")
    return None

def install_tesseract_macos():
    """Install Tesseract on macOS."""
    print("🍎 Installing Tesseract on macOS...")
    
    try:
        # Check if Homebrew is installed
        result = subprocess.run(["brew", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
        
        # Install Tesseract
        print("📦 Installing Tesseract via Homebrew...")
        result = subprocess.run(["brew", "install", "tesseract"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Tesseract installed successfully")
            return True
        else:
            print(f"❌ Failed to install Tesseract: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing Tesseract: {e}")
        return False

def install_tesseract_linux():
    """Install Tesseract on Linux."""
    print("🐧 Installing Tesseract on Linux...")
    
    try:
        # Try different package managers
        package_managers = [
            ("apt", ["sudo", "apt", "update", "&&", "sudo", "apt", "install", "-y", "tesseract-ocr"]),
            ("yum", ["sudo", "yum", "install", "-y", "tesseract"]),
            ("dnf", ["sudo", "dnf", "install", "-y", "tesseract"]),
            ("pacman", ["sudo", "pacman", "-S", "--noconfirm", "tesseract"])
        ]
        
        for manager, command in package_managers:
            try:
                result = subprocess.run(["which", manager], capture_output=True)
                if result.returncode == 0:
                    print(f"📦 Installing Tesseract via {manager}...")
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ Tesseract installed successfully")
                        return True
                    else:
                        print(f"❌ Failed to install via {manager}: {result.stderr}")
            except Exception:
                continue
        
        print("❌ No supported package manager found")
        return False
        
    except Exception as e:
        print(f"❌ Error installing Tesseract: {e}")
        return False

def install_tesseract_windows():
    """Install Tesseract on Windows."""
    print("🪟 Installing Tesseract on Windows...")
    
    print("📦 Please install Tesseract manually:")
    print("   1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   2. Install to C:\\Program Files\\Tesseract-OCR")
    print("   3. Add to PATH environment variable")
    print("   4. Restart your terminal")
    
    return False

def test_tesseract_python():
    """Test Tesseract with Python pytesseract."""
    print("\n🐍 Testing Tesseract with Python...")
    
    try:
        import pytesseract
        print("✅ pytesseract imported successfully")
        
        # Test Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
        # Test OCR on a simple image (if available)
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple test image
            test_image = Image.fromarray(np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8))
            
            # Test OCR
            text = pytesseract.image_to_string(test_image)
            print("✅ OCR test successful")
            
        except Exception as e:
            print(f"⚠️  OCR test failed: {e}")
        
        return True
        
    except ImportError:
        print("❌ pytesseract not installed")
        print("   Install with: pip install pytesseract")
        return False
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        return False

def configure_tesseract_path():
    """Configure Tesseract path for the vision engine."""
    print("\n⚙️  Configuring Tesseract path...")
    
    tesseract_path = check_tesseract_installation()
    if not tesseract_path:
        return False
    
    # Create environment configuration
    config_content = f"""# Tesseract Configuration
export TESSERACT_CMD="{tesseract_path}"
"""
    
    # Write to .env file
    env_file = Path(".env")
    if env_file.exists():
        # Append to existing .env file
        with open(env_file, "a") as f:
            f.write(f"\n# Tesseract Configuration\nTESSERACT_CMD={tesseract_path}\n")
    else:
        # Create new .env file
        with open(env_file, "w") as f:
            f.write(config_content)
    
    print(f"✅ Tesseract path configured: {tesseract_path}")
    print(f"   Added to .env file: TESSERACT_CMD={tesseract_path}")
    
    return True

def main():
    """Main function."""
    print("🔧 Tesseract OCR Installation and Configuration")
    print("=" * 50)
    
    # Check current installation
    tesseract_path = check_tesseract_installation()
    
    if not tesseract_path:
        print("\n📦 Tesseract not found. Installing...")
        
        system = platform.system().lower()
        if system == "darwin":
            success = install_tesseract_macos()
        elif system == "linux":
            success = install_tesseract_linux()
        elif system == "windows":
            success = install_tesseract_windows()
        else:
            print(f"❌ Unsupported operating system: {system}")
            success = False
        
        if success:
            tesseract_path = check_tesseract_installation()
    
    if tesseract_path:
        # Test Python integration
        test_tesseract_python()
        
        # Configure path
        configure_tesseract_path()
        
        print("\n✅ Tesseract setup completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Restart your Python environment")
        print("   2. Run the vision engine again")
        print("   3. OCR should now work properly")
    else:
        print("\n❌ Tesseract setup failed")
        print("\n📝 Manual installation:")
        print("   - macOS: brew install tesseract")
        print("   - Ubuntu/Debian: sudo apt install tesseract-ocr")
        print("   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")

if __name__ == "__main__":
    main()
