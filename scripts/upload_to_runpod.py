#!/usr/bin/env python3
"""
Upload to RunPod Helper Script

This script helps prepare your codebase for upload to RunPod.
It creates a clean zip file excluding unnecessary files.
"""

import os
import zipfile
import shutil
import subprocess
from pathlib import Path

def create_runpod_package():
    """Create a clean package for RunPod upload."""
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    package_name = "AA_VA-Phi-RunPod"
    
    # Create temporary directory
    temp_dir = project_root / "temp_runpod_upload"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    # Files/directories to include
    include_patterns = [
        "src/",
        "scripts/",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        "DISTRIBUTED_SETUP_GUIDE.md",
        "RUNPOD_SETUP_GUIDE.md",
        "TESTING_GUIDE.md",
        "docs/",
        "*.ipynb"
    ]
    
    # Files/directories to exclude
    exclude_patterns = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".git",
        ".gitignore",
        "*.log",
        "test_reports",
        "distributed_test_reports",
        "logs",
        "*.png",
        "*.jpg",
        "*.jpeg",
        ".DS_Store",
        "Thumbs.db",
        "*.tmp",
        "*.temp",
        "temp_*",
        "node_modules",
        ".vscode",
        ".idea"
    ]
    
    print(f"📦 Creating RunPod package in {temp_dir}")
    
    # Copy files
    for pattern in include_patterns:
        if pattern.endswith("/"):
            # Directory
            src_dir = project_root / pattern[:-1]
            if src_dir.exists():
                dst_dir = temp_dir / pattern[:-1]
                shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(*exclude_patterns))
                print(f"  ✅ Copied directory: {pattern}")
        else:
            # File
            src_file = project_root / pattern
            if src_file.exists():
                dst_file = temp_dir / pattern
                shutil.copy2(src_file, dst_file)
                print(f"  ✅ Copied file: {pattern}")
    
    # Create zip file
    zip_path = project_root / f"{package_name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    
    print(f"🗜️  Creating zip file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    print(f"✅ Package created: {zip_path}")
    print(f"📏 Size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
    
    return zip_path

def print_upload_instructions():
    """Print instructions for uploading to RunPod."""
    
    print("\n" + "="*60)
    print("🚀 RUNPOD UPLOAD INSTRUCTIONS")
    print("="*60)
    
    print("\n1️⃣  **Upload to RunPod**:")
    print("   - Go to RunPod Jupyter Lab")
    print("   - Navigate to /workspace/")
    print("   - Upload the AA_VA-Phi-RunPod.zip file")
    print("   - Extract the zip file")
    
    print("\n2️⃣  **Install Dependencies**:")
    print("   cd /workspace/AA_VA-Phi")
    print("   pip install -r requirements.txt")
    print("   pip install fastapi uvicorn aiohttp requests")
    
    print("\n3️⃣  **Configure Local IP**:")
    print("   # On your local machine, get your IP:")
    print("   ifconfig | grep 'inet ' | grep -v 127.0.0.1")
    
    print("\n4️⃣  **Start Local ADB Server**:")
    print("   # On your local machine:")
    print("   python scripts/local_adb_server.py --host 0.0.0.0 --port 8000")
    
    print("\n5️⃣  **Test on RunPod**:")
    print("   # On RunPod:")
    print("   python scripts/quick_start_distributed.py --local-ip YOUR_LAPTOP_IP")
    
    print("\n📖 **For detailed instructions, see RUNPOD_SETUP_GUIDE.md**")
    print("="*60)

def main():
    """Main function."""
    
    print("🚀 AA_VA-Phi RunPod Upload Helper")
    print("="*40)
    
    try:
        # Create package
        zip_path = create_runpod_package()
        
        # Print instructions
        print_upload_instructions()
        
        print(f"\n🎉 Ready! Upload {zip_path.name} to RunPod")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
