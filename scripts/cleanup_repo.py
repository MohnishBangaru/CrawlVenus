#!/usr/bin/env python3
"""
Repository Cleanup Script

This script removes all clutter from the AA_VA-Phi repository including:
- Screenshot files
- Log files
- Cache directories
- Temporary files
- Test outputs
"""

import os
import shutil
import glob
from pathlib import Path
import argparse

def get_directory_size(path):
    """Get the size of a directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

def cleanup_screenshots():
    """Remove all screenshot files."""
    print("🗑️  Cleaning up screenshot files...")
    
    # Find all screenshot files
    screenshot_patterns = [
        "screenshot_*.png",
        "*.png",
        "*.jpg",
        "*.jpeg"
    ]
    
    removed_count = 0
    total_size = 0
    
    for pattern in screenshot_patterns:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            # Skip files in .git directory
            if '.git' in file:
                continue
                
            try:
                size = os.path.getsize(file) / (1024 * 1024)  # MB
                os.remove(file)
                removed_count += 1
                total_size += size
                print(f"   Removed: {file} ({size:.1f}MB)")
            except Exception as e:
                print(f"   Error removing {file}: {e}")
    
    print(f"✅ Removed {removed_count} screenshot files ({total_size:.1f}MB)")

def cleanup_logs():
    """Remove log files and directories."""
    print("🗑️  Cleaning up log files...")
    
    log_dirs = [
        "logs",
        "LLM_logs",
        "*.log"
    ]
    
    removed_count = 0
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            try:
                if os.path.isdir(log_dir):
                    shutil.rmtree(log_dir)
                    print(f"   Removed directory: {log_dir}")
                else:
                    os.remove(log_dir)
                    print(f"   Removed file: {log_dir}")
                removed_count += 1
            except Exception as e:
                print(f"   Error removing {log_dir}: {e}")
    
    # Remove individual log files
    log_files = glob.glob("*.log", recursive=True)
    for log_file in log_files:
        try:
            os.remove(log_file)
            print(f"   Removed: {log_file}")
            removed_count += 1
        except Exception as e:
            print(f"   Error removing {log_file}: {e}")
    
    print(f"✅ Removed {removed_count} log files/directories")

def cleanup_cache():
    """Remove cache directories and files."""
    print("🗑️  Cleaning up cache files...")
    
    cache_patterns = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        "*.cache"
    ]
    
    removed_count = 0
    
    for pattern in cache_patterns:
        if os.path.exists(pattern):
            try:
                if os.path.isdir(pattern):
                    shutil.rmtree(pattern)
                    print(f"   Removed cache directory: {pattern}")
                else:
                    os.remove(pattern)
                    print(f"   Removed cache file: {pattern}")
                removed_count += 1
            except Exception as e:
                print(f"   Error removing {pattern}: {e}")
    
    # Remove __pycache__ directories recursively
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            cache_dir = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(cache_dir)
                print(f"   Removed: {cache_dir}")
                removed_count += 1
            except Exception as e:
                print(f"   Error removing {cache_dir}: {e}")
    
    print(f"✅ Removed {removed_count} cache files/directories")

def cleanup_test_outputs():
    """Remove test output directories."""
    print("🗑️  Cleaning up test outputs...")
    
    test_dirs = [
        "test_reports",
        "test_outputs",
        "test_results",
        "reports",
        "outputs",
        "results",
        "screenshots",
        "vision_debug",
        "ocr_images"
    ]
    
    removed_count = 0
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"   Removed test directory: {test_dir}")
                removed_count += 1
            except Exception as e:
                print(f"   Error removing {test_dir}: {e}")
    
    print(f"✅ Removed {removed_count} test output directories")

def cleanup_temporary_files():
    """Remove temporary files."""
    print("🗑️  Cleaning up temporary files...")
    
    temp_patterns = [
        "*.tmp",
        "*.temp",
        "*.swp",
        "*.swo",
        "*~",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    removed_count = 0
    
    for pattern in temp_patterns:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            try:
                os.remove(file)
                print(f"   Removed: {file}")
                removed_count += 1
            except Exception as e:
                print(f"   Error removing {file}: {e}")
    
    print(f"✅ Removed {removed_count} temporary files")

def cleanup_apk_files():
    """Remove APK files."""
    print("🗑️  Cleaning up APK files...")
    
    apk_files = glob.glob("*.apk", recursive=True)
    removed_count = 0
    total_size = 0
    
    for apk_file in apk_files:
        try:
            size = os.path.getsize(apk_file) / (1024 * 1024)  # MB
            os.remove(apk_file)
            print(f"   Removed: {apk_file} ({size:.1f}MB)")
            removed_count += 1
            total_size += size
        except Exception as e:
            print(f"   Error removing {apk_file}: {e}")
    
    print(f"✅ Removed {removed_count} APK files ({total_size:.1f}MB)")

def cleanup_zip_files():
    """Remove ZIP files."""
    print("🗑️  Cleaning up ZIP files...")
    
    zip_files = glob.glob("*.zip", recursive=True)
    removed_count = 0
    total_size = 0
    
    for zip_file in zip_files:
        try:
            size = os.path.getsize(zip_file) / (1024 * 1024)  # MB
            os.remove(zip_file)
            print(f"   Removed: {zip_file} ({size:.1f}MB)")
            removed_count += 1
            total_size += size
        except Exception as e:
            print(f"   Error removing {zip_file}: {e}")
    
    print(f"✅ Removed {removed_count} ZIP files ({total_size:.1f}MB)")

def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description="Clean up AA_VA-Phi repository")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually removing")
    parser.add_argument("--skip-screenshots", action="store_true", help="Skip removing screenshot files")
    parser.add_argument("--skip-logs", action="store_true", help="Skip removing log files")
    parser.add_argument("--skip-cache", action="store_true", help="Skip removing cache files")
    parser.add_argument("--skip-tests", action="store_true", help="Skip removing test outputs")
    parser.add_argument("--skip-temp", action="store_true", help="Skip removing temporary files")
    parser.add_argument("--skip-apk", action="store_true", help="Skip removing APK files")
    parser.add_argument("--skip-zip", action="store_true", help="Skip removing ZIP files")
    
    args = parser.parse_args()
    
    print("🧹 AA_VA-Phi Repository Cleanup")
    print("=" * 50)
    
    # Get initial size
    initial_size = get_directory_size(".")
    print(f"📊 Initial repository size: {initial_size:.1f}MB")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be actually removed")
        print()
        return
    
    # Perform cleanup
    if not args.skip_screenshots:
        cleanup_screenshots()
        print()
    
    if not args.skip_logs:
        cleanup_logs()
        print()
    
    if not args.skip_cache:
        cleanup_cache()
        print()
    
    if not args.skip_tests:
        cleanup_test_outputs()
        print()
    
    if not args.skip_temp:
        cleanup_temporary_files()
        print()
    
    if not args.skip_apk:
        cleanup_apk_files()
        print()
    
    if not args.skip_zip:
        cleanup_zip_files()
        print()
    
    # Get final size
    final_size = get_directory_size(".")
    space_saved = initial_size - final_size
    
    print("=" * 50)
    print("📊 Cleanup Summary:")
    print(f"   Initial size: {initial_size:.1f}MB")
    print(f"   Final size: {final_size:.1f}MB")
    print(f"   Space saved: {space_saved:.1f}MB")
    print(f"   Reduction: {(space_saved/initial_size)*100:.1f}%")
    print()
    print("✅ Repository cleanup completed!")

if __name__ == "__main__":
    main()
