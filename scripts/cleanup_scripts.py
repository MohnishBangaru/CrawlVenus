#!/usr/bin/env python3
"""
Scripts Directory Cleanup

This script removes redundant and temporary scripts from the scripts directory,
keeping only the essential ones for the AA_VA-Phi project.
"""

import os
import shutil
from pathlib import Path

def get_script_info():
    """Get information about all scripts in the directory."""
    scripts_dir = Path(__file__).parent
    scripts = []
    
    for file in scripts_dir.iterdir():
        if file.is_file():
            size = file.stat().st_size / 1024  # KB
            scripts.append({
                'name': file.name,
                'size': size,
                'path': file
            })
    
    return scripts

def categorize_scripts():
    """Categorize scripts into keep, remove, and review."""
    
    # Essential scripts to keep
    essential_scripts = {
        'cleanup_repo.py',                    # Main cleanup script
        'distributed_apk_tester.py',          # Main APK tester
        'local_adb_server.py',                # ADB server for distributed setup
        'quick_start_distributed.py',         # Quick start for distributed testing
        'setup_api_keys.py',                  # API key setup
        'runpod_minimal_config.py',           # RunPod configuration
        'install_tesseract.py',               # Tesseract installation
        'setup_distributed_testing.py',       # Distributed testing setup
        'upload_to_runpod.py',                # RunPod upload utility
        'universal_apk_tester.py',            # Universal APK tester
        'final_report_generator.py',          # Report generation
        'test_phi_ground.py',                 # Phi Ground testing
        'runpod_ai_fix.py',                   # RunPod AI fixes
        'fix_critical_components.py',         # Critical component fixes
        'fix_runpod_dependencies.py',         # RunPod dependency fixes
        'emergency_fix_runpod.py',            # Emergency RunPod fixes
        'quick_fix_env.py',                   # Environment fixes
        'fix_test_directories.py',            # Test directory fixes
        'quick_fix_dotenv.py',                # Dotenv fixes
    }
    
    # Scripts to remove (redundant, temporary, or obsolete)
    remove_scripts = {
        # Vision fixes (redundant)
        'force_vision_support.py',
        'simple_vision_fix.py',
        'final_vision_fix.py',
        'direct_vision_fix.py',
        'vision_tokenization_fix.py',
        'fix_vision_tokenization.py',
        'fix_vision_support.py',
        'fix_vision_indentation.py',
        'debug_vision_tokenization.py',
        
        # Phi3 fixes (redundant)
        'simple_phi3_fix.py',
        'phi3_vision_fix.py',
        'fix_phi_ground_tokenizer.py',
        
        # Flash attention fixes (redundant)
        'enable_gpu.py',
        'clean_gpu_fix.py',
        'fix_flash_attention.py',
        'fix_flash_attention_comprehensive.py',
        'fix_flash_attention_compatibility.py',
        'install_flash_attention.py',
        'install_flash_attention_runpod.sh',
        'install_stable_flash.sh',
        'enable_flash_attention.sh',
        'quick_flash_fix.sh',
        'one_liner_fix.sh',
        'quick_model_fix.sh',
        
        # OmniParser fixes (redundant)
        'fix_omniparser.py',
        'fix_omniparser_syntax.py',
        'fix_omniparser_integration.py',
        'find_omniparser_source.py',
        'find_none_model.py',
        'monkey_patch_none_model.py',
        'quick_omniparser_fix.py',
        'fix_constructor.py',  # OmniParser constructor fix
        
        # Cache fixes (redundant)
        'fix_dynamic_cache.py',
        'fix_model_caching.py',
        'memory_cleanup.py',
        'clear_cache.sh',
        
        # Coordinate fixes (redundant)
        'debug_coordinates.py',
        'fix_coordinate_issue.py',
        
        # Device fixes (redundant)
        'simple_device_fix.py',
        
        # Debug scripts (temporary)
        'debug_phi_ground_action.py',
        'test_phi_vision.py',
        
        # Empty or broken files
        'fix_flash_attention.ipynb',
        
        # Jupyter notebooks (keep Python versions)
        'phi_ground_example.ipynb',
        'universal_apk_tester.ipynb',
        'test_phi_ground.ipynb',
        'final_report_generator.ipynb',
        'tester_v2.ipynb',
        
        # Example scripts (redundant with main functionality)
        'phi_ground_example.py',
        
        # Self-cleanup script (will be removed last)
        'cleanup_scripts.py',
    }
    
    return essential_scripts, remove_scripts

def cleanup_scripts(dry_run=True):
    """Clean up redundant scripts."""
    scripts_dir = Path(__file__).parent
    essential_scripts, remove_scripts = categorize_scripts()
    
    print("🧹 Scripts Directory Cleanup")
    print("=" * 50)
    
    # Get all scripts
    all_scripts = get_script_info()
    
    # Categorize existing scripts
    to_keep = []
    to_remove = []
    to_review = []
    
    total_size = 0
    removed_size = 0
    
    for script in all_scripts:
        total_size += script['size']
        
        if script['name'] in essential_scripts:
            to_keep.append(script)
        elif script['name'] in remove_scripts:
            to_remove.append(script)
            removed_size += script['size']
        else:
            to_review.append(script)
    
    print(f"📊 Script Analysis:")
    print(f"   Total scripts: {len(all_scripts)}")
    print(f"   Essential scripts: {len(to_keep)}")
    print(f"   Scripts to remove: {len(to_remove)}")
    print(f"   Scripts to review: {len(to_review)}")
    print(f"   Total size: {total_size:.1f}KB")
    print(f"   Size to remove: {removed_size:.1f}KB")
    print()
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be actually removed")
        print()
    
    # Show scripts to remove
    if to_remove:
        print("🗑️  Scripts to remove:")
        for script in sorted(to_remove, key=lambda x: x['name']):
            print(f"   {script['name']} ({script['size']:.1f}KB)")
        print()
    
    # Show scripts to review
    if to_review:
        print("🤔 Scripts to review (not categorized):")
        for script in sorted(to_review, key=lambda x: x['name']):
            print(f"   {script['name']} ({script['size']:.1f}KB)")
        print()
    
    # Show essential scripts
    print("✅ Essential scripts (keeping):")
    for script in sorted(to_keep, key=lambda x: x['name']):
        print(f"   {script['name']} ({script['size']:.1f}KB)")
    print()
    
    if not dry_run and to_remove:
        print("🗑️  Removing redundant scripts...")
        for script in to_remove:
            try:
                script['path'].unlink()
                print(f"   Removed: {script['name']}")
            except Exception as e:
                print(f"   Error removing {script['name']}: {e}")
        print()
        
        # Calculate final stats
        remaining_scripts = get_script_info()
        final_size = sum(s['size'] for s in remaining_scripts)
        space_saved = total_size - final_size
        
        print("=" * 50)
        print("📊 Cleanup Summary:")
        print(f"   Scripts removed: {len(to_remove)}")
        print(f"   Scripts remaining: {len(remaining_scripts)}")
        print(f"   Space saved: {space_saved:.1f}KB")
        print(f"   Reduction: {(space_saved/total_size)*100:.1f}%")
        print()
        print("✅ Scripts cleanup completed!")
    
    return to_remove, to_review, to_keep

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up scripts directory")
    parser.add_argument("--execute", action="store_true", help="Actually remove files (default is dry run)")
    parser.add_argument("--review-only", action="store_true", help="Only show scripts to review")
    
    args = parser.parse_args()
    
    if args.review_only:
        # Only show scripts that need review
        _, to_review, _ = cleanup_scripts(dry_run=True)
        if to_review:
            print("🤔 These scripts need manual review:")
            for script in sorted(to_review, key=lambda x: x['name']):
                print(f"   {script['name']} ({script['size']:.1f}KB)")
        else:
            print("✅ No scripts need review!")
        return
    
    # Perform cleanup
    cleanup_scripts(dry_run=not args.execute)

if __name__ == "__main__":
    main()
