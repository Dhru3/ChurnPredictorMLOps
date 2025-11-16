#!/usr/bin/env python3
"""
Cleanup script to remove unnecessary files from the project
Run this once to clean up your repository
"""
import os
import shutil
from pathlib import Path

def cleanup():
    """Remove unnecessary files and folders"""
    
    files_to_remove = [
        'extract_model.py',
        '.env',
        '.python-version',
        '.DS_Store',
    ]
    
    folders_to_remove = [
        'mlruns',
        '__pycache__',
    ]
    
    print("🧹 Starting cleanup...\n")
    
    # Remove files
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"✅ Removed: {file}")
        else:
            print(f"⏭️  Skipped (not found): {file}")
    
    # Remove folders
    for folder in folders_to_remove:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"✅ Removed folder: {folder}/")
        else:
            print(f"⏭️  Skipped (not found): {folder}/")
    
    # Remove all .pyc files
    pyc_count = 0
    for pyc_file in Path('.').rglob('*.pyc'):
        pyc_file.unlink()
        pyc_count += 1
    
    if pyc_count > 0:
        print(f"✅ Removed {pyc_count} .pyc files")
    
    print("\n✨ Cleanup complete!")
    print("\n📝 Next steps:")
    print("1. Review the changes")
    print("2. git add -A")
    print("3. git commit -m 'Clean up unnecessary files and improve UI colors'")
    print("4. git push origin main")

if __name__ == "__main__":
    cleanup()