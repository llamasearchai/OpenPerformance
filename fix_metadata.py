#!/usr/bin/env python3
"""Fix metadata column names in models."""

import os
import re
from pathlib import Path

def fix_metadata(file_path):
    """Fix metadata column in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace metadata = Column with data_metadata = Column
    original_content = content
    content = re.sub(r'(\s+)metadata = Column', r'\1data_metadata = Column', content)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed metadata in {file_path}")
        return True
    return False

def main():
    """Fix all metadata columns in the project."""
    models_dir = Path("python/mlperf/models")
    fixed_count = 0
    
    for py_file in models_dir.rglob("*.py"):
        if fix_metadata(py_file):
            fixed_count += 1
    
    print(f"\nFixed metadata in {fixed_count} files")

if __name__ == "__main__":
    main()