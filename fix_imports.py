#!/usr/bin/env python3
"""Fix imports from python.mlperf to mlperf."""

import os
import re
from pathlib import Path

def fix_imports(file_path):
    """Fix imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace python.mlperf with mlperf
    original_content = content
    content = re.sub(r'from python\.mlperf', 'from mlperf', content)
    content = re.sub(r'import python\.mlperf', 'import mlperf', content)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed imports in {file_path}")
        return True
    return False

def main():
    """Fix all imports in the project."""
    python_dir = Path("python")
    fixed_count = 0
    
    for py_file in python_dir.rglob("*.py"):
        if fix_imports(py_file):
            fixed_count += 1
    
    print(f"\nFixed imports in {fixed_count} files")

if __name__ == "__main__":
    main()