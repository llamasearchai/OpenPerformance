#!/usr/bin/env python3
"""Manual PyPI publication script."""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Publish package to PyPI."""
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Load PyPI token from .env
    from dotenv import load_dotenv
    load_dotenv()
    
    pypi_token = os.getenv("PYPI_API_KEY")
    if not pypi_token:
        print("Error: PYPI_API_KEY not found in .env file")
        sys.exit(1)
    
    # Clean previous builds
    print("Cleaning previous builds...")
    subprocess.run(["rm", "-rf", "dist/", "build/", "*.egg-info"], shell=True)
    
    # Build package
    print("Building package...")
    result = subprocess.run([sys.executable, "-m", "build"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        sys.exit(1)
    
    # Upload to PyPI
    print("Uploading to PyPI...")
    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = pypi_token.strip("'\"")
    
    result = subprocess.run(
        ["twine", "upload", "dist/*"],
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Upload failed: {result.stderr}")
        sys.exit(1)
    
    print("Successfully published to PyPI!")
    print(result.stdout)

if __name__ == "__main__":
    main()