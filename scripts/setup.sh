#!/bin/bash
set -e

# Install Python dependencies
if [ -f requirements.txt ]; then
  echo "Installing Python dependencies..."
  pip install --upgrade pip
  pip install -r requirements.txt
fi

# Install dev dependencies if available
if [ -f dev-requirements.txt ]; then
  echo "Installing development dependencies..."
  pip install -r dev-requirements.txt
fi

# Install system dependencies (example for Ubuntu/Debian)
if [ "$(uname)" = "Linux" ]; then
  echo "Installing system dependencies..."
  sudo apt-get update
  sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev
  sudo apt-get install -y docker.io
  sudo apt-get install -y nvidia-driver-525 || true
fi

echo "Setup complete." 