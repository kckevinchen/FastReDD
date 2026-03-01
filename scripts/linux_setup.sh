#!/usr/bin/env bash
set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_VERSION="3.10"
CONDA_ENV_NAME="redd"
INSTALL_DIR="$HOME/miniconda3"
ARCH="$(uname -m)"

if [[ "$ARCH" == "x86_64" ]]; then
  MINICONDA_PKG="Miniconda3-latest-Linux-x86_64.sh"
elif [[ "$ARCH" == "aarch64" ]]; then
  MINICONDA_PKG="Miniconda3-latest-Linux-aarch64.sh"
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi

MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_PKG}"

echo "Architecture: $ARCH"
echo "Python version: $PYTHON_VERSION"
echo "Conda environment: $CONDA_ENV_NAME"

# Check if Miniconda is already installed
if [[ -f "$INSTALL_DIR/bin/conda" ]]; then
  echo "Miniconda is already installed at $INSTALL_DIR, skipping installation..."
else
  # Download Miniconda
  cd /tmp
  if [[ ! -f "$MINICONDA_PKG" ]]; then
    wget "$MINICONDA_URL"
  fi

  # Install Miniconda
  bash "$MINICONDA_PKG" -b -p "$INSTALL_DIR"
fi

# Initialize conda
"$INSTALL_DIR/bin/conda" init bash

# Source conda
source "$INSTALL_DIR/etc/profile.d/conda.sh"

# Accept conda Terms of Service (required for non-interactive use)
echo "Accepting conda Terms of Service..."
"$INSTALL_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
"$INSTALL_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Create Python environment if it doesn't exist
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
  echo "Conda environment '$CONDA_ENV_NAME' already exists, skipping creation..."
else
  conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
fi

# Activate environment and install packages
conda activate "$CONDA_ENV_NAME"
pip install -U pip

# Install PyTorch
pip install torch torchvision torchaudio

# Install requirements
cd "$PROJECT_ROOT"
pip install -r requirements.txt

# Install package in development mode
pip install -e .

echo "Installation complete!"
echo "Activate environment: conda activate $CONDA_ENV_NAME"
