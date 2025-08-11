#!/usr/bin/env bash
# Conda-based installation instructions for non-docker environments
set -e

# Initialize conda for the script
eval "$(conda shell.bash hook)"

# Extract the environment name from the environment.yml file
ENV_NAME="sglang_sssd"

# Check if the environment already exists
if conda env list | grep -q "^$ENV_NAME\s"; then
  echo "Environment '$ENV_NAME' already exists."
else
  echo "Environment '$ENV_NAME' does not exist. Creating it now..."
  # Some new Terms of service BS that conda introduced
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  conda create -n "$ENV_NAME" python=3.11 -y
fi

# Activate the environment
echo "Activating the '$ENV_NAME' environment..."
conda activate "$ENV_NAME"

# Install CUDA toolkit for SGLang
conda install -y -c nvidia -c conda-forge cuda-toolkit=12.6.3
# Install ninja for flashinfer (dependency for SGLang)
conda install -y -c conda-forge gcc=12.1.0 ninja
# Install SGLang
python3 -m pip install -e "python[all]"

conda install -y cmake

# Purge any existing SSSD installation
(cd sssd_speculator && pip uninstall sssd_speculator -y && rm -rf build/ && rm -rf *.egg-info/ && rm -rf dist/ && rm -f sssd_speculator/*.so)
# Install SSSD
(cd sssd_speculator && pip install -e . --config-settings editable_mode=compat)
