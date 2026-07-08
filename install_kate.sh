#!/bin/bash
#  KATE - Clean install
#  Usage:  bash install_kate.sh
#  This script:
#    1. Deactivates any active conda env
#    2. Removes existing kate env (if any)
#    3. Creates a fresh kate env
#    4. Installs conda dependencies (mdtraj, deeptime, matplotlib)
#    5. Installs pip dependencies (torch)
#    6. Builds and installs KATE from wheel
#    7. Runs tests to verify
set -e
ENV_NAME="kate"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo ""
echo "  KATE - Clean install"
echo "  Date: $(date)"
# 1. Deactivate current env
echo ""
echo "[1/7] Deactivating current conda environment."
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
echo " Deactivated"
# 2. Remove existing kate env (if found)
echo ""
echo "[2/7] Removing existing '$ENV_NAME' environment."
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
rm -rf "$HOME/.conda/envs/$ENV_NAME" 2>/dev/null || true
echo "Clean slate"
# 3. Create fresh env
echo ""
echo "[3/7] Creating fresh conda env: $ENV_NAME (Python $PYTHON_VERSION)."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
# Pin to the env's interpreter by absolute path so PATH/activation quirks
# cannot redirect the install to another interpreter.
ENV_PY="$HOME/.conda/envs/$ENV_NAME/bin/python"
echo "Created and activated: $("$ENV_PY" --version)"
echo "Using interpreter: $ENV_PY"
"$ENV_PY" -m pip --version
# 4. Conda dependencies
echo ""
echo "[4/7] Installing conda dependencies (mdtraj, deeptime, matplotlib)."
conda install -c conda-forge mdtraj deeptime matplotlib -y
echo "conda dependencies installed"
# 5. pip dependencies
echo ""
echo "[5/7] Installing pip dependencies (torch)."
"$ENV_PY" -m pip install torch
echo "pip dependencies installed"
# 6. Build and install KATE
echo ""
echo "[6/7] Building and installing KATE."
cd "$SCRIPT_DIR"
"$ENV_PY" -m pip install build
"$ENV_PY" -m build --wheel
WHEEL="$SCRIPT_DIR/dist/kate-0.1.0-py3-none-any.whl"
if [ ! -f "$WHEEL" ]; then
    echo "ERROR: Wheel not found: $WHEEL"
    echo "Make sure you run this from the KATE directory."
    exit 1
fi
"$ENV_PY" -m pip install "${WHEEL}[kinetics,test]" --force-reinstall
echo "KATE installed"
# 7. Verify
echo ""
echo "[7/7] Verifying installation."
echo ""
"$ENV_PY" -c "import kate; print(f'KATE {kate.__version__}')"
"$ENV_PY" -c "import numpy; print(f'NumPy {numpy.__version__}')"
"$ENV_PY" -c "import scipy; print(f'SciPy {scipy.__version__}')"
"$ENV_PY" -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
"$ENV_PY" -c "import torch; print(f'PyTorch {torch.__version__}')"
"$ENV_PY" -c "import mdtraj; print(f'MDTraj {mdtraj.version.version}')"
"$ENV_PY" -c "import deeptime; print(f'deeptime {deeptime.__version__}')" 2>/dev/null || echo "deeptime not available"
which kate >/dev/null 2>&1 && echo "kate CLI on PATH" || echo "kate CLI not found"
# Run tests
echo ""
echo "Running tests."
cd "$SCRIPT_DIR"
"$ENV_PY" -m pytest tests/ -q --tb=short 2>&1 | tail -3
echo ""
echo "  Installation complete!"
echo ""
