#!/bin/bash
#  KATE - Clean install
#  Usage:  bash install_kate.sh
#  This script:
#    1. Locates conda and deactivates any active env
#    2. Removes existing kate env (if any)
#    3. Creates a fresh kate env
#    4. Installs conda dependencies (mdtraj, deeptime, matplotlib)
#    5. Installs pip dependencies (torch)
#    6. Builds and installs KATE from wheel
#    7. Runs tests to verify
set -eo pipefail
ENV_NAME="kate"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo ""
echo "  KATE - Clean install"
echo "  Date: $(date)"
# 1. Locate conda. A non interactive shell may not source the user rc file, so
# conda can be a shell function invisible to this script even on a machine that
# has it; the parent shell exports CONDA_EXE, which covers that case.
echo ""
echo "[1/7] Locating conda."
if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
elif [ -n "$CONDA_EXE" ] && [ -x "$CONDA_EXE" ]; then
    CONDA_BIN="$CONDA_EXE"
else
    echo "ERROR: conda not found. Install Miniforge or Miniconda first."
    exit 1
fi
eval "$("$CONDA_BIN" shell.bash hook)"
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
echo "Using conda: $CONDA_BIN"
# 2. Remove existing kate env (if found)
echo ""
echo "[2/7] Removing existing '$ENV_NAME' environment."
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
echo "Clean slate"
# 3. Create fresh env
echo ""
echo "[3/7] Creating fresh conda env: $ENV_NAME (Python $PYTHON_VERSION)."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME"
# The env lands wherever this conda keeps its envs, so take the interpreter
# from the activated prefix instead of assuming a fixed path.
ENV_PY="$CONDA_PREFIX/bin/python"
if [ "$(basename "$CONDA_PREFIX")" != "$ENV_NAME" ] || [ ! -x "$ENV_PY" ]; then
    echo "ERROR: activation did not land in the '$ENV_NAME' env (CONDA_PREFIX=$CONDA_PREFIX)."
    exit 1
fi
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
rm -rf "$SCRIPT_DIR/dist"
"$ENV_PY" -m build --wheel
WHEEL="$(ls "$SCRIPT_DIR"/dist/kate-*.whl 2>/dev/null | head -n 1)"
if [ -z "$WHEEL" ]; then
    echo "ERROR: no wheel found in $SCRIPT_DIR/dist."
    echo "Make sure you run this from the KATE directory."
    exit 1
fi
# The env was created fresh above, so --force-reinstall would only rebuild the
# already-correct dependency tree, which for torch means re-fetching several GB
# of CUDA wheels.
"$ENV_PY" -m pip install "${WHEEL}[kinetics,test]"
echo "KATE installed"
# 7. Verify. These probes are informational, so a missing __version__ on some
# dependency reports itself rather than aborting an install that succeeded.
echo ""
echo "[7/7] Verifying installation."
echo ""
"$ENV_PY" - <<'PY' || echo "WARNING: a version probe failed, see above"
import importlib
for mod in ("kate", "numpy", "scipy", "sklearn", "torch", "mdtraj", "deeptime"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod} {getattr(m, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{mod} NOT AVAILABLE ({exc})")
PY
command -v kate >/dev/null 2>&1 && echo "kate CLI on PATH" || echo "kate CLI not found"
# Run tests. The suite decides whether the install is good, so its exit status is
# reported explicitly instead of being swallowed by the pipe to tail. MDTraj's DCD
# reader writes to the same descriptor from C stdio and flushes at exit, which
# overwrites the pytest summary line, so the verdict comes from the exit status and
# the log is filtered down to the lines that carry information.
echo ""
echo "Running tests."
cd "$SCRIPT_DIR"
set +e
"$ENV_PY" -m pytest tests/ -q --tb=short > "$SCRIPT_DIR/install_pytest.log" 2>&1
STATUS=$?
set -e
# grep returns 1 when every line is filtered out, which under pipefail would abort
# the script just before it reports the verdict.
{ grep -avE "dcdplugin\)|^$" "$SCRIPT_DIR/install_pytest.log" || true; } | tail -12
echo ""
if [ "$STATUS" -eq 0 ]; then
    echo "  Test suite passed."
    echo "  Installation complete!"
    rm -f "$SCRIPT_DIR/install_pytest.log"
else
    echo "  Installation finished but the test suite failed (pytest exit $STATUS)."
    echo "  Full log: $SCRIPT_DIR/install_pytest.log"
    exit "$STATUS"
fi
echo ""
