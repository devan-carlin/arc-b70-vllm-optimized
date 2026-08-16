#!/usr/bin/env bash
# =============================================================================
# build-vllm-xpu.sh — Build vLLM (XPU) + vllm-xpu-kernels from source on an
# Intel Arc Pro B70 (Battlemage, Xe2) box, with the B70 patches applied.
#
# What it does:
#   1. Pre-flight checks (venv, oneAPI, GPUs, disk)
#   2. Installs PyTorch XPU
#   3. Clones vllm + vllm-xpu-kernels (main)
#   4. Builds vllm-xpu-kernels (SYCL, oneDNN + CUTLASS-SYCL)
#   5. Applies optional B70 patches (MoE TopK, mamba XPU ptr)
#   6. Builds vLLM, restores the local kernels, enforces triton-xpu
#   7. Verifies the install
#
# Usage:
#   bash build-vllm-xpu.sh                 # full build
#   bash build-vllm-xpu.sh --skip-clone    # reuse existing clones
#   bash build-vllm-xpu.sh --no-moe-patch  # skip the MoE TopK patch
#
# Env overrides:
#   VENV_DIR      virtualenv (default: $HOME/vllm-xpu/.venv)
#   BUILD_DIR     source tree (default: $HOME/vllm-xpu/vllm-src)
#   MAX_JOBS      kernel build parallelism (default: 4; raise if you have RAM)
# =============================================================================
set -uo pipefail

VENV_DIR="${VENV_DIR:-$HOME/vllm-xpu/.venv}"
BUILD_DIR="${BUILD_DIR:-$HOME/vllm-xpu/vllm-src}"
MAX_JOBS="${MAX_JOBS:-4}"
SKIP_CLONE=0
NO_MOE_PATCH=0
for a in "$@"; do
  case "$a" in
    --skip-clone) SKIP_CLONE=1 ;;
    --no-moe-patch) NO_MOE_PATCH=1 ;;
  esac
done

# Preserve HOME if run via sudo
if [ -n "${SUDO_USER:-}" ]; then
  export HOME=$(eval echo ~$SUDO_USER)
  export PIP_CACHE_DIR="/tmp/pip-cache-sudo"
fi

fail() { echo ""; echo "ERROR: $1"; echo ""; exit 1; }

echo "=========================================="
echo "  vLLM XPU Build — Pre-flight"
echo "=========================================="
command -v git >/dev/null 2>&1 || fail "git not installed"
[ -d "$VENV_DIR" ] || { echo "No venv at $VENV_DIR — creating one..."; python3 -m venv "$VENV_DIR"; }
source "$VENV_DIR/bin/activate"
echo "python: $(python --version)"

# oneAPI
if [ -f /opt/intel/oneapi/setvars.sh ]; then
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
  command -v icpx >/dev/null 2>&1 || fail "icpx not found after sourcing oneAPI"
  echo "icpx: $(icpx --version 2>&1 | head -1)"
else
  fail "oneAPI setvars.sh not found at /opt/intel/oneapi/"
fi

# GPUs
lspci | grep -qi "battlemage\|arc" || echo "WARNING: no Battlemage/Arc GPU detected by lspci"
gpu_count=$(lspci | grep -ci "battlemage\|arc" || true)
echo "GPUs detected: ${gpu_count:-?}"

# disk
available=$(df --output=avail -BM "$HOME" | tail -1 | awk '{printf "%d", $1/1024}')
[ "$available" -lt 50 ] && fail "Need 50GB free, have ${available}GB"
echo "Disk free: ${available}GB"

echo ""
echo "[1/6] Installing PyTorch XPU..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
python -c "import torch; print('torch', torch.__version__, 'xpu', torch.xpu.is_available(), 'devices', torch.xpu.device_count())"

echo ""
echo "[2/6] Cloning sources..."
mkdir -p "$BUILD_DIR"
if [ "$SKIP_CLONE" -eq 0 ]; then
  if [ -d "$BUILD_DIR/vllm-xpu-kernels/.git" ]; then
    (cd "$BUILD_DIR/vllm-xpu-kernels" && git fetch -q && git checkout -q main && git reset -q --hard origin/main)
  else
    git clone -q https://github.com/vllm-project/vllm-xpu-kernels.git "$BUILD_DIR/vllm-xpu-kernels"
  fi
  if [ -d "$BUILD_DIR/vllm/.git" ]; then
    (cd "$BUILD_DIR/vllm" && git fetch -q && git checkout -q main && git reset -q --hard origin/main)
  else
    git clone -q https://github.com/vllm-project/vllm.git "$BUILD_DIR/vllm"
  fi
else
  echo "  (skipping clone, reusing $BUILD_DIR)"
fi
echo "  vllm:            $(cd $BUILD_DIR/vllm && git rev-parse --short HEAD)"
echo "  vllm-xpu-kernels:$(cd $BUILD_DIR/vllm-xpu-kernels && git rev-parse --short HEAD)"

echo ""
echo "[3/6] Building vllm-xpu-kernels (this is the long step)..."
cd "$BUILD_DIR/vllm-xpu-kernels"
export VLLM_XPU_KERNELS_MAX_JOBS="$MAX_JOBS"
pip install -q setuptools setuptools-scm cmake ninja packaging psutil
pip install . --no-build-isolation 2>&1 | tail -15
echo "  kernels build exit: ${PIPESTATUS[0]}"

echo ""
echo "[4/6] Applying B70 patches..."
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/patches"
if [ "$NO_MOE_PATCH" -eq 0 ] && [ -f "$PATCH_DIR/moe-topk-16-32-64.patch" ]; then
  # MoE TopK patch is a git patch against vllm-xpu-kernels
  if git apply --check "$PATCH_DIR/moe-topk-16-32-64.patch" 2>/dev/null; then
    git apply "$PATCH_DIR/moe-topk-16-32-64.patch" && echo "  applied moe-topk-16-32-64.patch"
  else
    echo "  NOTE: moe-topk patch did not apply cleanly (upstream may have changed); skipping"
  fi
else
  echo "  (skipping MoE TopK patch)"
fi
# mamba XPU ptr patch applies to the vllm tree
if [ -f "$PATCH_DIR/mamba-xpu-ptr.patch" ]; then
  (cd "$BUILD_DIR/vllm" && git apply --check "$PATCH_DIR/mamba-xpu-ptr.patch" 2>/dev/null \
    && git apply "$PATCH_DIR/mamba-xpu-ptr.patch" && echo "  applied mamba-xpu-ptr.patch") \
    || echo "  NOTE: mamba patch did not apply cleanly; skipping"
fi

echo ""
echo "[5/6] Building vLLM engine..."
cd "$BUILD_DIR/vllm"
pip install -q setuptools-rust
pip install -r requirements-build.txt 2>/dev/null || pip install -r requirements/build.txt 2>/dev/null || true
export VLLM_TARGET_DEVICE="xpu"
pip install . --no-build-isolation 2>&1 | tail -15
echo "  vllm build exit: ${PIPESTATUS[0]}"

# vLLM's pyproject pins a pre-built vllm_xpu_kernels wheel, which overwrites the
# local build. Restore the local (patched) build.
cd "$BUILD_DIR/vllm-xpu-kernels"
pip install . --no-build-isolation --force-reinstall --no-deps 2>&1 | tail -3
echo "  kernels restore exit: ${PIPESTATUS[0]}"

# Enforce triton-xpu override (critical for the Intel backend)
pip install triton-xpu --index-url https://download.pytorch.org/whl/xpu --force-reinstall --no-deps 2>&1 | tail -2
python -c "import triton; print('triton from:', triton.__file__)"

# gguf plugin (only needed for GGUF models)
pip install --no-deps vllm-gguf-plugin 2>&1 | tail -1 || true

echo ""
echo "[6/6] Verifying..."
python -c "
import torch, vllm, vllm_xpu_kernels
print('vllm', vllm.__version__)
print('xpu devices', torch.xpu.device_count())
import vllm_xpu_kernels._C as c
print('gemm ops:', [o for o in dir(c) if 'gemm' in o])
"

echo ""
echo "=========================================="
echo "  Build complete."
echo "  Activate with:  source $VENV_DIR/bin/activate"
echo "=========================================="
