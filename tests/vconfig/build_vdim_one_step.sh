#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLVM_BUILD_DEFAULT="$REPO_ROOT/rapid-llvm/build"
LLVM_BUILD="${LLVM_BUILD:-$LLVM_BUILD_DEFAULT}"
VCONFIG_BUILD_HELPER="$REPO_ROOT/tools/rapid-vconfig-instrument/build.py"

SRC="${1:-$SCRIPT_DIR/vdim_end2end.cu}"
OUT="${2:-$SCRIPT_DIR/bin/vdim_end2end}"

if [[ -z "${CLANGXX:-}" ]]; then
  if [[ -x "$LLVM_BUILD/bin/clang++" ]]; then
    CLANGXX="$LLVM_BUILD/bin/clang++"
  else
    CLANGXX="/usr/bin/clang++"
  fi
fi
VDIM_PLUGIN="${VDIM_PLUGIN:-$SCRIPT_DIR/bin/rapid-vconfig-pass-plugin.so}"
VCONFIG_TOOL="${VCONFIG_TOOL:-$SCRIPT_DIR/bin/rapid-vconfig-instrument}"
PYTHON="${PYTHON:-python3}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
ARCH="${CUDA_ARCH:-sm_80}"

if [[ ! -f "$SRC" ]]; then
  echo "error: source not found: $SRC" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUT")" "$(dirname "$VDIM_PLUGIN")"
if [[ ! -f "$VDIM_PLUGIN" ]]; then
  "$PYTHON" "$VCONFIG_BUILD_HELPER" \
    --output "$VCONFIG_TOOL" \
    --plugin-output "$VDIM_PLUGIN"
fi
if [[ ! -x "$CLANGXX" ]]; then
  echo "error: clang++ not found/executable: $CLANGXX" >&2
  exit 1
fi

"$CLANGXX" "$SRC" \
  --cuda-path="$CUDA_PATH" \
  --cuda-gpu-arch="$ARCH" \
  -I"$CUDA_PATH/include" \
  -L"$CUDA_PATH/lib64" \
  -lcudart \
  -std=c++17 \
  -fpass-plugin="$VDIM_PLUGIN" \
  -o "$OUT"

echo "Built: $OUT"
