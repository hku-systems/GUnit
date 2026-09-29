#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPO_ROOT/rapid-llvm/llvm"
BUILD_DIR="${LLVM_BUILD_DIR:-$REPO_ROOT/rapid-llvm/build}"

PROJECTS="${LLVM_ENABLE_PROJECTS:-clang}"
TARGETS="${LLVM_TARGETS_TO_BUILD:-X86;NVPTX}"
BUILD_TYPE="${LLVM_BUILD_TYPE:-Release}"

GENERATOR="${CMAKE_GENERATOR:-}"
if [[ -z "$GENERATOR" ]]; then
  if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
  else
    GENERATOR="Unix Makefiles"
  fi
fi

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G "$GENERATOR" \
  -DLLVM_ENABLE_PROJECTS="$PROJECTS" \
  -DLLVM_TARGETS_TO_BUILD="$TARGETS" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

cmake --build "$BUILD_DIR" --target \
  clang \
  opt \
  llc \
  clang-linker-wrapper

echo "Built LLVM tools in: $BUILD_DIR"
