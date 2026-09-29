#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLVM_BUILD_DEFAULT="$REPO_ROOT/rapid-llvm/build"
LLVM_BUILD="${LLVM_BUILD:-$LLVM_BUILD_DEFAULT}"
VCONFIG_BUILD_HELPER="$REPO_ROOT/tools/rapid-vconfig-instrument/build.py"

CHECK="${CHECK:-1}"
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --no-check) CHECK=0 ;;
    *) ARGS+=("$arg") ;;
  esac
done
set -- "${ARGS[@]}"

SRC="${1:-$SCRIPT_DIR/vdim_end2end.cu}"
OUT="${2:-$SCRIPT_DIR/bin/vdim_end2end}"
TMP="${TMP:-/tmp/vdim_end2end_build}"

if [[ -z "${CLANG:-}" ]]; then
  if [[ -x "$LLVM_BUILD/bin/clang" ]]; then
    CLANG="$LLVM_BUILD/bin/clang"
  else
    CLANG="/usr/bin/clang"
  fi
fi
if [[ -z "${OPT:-}" ]]; then
  if [[ -x "$LLVM_BUILD/bin/opt" ]]; then
    OPT="$LLVM_BUILD/bin/opt"
  else
    OPT="/usr/bin/opt"
  fi
fi
if [[ -z "${LLC:-}" ]]; then
  if [[ -x "$LLVM_BUILD/bin/llc" ]]; then
    LLC="$LLVM_BUILD/bin/llc"
  else
    LLC="/usr/bin/llc"
  fi
fi

NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
FATBINARY="${FATBINARY:-/usr/local/cuda/bin/fatbinary}"
VDIM_PLUGIN="${VDIM_PLUGIN:-$SCRIPT_DIR/bin/rapid-vconfig-pass-plugin.so}"
VCONFIG_TOOL="${VCONFIG_TOOL:-$SCRIPT_DIR/bin/rapid-vconfig-instrument}"
PYTHON="${PYTHON:-python3}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
ARCH="${CUDA_ARCH:-sm_80}"
SM="${ARCH#sm_}"

if [[ ! -f "$SRC" ]]; then
  echo "error: source not found: $SRC" >&2
  exit 1
fi
if [[ ! -x "$CLANG" ]]; then
  echo "error: clang not found/executable: $CLANG" >&2
  exit 1
fi
if [[ ! -x "$OPT" ]]; then
  echo "error: opt not found/executable: $OPT" >&2
  exit 1
fi
if [[ ! -x "$LLC" ]]; then
  echo "error: llc not found/executable: $LLC" >&2
  exit 1
fi
if [[ ! -x "$NVCC" ]]; then
  echo "error: nvcc not found/executable: $NVCC" >&2
  exit 1
fi
if [[ ! -x "$FATBINARY" ]]; then
  echo "error: fatbinary not found/executable: $FATBINARY" >&2
  exit 1
fi

mkdir -p "$TMP" "$(dirname "$OUT")" "$(dirname "$VDIM_PLUGIN")"
if [[ ! -f "$VDIM_PLUGIN" ]]; then
  "$PYTHON" "$VCONFIG_BUILD_HELPER" \
    --output "$VCONFIG_TOOL" \
    --plugin-output "$VDIM_PLUGIN"
fi

# 1) 生成 host/device IR -> 过 pass
"$CLANG" -x cuda --cuda-host-only -S -emit-llvm "$SRC" -o "$TMP/host.ll"
"$CLANG" -x cuda --cuda-device-only --cuda-path="$CUDA_PATH" \
  --cuda-gpu-arch="$ARCH" -S -emit-llvm "$SRC" -o "$TMP/device.ll"
"$OPT" -load-pass-plugin "$VDIM_PLUGIN" -passes=rapid-vconfig -S "$TMP/host.ll" -o "$TMP/host.vdim.ll"
"$OPT" -load-pass-plugin "$VDIM_PLUGIN" -passes=rapid-vconfig -S "$TMP/device.ll" -o "$TMP/device.vdim.ll"

# 2) host ll -> obj；device ll -> ptx
"$LLC" -filetype=obj -relocation-model=pic "$TMP/host.vdim.ll" -o "$TMP/host.o"
"$LLC" -march=nvptx64 -mcpu="$ARCH" "$TMP/device.vdim.ll" -o "$TMP/device.ptx"

# 3) ptx -> fatbin.c
"$FATBINARY" --64 --create="$TMP/device.fatbin" \
  --embedded-fatbin="$TMP/device.fatbin.c" \
  --image3=kind=ptx,sm="$SM",file="$TMP/device.ptx"

cat > "$TMP/vdim_register_min.cpp" <<'EOF'
#include <cstdlib>
#include "device.fatbin.c"

struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; };

extern "C" void **__cudaRegisterFatBinary(void *fatCubin);
extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun,
                                       char *deviceFun,
                                       const char *deviceName,
                                       int thread_limit,
                                       uint3 *tid, uint3 *bid,
                                       dim3 *bDim, dim3 *gDim,
                                       int *wSize);
extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle);

extern "C" void __device_stub__vdim_kernel(const float *, float *, int, int);

static void **__cudaFatCubinHandle;
static void __cudaUnregisterBinaryUtil(void) {
  __cudaUnregisterFatBinary(__cudaFatCubinHandle);
}

static void __vdim_register_all(void) __attribute__((constructor));
static void __vdim_register_all(void) {
  __cudaFatCubinHandle = __cudaRegisterFatBinary((void *)&__fatDeviceText);
  __cudaRegisterFunction(__cudaFatCubinHandle,
                         (const char *)&__device_stub__vdim_kernel,
                         (char *)"vdim_kernel",
                         "vdim_kernel",
                         -1, (uint3 *)0, (uint3 *)0, (dim3 *)0, (dim3 *)0,
                         (int *)0);
  __cudaRegisterFatBinaryEnd(__cudaFatCubinHandle);
  atexit(__cudaUnregisterBinaryUtil);
}
EOF

g++ -I/usr/local/cuda/targets/x86_64-linux/include -I"$TMP" \
  -c "$TMP/vdim_register_min.cpp" -o "$TMP/vdim_register_min.o"

# 4) host obj + fatbin 注册 glue -> 可执行文件
"$NVCC" -arch="$ARCH" "$TMP/host.o" "$TMP/vdim_register_min.o" -o "$OUT"

if [[ "$CHECK" -eq 1 ]]; then
  echo "Host stub smoke check:"
  if command -v rg >/dev/null 2>&1; then
    rg -n "vdim_set|__vdim_|select i1" "$TMP/host.vdim.ll"
  else
    grep -nE "vdim_set|__vdim_|select i1" "$TMP/host.vdim.ll"
  fi

  echo "Device guard smoke check:"
  if command -v rg >/dev/null 2>&1; then
    rg -n "vdim\\.inactive|getelementptr inbounds.*%struct\\.dim3" "$TMP/device.vdim.ll"
  else
    grep -nE "vdim\\.inactive|getelementptr inbounds.*%struct\\.dim3" "$TMP/device.vdim.ll"
  fi
fi

echo "Tip: one-step clang++ can use pass plugin:"
echo "  clang++ -fpass-plugin=$VDIM_PLUGIN \\"
echo "    --cuda-gpu-arch=$ARCH -std=c++17 $SRC -o $OUT"

echo "Built: $OUT"
