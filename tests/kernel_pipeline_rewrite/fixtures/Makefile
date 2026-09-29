OUT_DIR ?= $(CURDIR)/out
CLANG ?= clang++-22
LLVM_LINK ?= llvm-link-22
LLC ?= llc-22
CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH ?= sm_86
REPO_ROOT ?= $(abspath $(CURDIR)/../../..)
CUDA_KERNEL_UTILS_DIR ?= $(REPO_ROOT)/cuda-kernel/utils

PHASE2_INVOKE_HEADER ?=
PHASE2_DEVICE_BC ?=

WRAPPER_SRC := $(CURDIR)/wrapper.cu
HOST_SRC := $(CURDIR)/host.cpp
WRAPPER_BC := $(OUT_DIR)/wrapper.bc
LINKED_BC := $(OUT_DIR)/linked.bc
LINKED_PTX := $(OUT_DIR)/linked.ptx
HOST_EXE := $(OUT_DIR)/host.out

.PHONY: all clean

all: $(HOST_EXE)

$(OUT_DIR):
	mkdir -p "$@"

$(WRAPPER_BC): $(WRAPPER_SRC) | $(OUT_DIR)
	@test -n "$(PHASE2_INVOKE_HEADER)" || (echo "PHASE2_INVOKE_HEADER is required" >&2; exit 2)
	$(CLANG) -x cuda --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=$(CUDA_ARCH) --cuda-device-only -emit-llvm -c \
		-I"$(CUDA_KERNEL_UTILS_DIR)" \
		-DPHASE2_INVOKE_HEADER=\"$(PHASE2_INVOKE_HEADER)\" \
		"$(WRAPPER_SRC)" -o "$@"

$(LINKED_BC): $(WRAPPER_BC)
	@test -n "$(PHASE2_DEVICE_BC)" || (echo "PHASE2_DEVICE_BC is required" >&2; exit 2)
	$(LLVM_LINK) "$(WRAPPER_BC)" "$(PHASE2_DEVICE_BC)" -o "$@"

$(LINKED_PTX): $(LINKED_BC)
	$(LLC) -march=nvptx64 -mcpu=$(CUDA_ARCH) "$(LINKED_BC)" -o "$@"

$(HOST_EXE): $(HOST_SRC) $(LINKED_PTX)
	$(CLANG) "$(HOST_SRC)" -I$(CUDA_PATH)/targets/x86_64-linux/include -L/usr/lib/x86_64-linux-gnu -lcuda -o "$@"

clean:
	rm -rf "$(OUT_DIR)"
