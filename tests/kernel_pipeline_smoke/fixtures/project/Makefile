# project/Makefile – builds mini_project from main.cu (includes kernels via header).
# No relocatable-device-code flags.
#
# Compiler selection: prefers nvcc; falls back to CUDA-capable clang++.

CUDA_COMPILER ?= $(shell if command -v nvcc >/dev/null 2>&1; then echo nvcc; elif command -v clang++ >/dev/null 2>&1 && echo 'int main(){return 0;}' | clang++ --cuda-path=$(CUDA_PATH) -xcuda - -c -o /dev/null >/dev/null 2>&1; then echo clang++; else echo NONE; fi)
CUDA_PATH     ?= /usr/local/cuda
CUDA_ARCH     ?= sm_86

TARGET := mini_project
SRC    := main.cu
DEPS   := kernels.cu kernels.cuh utils.cuh

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(DEPS)
	@if [ "$(CUDA_COMPILER)" = "NONE" ]; then \
		echo "ERROR: no CUDA compiler found (need clang++ or nvcc)" >&2; exit 1; \
	elif [ "$(notdir $(CUDA_COMPILER))" = "nvcc" ]; then \
		echo "[nvcc]  $< -> $@"; \
		$(CUDA_COMPILER) -O2 -o $@ $<; \
	else \
		echo "[clang] $< -> $@"; \
		$(CUDA_COMPILER) -O2 --cuda-gpu-arch=$(CUDA_ARCH) --cuda-path=$(CUDA_PATH) \
			-L$(CUDA_PATH)/lib64 -lcudart -o $@ $<; \
	fi

clean:
	rm -f $(TARGET)
