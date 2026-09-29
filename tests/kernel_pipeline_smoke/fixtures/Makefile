# Fixtures-level Makefile: canonical build entry point.
# Builds kernel object files from complex_kernels.cu, math_kernels.cu,
# ns_kernels.cu, and cublas_host_api_smoke.cu, then links them with
# fixture_runner.cu into the fixtures_runner binary.
#
# No relocatable-device-code flags are used anywhere in this file.
#
# Compiler selection: prefers nvcc; falls back to CUDA-capable clang++.
# Override via: make CUDA_COMPILER=nvcc
# Optional capture wrapper prefix: make WRAP_CMD=/abs/path/to/rapid-wrap

CUDA_COMPILER ?= $(shell if command -v nvcc >/dev/null 2>&1; then echo nvcc; elif command -v clang++ >/dev/null 2>&1 && echo 'int main(){return 0;}' | clang++ --cuda-path=$(CUDA_PATH) -xcuda - -c -o /dev/null >/dev/null 2>&1; then echo clang++; else echo NONE; fi)
CUDA_PATH     ?= /usr/local/cuda
CUDA_ARCH     ?= sm_86
WRAP_CMD      ?=

BUILD_DIR     := build-fixtures
FIXTURE_SRCS  := complex_kernels.cu math_kernels.cu ns_kernels.cu cublas_host_api_smoke.cu template_instantiation_kernels.cu
FIXTURE_OBJS  := $(addprefix $(BUILD_DIR)/,$(FIXTURE_SRCS:.cu=.o))
RUNNER_SRC    := fixture_runner.cu
RUNNER_BIN    := fixtures_runner
HEADERS       := common_helpers.cuh fixture_api.cuh scoped_header_types.cuh
LIB_ONLY_SRC  := lib_only_kernels.cu
LIB_ONLY_OBJ  := $(BUILD_DIR)/lib_only_kernels.o
LIB_ONLY_SO   := $(BUILD_DIR)/libonly_kernels.so
LIB_ONLY_A    := $(BUILD_DIR)/libonly_kernels.a

# Detect cuBLAS: require both header and library.
CUBLAS_AVAILABLE := $(shell test -f "$(CUDA_PATH)/include/cublas_v2.h" && (test -f "$(CUDA_PATH)/lib64/libcublas.so" || test -f "$(CUDA_PATH)/lib64/libcublas_static.a") && echo yes || echo no)
CUBLAS_CFLAGS    :=
CUBLAS_LDFLAGS   :=
ifeq ($(CUBLAS_AVAILABLE),yes)
  CUBLAS_CFLAGS  := -DHAVE_CUBLAS
  CUBLAS_LDFLAGS := -lcublas
endif

.PHONY: all compile_fixtures run clean mini_project lib_only

all: $(RUNNER_BIN) mini_project lib_only

mini_project:
	$(MAKE) -C project

compile_fixtures: $(FIXTURE_OBJS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile each kernel .cu into an object file (no RDC).
$(BUILD_DIR)/%.o: %.cu $(HEADERS) | $(BUILD_DIR)
	@if [ "$(CUDA_COMPILER)" = "NONE" ]; then \
		echo "ERROR: no CUDA compiler found (need clang++ or nvcc)" >&2; exit 1; \
	elif [ "$(notdir $(CUDA_COMPILER))" = "nvcc" ]; then \
		echo "[nvcc]  $< -> $@"; \
		$(WRAP_CMD) $(CUDA_COMPILER) -c -O2 $(CUBLAS_CFLAGS) -o $@ $<; \
	else \
		echo "[clang] $< -> $@"; \
		$(WRAP_CMD) $(CUDA_COMPILER) -c -O2 $(CUBLAS_CFLAGS) --cuda-gpu-arch=$(CUDA_ARCH) --cuda-path=$(CUDA_PATH) -o $@ $<; \
	fi

# Link runner binary from fixture_runner.cu + kernel object files.
$(RUNNER_BIN): $(FIXTURE_OBJS) $(RUNNER_SRC) $(HEADERS)
	@if [ "$(CUDA_COMPILER)" = "NONE" ]; then \
		echo "ERROR: no CUDA compiler found (need clang++ or nvcc)" >&2; exit 1; \
	elif [ "$(notdir $(CUDA_COMPILER))" = "nvcc" ]; then \
		echo "[nvcc]  linking $@"; \
		$(WRAP_CMD) $(CUDA_COMPILER) -O2 $(CUBLAS_CFLAGS) -o $@ $(RUNNER_SRC) $(FIXTURE_OBJS) $(CUBLAS_LDFLAGS); \
	else \
		echo "[clang] linking $@"; \
		$(WRAP_CMD) $(CUDA_COMPILER) -O2 $(CUBLAS_CFLAGS) --cuda-gpu-arch=$(CUDA_ARCH) --cuda-path=$(CUDA_PATH) \
			-L$(CUDA_PATH)/lib64 -lcudart $(CUBLAS_LDFLAGS) -o $@ $(RUNNER_SRC) $(FIXTURE_OBJS); \
	fi

# Library-only mode for capture/replay validation.
# Generates both a shared library (.so) and a static archive (.a)
# from a dedicated fixture source containing __global__ kernels.
lib_only: $(LIB_ONLY_SO) $(LIB_ONLY_A)

$(LIB_ONLY_SO): $(LIB_ONLY_SRC) $(HEADERS) | $(BUILD_DIR)
	@if [ "$(CUDA_COMPILER)" = "NONE" ]; then \
		echo "ERROR: no CUDA compiler found (need clang++ or nvcc)" >&2; exit 1; \
	elif [ "$(notdir $(CUDA_COMPILER))" = "nvcc" ]; then \
		echo "[nvcc]  shared $< -> $@"; \
		$(WRAP_CMD) $(CUDA_COMPILER) -O2 -shared -Xcompiler -fPIC -o $@ $<; \
	else \
		echo "[clang] shared $< -> $@"; \
		$(WRAP_CMD) $(CUDA_COMPILER) -O2 -fPIC -shared --cuda-gpu-arch=$(CUDA_ARCH) --cuda-path=$(CUDA_PATH) \
			-L$(CUDA_PATH)/lib64 -lcudart -o $@ $<; \
	fi

$(LIB_ONLY_OBJ): $(LIB_ONLY_SRC) $(HEADERS) | $(BUILD_DIR)
	@if [ "$(CUDA_COMPILER)" = "NONE" ]; then \
		echo "ERROR: no CUDA compiler found (need clang++ or nvcc)" >&2; exit 1; \
	elif [ "$(notdir $(CUDA_COMPILER))" = "nvcc" ]; then \
		echo "[nvcc]  static obj $< -> $@"; \
		$(WRAP_CMD) $(CUDA_COMPILER) -c -O2 -Xcompiler -fPIC -o $@ $<; \
	else \
		echo "[clang] static obj $< -> $@"; \
		$(WRAP_CMD) $(CUDA_COMPILER) -c -O2 -fPIC --cuda-gpu-arch=$(CUDA_ARCH) --cuda-path=$(CUDA_PATH) -o $@ $<; \
	fi

$(LIB_ONLY_A): $(LIB_ONLY_OBJ)
	ar rcs $@ $<

run: $(RUNNER_BIN)
	./$(RUNNER_BIN)

clean:
	rm -rf $(BUILD_DIR) $(RUNNER_BIN)
	$(MAKE) -C project clean
