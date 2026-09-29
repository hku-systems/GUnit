BUILD_DIR ?= build
BUILD_TYPE ?= Release
CUDA_ARCH ?=

ENABLE_TIMING ?=
ENABLE_NVTX_PROFILING ?=

.PHONY: all configure kernels fuzzers clean help
.PHONY: compile_commands

all: configure
	@cmake --build "$(BUILD_DIR)" --target rapid_all -j

kernels: configure
	@cmake --build "$(BUILD_DIR)" --target rapid_kernels -j

fuzzers: configure
	@cmake --build "$(BUILD_DIR)" --target rapid_fuzzers -j

configure:
	@cmake -S . -B "$(BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE="$(BUILD_TYPE)" \
		$$(if [ -n "$(ENABLE_TIMING)" ] && [ "$(ENABLE_TIMING)" != "0" ]; then echo "-DRAPID_ENABLE_TIMING=ON"; else echo ""; fi) \
		$$(if [ -n "$(ENABLE_NVTX_PROFILING)" ] && [ "$(ENABLE_NVTX_PROFILING)" != "0" ]; then echo "-DRAPID_ENABLE_NVTX_PROFILING=ON"; else echo ""; fi) \
		$$(if [ -n "$(CUDA_ARCH)" ]; then echo "-DRAPID_CUDA_ARCHITECTURES=$(CUDA_ARCH)"; else echo ""; fi)

compile_commands:
	@cmake -S . -B "$(BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE="$(BUILD_TYPE)" \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		$$(if [ -n "$(ENABLE_TIMING)" ] && [ "$(ENABLE_TIMING)" != "0" ]; then echo "-DRAPID_ENABLE_TIMING=ON"; else echo ""; fi) \
		$$(if [ -n "$(ENABLE_NVTX_PROFILING)" ] && [ "$(ENABLE_NVTX_PROFILING)" != "0" ]; then echo "-DRAPID_ENABLE_NVTX_PROFILING=ON"; else echo ""; fi) \
		$$(if [ -n "$(CUDA_ARCH)" ]; then echo "-DRAPID_CUDA_ARCHITECTURES=$(CUDA_ARCH)"; else echo ""; fi)
	@echo "compile_commands.json: $(BUILD_DIR)/compile_commands.json"

clean:
	@rm -rf "$(BUILD_DIR)"

help:
	@echo "Usage: make [all|kernels|fuzzers|compile_commands] [BUILD_DIR=build] [BUILD_TYPE=Release] [CUDA_ARCH=<nn>]"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build kernels + fuzzers (default)"
	@echo "  kernels  - Build CUDA kernel shared libraries + test_main"
	@echo "  fuzzers  - Build Rust fuzzer binaries (cargo)"
	@echo "  compile_commands - Generate compile_commands.json via CMake"
	@echo "  clean    - Remove build directory"
	@echo ""
	@echo "Profile:"
	@echo "  BUILD_TYPE: passed to CMake as -DCMAKE_BUILD_TYPE (e.g. Release, Debug, RelWithDebInfo)"
	@echo "  ENABLE_TIMING: if non-empty and not 0, enables -DRAPID_ENABLE_TIMING=ON"
	@echo "  ENABLE_NVTX_PROFILING: if non-empty and not 0, enables -DRAPID_ENABLE_NVTX_PROFILING=ON"
	@echo ""
	@echo "CUDA:"
	@echo "  If CUDA_ARCH is set (e.g. 86), it is passed to CMake as -DRAPID_CUDA_ARCHITECTURES."
