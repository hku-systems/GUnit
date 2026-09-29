cmake_minimum_required(VERSION 3.20)

# Enable C as well as CXX so CMake's built-in packages (notably Threads, which
# is pulled in by FindCUDAToolkit) can reliably run their feature/link checks.
project(cuda_kernel_harness LANGUAGES C CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RAPID_ENABLE_TIMING "Enable kernel timing stats collection" OFF)
option(RAPID_ENABLE_NVTX_PROFILING "Enable NVTX profiling markers" OFF)
option(RAPID_ENABLE_CANARY "Enable the L1 payload canary" ON)
option(RAPID_AUTO_DETECT_CUDA_ARCH "Auto-detect local GPU SM and set CMAKE_CUDA_ARCHITECTURES" ON)
# Where to place the final artifacts (.so, test_main). When this directory is
# added via add_subdirectory() from the repo root, this defaults to
# <top-level build>.
set(RAPID_KERNEL_OUTPUT_DIR
    "${CMAKE_BINARY_DIR}/cuda-kernel"
    CACHE PATH
          "Output directory for CUDA kernel shared libraries and test binaries"
)
# Optional override. If empty, we auto-detect from the local GPU using a CMake try_run.
set(RAPID_CUDA_ARCHITECTURES
    ""
    CACHE STRING
          "CUDA SM architectures (CMake numeric form, e.g. 86 or 75; empty = auto-detect)"
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(RapidDetectCudaArch)
rapid_configure_cuda_architectures()

# Explicit build summary for this subproject.
message(STATUS "=== CUDA Kernel Configure Summary ===")
if(CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Config types: ${CMAKE_CONFIGURATION_TYPES}")
else()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "Kernel output dir: ${RAPID_KERNEL_OUTPUT_DIR}")

# Place outputs under the chosen build directory (via -B ...), not in the source tree.
set(_rapid_output_dir "${RAPID_KERNEL_OUTPUT_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${_rapid_output_dir}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${_rapid_output_dir}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${_rapid_output_dir}")

set(_RAPID_PHASE2_EXPORT_DIR "${_rapid_output_dir}/builtin-phase-pipeline")
set(_RAPID_PHASE2_CUDA_ARCH "sm_86")
if(CMAKE_CUDA_ARCHITECTURES)
  list(GET CMAKE_CUDA_ARCHITECTURES 0 _rapid_first_cuda_arch)
  set(_RAPID_PHASE2_CUDA_ARCH "sm_${_rapid_first_cuda_arch}")
endif()
if(RAPID_ENABLE_CANARY)
  set(_RAPID_CANARY_MODE "enabled")
else()
  set(_RAPID_CANARY_MODE "disabled")
endif()
add_custom_target(
  builtin_phase_pipeline
  COMMAND
    ${Python3_EXECUTABLE}
    "${CMAKE_CURRENT_LIST_DIR}/builtin_phase_pipeline.py"
    --out-root
    "${_RAPID_PHASE2_EXPORT_DIR}"
    --cuda-arch
    "${_RAPID_PHASE2_CUDA_ARCH}"
    --canary
    "${_RAPID_CANARY_MODE}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  COMMENT "Running official Phase1/Phase2 for builtin kernel"
  VERBATIM)

add_executable(test_main test_main.cpp)
target_link_libraries(test_main PRIVATE dl)
set_target_properties(test_main PROPERTIES BUILD_RPATH "$ORIGIN")
