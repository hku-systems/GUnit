cmake_minimum_required(VERSION 3.16)
project(vconfig LANGUAGES C CXX)

include(ExternalProject)

set(RAPID_LLVM_SRC_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../rapid-llvm/llvm"
    CACHE PATH "Path to rapid-llvm/llvm source directory")
set(RAPID_LLVM_BUILD_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../rapid-llvm/build"
    CACHE PATH "Path to rapid-llvm build directory")

set(LLVM_ENABLE_PROJECTS
    "clang"
    CACHE STRING "LLVM projects to enable")
set(LLVM_TARGETS_TO_BUILD
    "X86;NVPTX"
    CACHE STRING "LLVM targets to build")
set(LLVM_BUILD_TYPE
    "Release"
    CACHE STRING "LLVM build type")

string(REPLACE ";" "\\;" LLVM_ENABLE_PROJECTS_ESCAPED "${LLVM_ENABLE_PROJECTS}")
string(REPLACE ";" "\\;" LLVM_TARGETS_TO_BUILD_ESCAPED "${LLVM_TARGETS_TO_BUILD}")

if (EXISTS "${RAPID_LLVM_BUILD_DIR}/CMakeCache.txt")
  set(RAPID_LLVM_CONFIGURE_COMMAND
      "${CMAKE_COMMAND}" -E echo
      "Using existing rapid-llvm build dir: ${RAPID_LLVM_BUILD_DIR}")
else()
  set(RAPID_LLVM_CONFIGURE_COMMAND
      "${CMAKE_COMMAND}"
      -G "${CMAKE_GENERATOR}"
      -DLLVM_ENABLE_PROJECTS=${LLVM_ENABLE_PROJECTS_ESCAPED}
      -DLLVM_TARGETS_TO_BUILD=${LLVM_TARGETS_TO_BUILD_ESCAPED}
      -DCMAKE_BUILD_TYPE=${LLVM_BUILD_TYPE}
      "${RAPID_LLVM_SRC_DIR}")
endif()

ExternalProject_Add(rapid_llvm
  SOURCE_DIR "${RAPID_LLVM_SRC_DIR}"
  BINARY_DIR "${RAPID_LLVM_BUILD_DIR}"
  CONFIGURE_COMMAND ${RAPID_LLVM_CONFIGURE_COMMAND}
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  UPDATE_COMMAND ""
)

ExternalProject_Add_Step(rapid_llvm build_tools
  COMMAND "${CMAKE_COMMAND}" --build "${RAPID_LLVM_BUILD_DIR}" --target
          clang opt llc clang-linker-wrapper
  DEPENDEES build
  ALWAYS 1
)

add_custom_target(rapid_llvm_tools ALL
  DEPENDS rapid_llvm
)

set(VDIM_SRC "${CMAKE_CURRENT_LIST_DIR}/vdim_end2end.cu")
set(VDIM_OUT_DIR "${CMAKE_CURRENT_LIST_DIR}/bin")
set(VDIM_OUT "${VDIM_OUT_DIR}/vdim_end2end")

if (NOT DEFINED VDIM_CUDA_PATH)
  find_program(VDIM_NVCC nvcc HINTS /usr/local/cuda/bin)
  if (VDIM_NVCC)
    get_filename_component(_cuda_bin "${VDIM_NVCC}" DIRECTORY)
    get_filename_component(VDIM_CUDA_PATH "${_cuda_bin}" DIRECTORY)
  else()
    set(VDIM_CUDA_PATH "/usr/local/cuda")
  endif()
endif()
set(VDIM_CUDA_PATH "${VDIM_CUDA_PATH}" CACHE PATH "CUDA toolkit path")

if (NOT DEFINED VDIM_CUDA_ARCH)
  if (DEFINED ENV{CUDA_ARCH})
    set(VDIM_CUDA_ARCH "$ENV{CUDA_ARCH}")
  else()
    execute_process(
      COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _cc
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if (_cc)
      string(REPLACE "." "" _cc_nodot "${_cc}")
      string(PREPEND _cc_nodot "sm_")
      set(VDIM_CUDA_ARCH "${_cc_nodot}")
    else()
      set(VDIM_CUDA_ARCH "sm_80")
    endif()
  endif()
endif()
set(VDIM_CUDA_ARCH "${VDIM_CUDA_ARCH}" CACHE STRING "CUDA arch, e.g. sm_80")

set(VDIM_CLANGXX "${RAPID_LLVM_BUILD_DIR}/bin/clang++")
set(VDIM_PLUGIN "${VDIM_OUT_DIR}/rapid-vconfig-pass-plugin.so")
set(VDIM_TOOL "${VDIM_OUT_DIR}/rapid-vconfig-instrument")
set(VDIM_BUILD_HELPER
    "${CMAKE_CURRENT_LIST_DIR}/../../tools/rapid-vconfig-instrument/build.py")
find_program(VDIM_PYTHON python3 REQUIRED)

message(STATUS "VDIM_CUDA_PATH=${VDIM_CUDA_PATH}")
message(STATUS "VDIM_CUDA_ARCH=${VDIM_CUDA_ARCH}")

add_custom_target(vdim_end2end
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${VDIM_OUT_DIR}"
  COMMAND "${VDIM_PYTHON}" "${VDIM_BUILD_HELPER}"
          --output "${VDIM_TOOL}"
          --plugin-output "${VDIM_PLUGIN}"
  COMMAND "${VDIM_CLANGXX}" "${VDIM_SRC}"
          --cuda-path=${VDIM_CUDA_PATH}
          --cuda-gpu-arch=${VDIM_CUDA_ARCH}
          -I${VDIM_CUDA_PATH}/include
          -L${VDIM_CUDA_PATH}/lib64
          -lcudart
          -std=c++17
          -fpass-plugin=${VDIM_PLUGIN}
          -o "${VDIM_OUT}"
  COMMAND "${CMAKE_COMMAND}" -E echo "Built: ${VDIM_OUT}"
  DEPENDS rapid_llvm_tools
  USES_TERMINAL
  VERBATIM
)
