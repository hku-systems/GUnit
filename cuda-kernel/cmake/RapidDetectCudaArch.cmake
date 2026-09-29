include_guard(GLOBAL)

function(_rapid_try_detect_cuda_sm out_var)
  if(NOT DEFINED CUDAToolkit_INCLUDE_DIRS)
    message(FATAL_ERROR "CUDAToolkit must be found before CUDA SM detection")
  endif()

  # Find a concrete path to libcudart for try_run.
  set(_cudart_location "")
  foreach(_prop IN ITEMS
                   IMPORTED_LOCATION
                   IMPORTED_LOCATION_RELEASE
                   IMPORTED_LOCATION_RELWITHDEBINFO
                   IMPORTED_LOCATION_MINSIZEREL
                   IMPORTED_LOCATION_DEBUG
                   LOCATION)
    if(NOT _cudart_location)
      get_target_property(_cudart_location CUDA::cudart ${_prop})
    endif()
  endforeach()
  if(NOT _cudart_location)
    message(FATAL_ERROR "Failed to locate CUDA::cudart for CUDA SM detection")
  endif()
  get_filename_component(_cudart_dir "${_cudart_location}" DIRECTORY)

  set(_include_flags "")
  foreach(_inc IN LISTS CUDAToolkit_INCLUDE_DIRS)
    string(APPEND _include_flags " -I${_inc}")
  endforeach()
  string(STRIP "${_include_flags}" _include_flags)

  set(_rpath_flags "")
  if(EXISTS "${_cudart_location}" AND NOT _cudart_location MATCHES "\\.a$")
    set(_rpath_flags "-Wl,-rpath,${_cudart_dir}")
  endif()

  set(_src "
#include <cuda_runtime.h>
#include <cstdio>

int main() {
  // Force CUDA runtime initialization.
  cudaError_t st = cudaFree(nullptr);
  if (st != cudaSuccess) {
    std::fprintf(stderr, \"cudaFree(nullptr) failed: %s\\n\", cudaGetErrorString(st));
    return 10;
  }

  int count = 0;
  st = cudaGetDeviceCount(&count);
  if (st != cudaSuccess) {
    std::fprintf(stderr, \"cudaGetDeviceCount failed: %s\\n\", cudaGetErrorString(st));
    return 11;
  }
  if (count <= 0) {
    std::fprintf(stderr, \"No CUDA devices found\\n\");
    return 12;
  }

  int device = 0;
  cudaDeviceProp prop;
  st = cudaGetDeviceProperties(&prop, device);
  if (st != cudaSuccess) {
    std::fprintf(stderr, \"cudaGetDeviceProperties failed: %s\\n\", cudaGetErrorString(st));
    return 13;
  }
  std::printf(\"%d;%d\\n\", prop.major, prop.minor);
  return 0;
}
")

  set(_src_dir "${CMAKE_BINARY_DIR}/cmake_cuda_sm_detect_src")
  file(MAKE_DIRECTORY "${_src_dir}")
  set(_src_file "${_src_dir}/rapid_detect_cuda_sm.cpp")
  file(WRITE "${_src_file}" "${_src}")

  set(_exe_dir "${CMAKE_BINARY_DIR}/cmake_cuda_sm_detect")
  file(MAKE_DIRECTORY "${_exe_dir}")
  set(_exe "${_exe_dir}/rapid_detect_cuda_sm")

  separate_arguments(_include_flags_list NATIVE_COMMAND "${_include_flags}")
  set(_compile_cmd
      "${CMAKE_CXX_COMPILER}"
      "${_src_file}"
      "-o"
      "${_exe}"
      ${_include_flags_list}
      "${_cudart_location}")
  if(_rpath_flags)
    list(APPEND _compile_cmd "${_rpath_flags}")
  endif()

  execute_process(
    COMMAND ${_compile_cmd}
    RESULT_VARIABLE _compile_result
    OUTPUT_VARIABLE _compile_stdout
    ERROR_VARIABLE _compile_stderr)

  if(NOT _compile_result EQUAL 0)
    set(${out_var} "" PARENT_SCOPE)
    string(STRIP "${_compile_stderr}" _compile_stderr)
    if(_compile_stderr)
      string(REPLACE "\n" " | " _compile_stderr_one_line "${_compile_stderr}")
      string(SUBSTRING "${_compile_stderr_one_line}" 0 200 _compile_stderr_one_line)
      set(_rapid_cuda_sm_detect_error "compile_failed:${_compile_stderr_one_line}" PARENT_SCOPE)
    else()
      set(_rapid_cuda_sm_detect_error "compile_failed" PARENT_SCOPE)
    endif()
    return()
  endif()

  execute_process(
    COMMAND "${_exe}"
    RESULT_VARIABLE _run_result
    OUTPUT_VARIABLE _run_output
    ERROR_VARIABLE _run_stderr)

  if(NOT _run_result EQUAL 0)
    set(${out_var} "" PARENT_SCOPE)
    string(STRIP "${_run_stderr}" _run_stderr)
    if(_run_stderr)
      string(REPLACE "\n" " | " _run_stderr_one_line "${_run_stderr}")
      string(SUBSTRING "${_run_stderr_one_line}" 0 200 _run_stderr_one_line)
      set(_rapid_cuda_sm_detect_error "run_failed:${_run_stderr_one_line}" PARENT_SCOPE)
    else()
      set(_rapid_cuda_sm_detect_error "run_failed" PARENT_SCOPE)
    endif()
    return()
  endif()

  string(STRIP "${_run_output}" _run_output)
  # Expect: "major;minor"
  string(REGEX MATCH "^([0-9]+);([0-9]+)$" _match "${_run_output}")
  if(NOT _match)
    set(${out_var} "" PARENT_SCOPE)
    set(_rapid_cuda_sm_detect_error "parse_failed" PARENT_SCOPE)
    return()
  endif()

  set(_major "${CMAKE_MATCH_1}")
  set(_minor "${CMAKE_MATCH_2}")

  math(EXPR _arch "${_major} * 10 + ${_minor}")
  set(${out_var} "${_arch}" PARENT_SCOPE)
  set(_rapid_cuda_sm_detect_error "" PARENT_SCOPE)
endfunction()

function(rapid_configure_cuda_architectures)
  # Highest priority: user override.
  if(RAPID_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES
        "${RAPID_CUDA_ARCHITECTURES}"
        CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "CUDA arch: user override RAPID_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    return()
  endif()

  # If auto-detect is disabled, respect CMake's value (if any).
  if(DEFINED RAPID_AUTO_DETECT_CUDA_ARCH AND NOT RAPID_AUTO_DETECT_CUDA_ARCH)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
      message(STATUS "CUDA arch: auto-detect disabled; using CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
      return()
    endif()
  endif()

  # Preferred: runtime detect using CUDA Runtime API (no nvidia-smi).
  _rapid_try_detect_cuda_sm(_detected_arch)
  if(_detected_arch)
    set(CMAKE_CUDA_ARCHITECTURES
        "${_detected_arch}"
        CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "CUDA arch: detected local GPU SM=${CMAKE_CUDA_ARCHITECTURES}")
    return()
  endif()

  message(FATAL_ERROR
          "CUDA arch auto-detect failed (${_rapid_cuda_sm_detect_error}). "
          "This usually means the build environment cannot access the NVIDIA driver/GPU during CMake configure. "
          "Fix the environment (e.g. enable NVIDIA container runtime / proper driver mounts), or set -DRAPID_CUDA_ARCHITECTURES=<nn> (e.g. 86).")
endfunction()
