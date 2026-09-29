cmake_minimum_required(VERSION 3.20)

# Enable C as well as CXX so CMake's built-in packages (notably Threads, which
# is pulled in by FindCUDAToolkit) can reliably run their feature/link checks.
project(rapid LANGUAGES C CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
endif()

option(RAPID_BUILD_KERNELS "Build CUDA kernel shared libraries" ON)
option(RAPID_BUILD_FUZZERS "Build Rust fuzzer binaries via cargo" ON)

# Print a short, explicit configure summary (especially useful to confirm
# Debug/Release when using single-config generators).
message(STATUS "=== RAPID Configure Summary ===")
message(STATUS "CMake generator: ${CMAKE_GENERATOR}")
if(CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Config types: ${CMAKE_CONFIGURATION_TYPES}")
else()
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# These options/vars are consumed by cuda-kernel/CMakeLists.txt.
option(RAPID_ENABLE_TIMING "Enable kernel timing stats collection" OFF)
option(RAPID_ENABLE_NVTX_PROFILING "Enable NVTX profiling markers (mainly useful in Debug/profiling builds)" OFF)
option(RAPID_AUTO_DETECT_CUDA_ARCH "Auto-detect local GPU SM and set CMAKE_CUDA_ARCHITECTURES" ON)
set(RAPID_CUDA_ARCHITECTURES
    ""
    CACHE STRING
          "CUDA SM architectures (CMake numeric form, e.g. 86 or 75; empty = auto-detect)"
)

if(RAPID_BUILD_KERNELS)
  set(RAPID_KERNEL_OUTPUT_DIR
      "${CMAKE_BINARY_DIR}/cuda-kernel"
      CACHE PATH
            "Output directory for CUDA kernel shared libraries and test binaries"
            FORCE)
  message(STATUS "Kernel output dir: ${RAPID_KERNEL_OUTPUT_DIR}")
  add_subdirectory(cuda-kernel)

  add_custom_target(rapid_kernels DEPENDS builtin_phase_pipeline test_main)
endif()

if(RAPID_BUILD_FUZZERS)
  find_program(RAPID_CARGO_EXECUTABLE cargo REQUIRED)

  set(_rapid_fuzzer_dir "${CMAKE_SOURCE_DIR}/cuda-fuzzer")
  # Place cargo outputs under the CMake build directory, without an extra
  # "cargo-target" prefix, so binaries end up at:
  #   <build>/cuda-fuzzer/<debug|release>/{fuzzer,fuzzer_async}
  set(_rapid_cargo_target_dir "${CMAKE_BINARY_DIR}/cuda-fuzzer")
  message(STATUS "Cargo target dir: ${_rapid_cargo_target_dir}")

  set(_rapid_cargo_env "CARGO_TARGET_DIR=${_rapid_cargo_target_dir}")

  # Map CMake build types to cargo profiles.
  #
  # Note: `cargo build --release` defaults to `debuginfo=0`. If you configure
  # with `-DCMAKE_BUILD_TYPE=RelWithDebInfo`, we explicitly enable DWARF debug
  # info for the Rust release profile so `gdb` can symbolicate optimized builds.
  if(CMAKE_BUILD_TYPE MATCHES "^[Rr]elWithDebInfo$")
    list(APPEND _rapid_cargo_env
         "CARGO_PROFILE_RELEASE_DEBUG=2"
         "CARGO_PROFILE_RELEASE_STRIP=none")
    set(_rapid_cargo_profile --release)
    set(_rapid_cargo_profile_dir relwithdebinfo)
  elseif(CMAKE_BUILD_TYPE MATCHES "^[Rr]el")
    set(_rapid_cargo_profile --release)
    set(_rapid_cargo_profile_dir release)
  else()
    set(_rapid_cargo_profile "")
    set(_rapid_cargo_profile_dir debug)
  endif()
  message(STATUS "Cargo profile: ${_rapid_cargo_profile_dir}")

  add_custom_target(rapid_fuzzers
                    COMMAND "${CMAKE_COMMAND}" -E env
                            ${_rapid_cargo_env}
                            "${RAPID_CARGO_EXECUTABLE}" build ${_rapid_cargo_profile} --bin fuzzer
                    COMMAND "${CMAKE_COMMAND}" -E env
                            ${_rapid_cargo_env}
                            "${RAPID_CARGO_EXECUTABLE}" build ${_rapid_cargo_profile} --bin fuzzer_async
                    WORKING_DIRECTORY "${_rapid_fuzzer_dir}"
                    USES_TERMINAL)
endif()

set(_rapid_all_deps "")
if(TARGET rapid_kernels)
  list(APPEND _rapid_all_deps rapid_kernels)
endif()
if(TARGET rapid_fuzzers)
  list(APPEND _rapid_all_deps rapid_fuzzers)
endif()

# Make the default build (`cmake --build <dir>` / `make`) build everything we
# enabled in this top-level project.
add_custom_target(rapid_all ALL DEPENDS ${_rapid_all_deps})
