cmake_minimum_required(VERSION 3.20)
project(rapid_vconfig_instrument LANGUAGES CXX)

find_program(LLVM_CONFIG llvm-config-22 REQUIRED)
execute_process(
  COMMAND "${LLVM_CONFIG}" --cxxflags
  OUTPUT_VARIABLE LLVM_CXXFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${LLVM_CONFIG}" --libdir
  OUTPUT_VARIABLE LLVM_LIBDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
separate_arguments(LLVM_CXXFLAGS NATIVE_COMMAND "${LLVM_CXXFLAGS}")
find_library(LLVM_DYLIB NAMES LLVM-22 HINTS "${LLVM_LIBDIR}" REQUIRED)

add_library(rapid-vconfig-pass STATIC virtual_dim_pass.cpp)
target_compile_features(rapid-vconfig-pass PRIVATE cxx_std_17)
target_compile_options(rapid-vconfig-pass PRIVATE ${LLVM_CXXFLAGS})
target_include_directories(rapid-vconfig-pass PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(rapid-vconfig-pass PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(rapid-vconfig-instrument rapid_vconfig_instrument.cpp)
target_compile_features(rapid-vconfig-instrument PRIVATE cxx_std_17)
target_compile_options(rapid-vconfig-instrument PRIVATE ${LLVM_CXXFLAGS})
target_link_libraries(rapid-vconfig-instrument PRIVATE rapid-vconfig-pass "${LLVM_DYLIB}")

add_library(rapid-vconfig-pass-plugin MODULE virtual_dim_plugin.cpp)
target_compile_features(rapid-vconfig-pass-plugin PRIVATE cxx_std_17)
target_compile_options(rapid-vconfig-pass-plugin PRIVATE ${LLVM_CXXFLAGS})
target_link_libraries(rapid-vconfig-pass-plugin PRIVATE rapid-vconfig-pass "${LLVM_DYLIB}")
