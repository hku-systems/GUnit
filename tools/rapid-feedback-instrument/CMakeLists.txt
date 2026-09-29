cmake_minimum_required(VERSION 3.20)
project(rapid_feedback_instrument LANGUAGES CXX)

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

add_executable(rapid-feedback-instrument rapid_feedback_instrument.cpp)
target_compile_features(rapid-feedback-instrument PRIVATE cxx_std_17)
target_compile_options(rapid-feedback-instrument PRIVATE ${LLVM_CXXFLAGS})
target_link_libraries(rapid-feedback-instrument PRIVATE "${LLVM_DYLIB}")
