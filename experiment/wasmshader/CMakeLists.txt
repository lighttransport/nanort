cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME wasmshader)
project(${PROJECT_NAME} C CXX)

set (WAMR_BUILD_PLATFORM "linux")
set (WAMR_BUILD_TARGET "X86_64")
set (WAMR_BUILD_INTERP 1)
set (WAMR_BUILD_FAST_INTERP 1)
set (WAMR_BUILD_AOT 1)
set (WAMR_BUILD_LIBC_BUILTIN 1)
set (WAMR_BUILD_LIBC_WASI 1)
set (WAMR_BUILD_SIMD 1)
set (WAMR_ROOT_DIR wasm-micro-runtime/)

include (${WAMR_ROOT_DIR}/build-scripts/runtime_lib.cmake)
add_library(vmlib ${WAMR_RUNTIME_LIB_SOURCE})


add_executable(${PROJECT_NAME} main.cc)

target_link_libraries (${PROJECT_NAME} vmlib)
