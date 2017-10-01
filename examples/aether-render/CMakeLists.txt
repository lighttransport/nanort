cmake_minimum_required(VERSION 3.1)
project(nanort-aether CXX)

message(STATUS ${PROJECT_SOURCE_DIR})

if(NOT IS_DIRECTORY "${PROJECT_SOURCE_DIR}/aether/")
  message(FATAL_ERROR "Aether directory is missing! "
    "You probably did not clone the project Aether. It is possible to recover "
    "by running \"clone https://github.com/aekul/aether.git\" at `aether-render` top directory.")
endif()


# Download automatically, you can also just copy the conan.cmake file
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/memsharded/cmake-conan")
   file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
                 "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

# Force use clang
# Force set c++1z and libc++
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -stdlib=libc++")

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_run(CONANFILE conanfile.txt  # or relative build/conanfile.txt
                BASIC_SETUP CMAKE_TARGETS
                BUILD missing)

include_directories(${PROJECT_SOURCE_DIR})

add_executable(nanort-aether src/main.cc)
target_link_libraries(nanort-aether CONAN_PKG::Boost CONAN_PKG::Eigen3)
