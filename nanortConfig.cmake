# The MIT License (MIT)
#
# Copyright (c) 2021 Light Transport Entertainment, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.1)

if (TARGET nanort::nanort)
  return()
endif()

macro(nanort_config_message)
  if (NOT DEFINED nanort_FIND_QUIETLY)
    message(${ARGN})
  endif()
endmacro()

## Setup base target ##

add_library(nanort::core INTERFACE IMPORTED)
target_include_directories(nanort::core INTERFACE ${CMAKE_CURRENT_LIST_DIR})

nanort_config_message(STATUS "Found nanort: ${CMAKE_CURRENT_LIST_DIR}")
nanort_config_message("    --> nanort core target available (nanort::core)")

## Setup target which uses C++11 threads ##

find_package(Threads QUIET)
if (TARGET Threads::Threads)
  add_library(nanort::threads INTERFACE IMPORTED)

  target_compile_features(nanort::threads
  INTERFACE
    cxx_std_11
  )

  target_link_libraries(nanort::threads
  INTERFACE
    Threads::Threads
    nanort::core
  )

  target_compile_definitions(nanort::threads
  INTERFACE
    -DNANORT_USE_CPP11_FEATURE
    -DNANORT_ENABLE_PARALLEL_BUILD
  )

  nanort_config_message("    --> nanort C++11 threading target available (nanort::threads)")
else()
  nanort_config_message(WARNING "nanort C++11 threading target NOT available! (unable to find Threads)")
endif()

## Setup target which uses OpenMP ##

find_package(OpenMP QUIET)
if (TARGET OpenMP::OpenMP_CXX)
  add_library(nanort::openmp INTERFACE IMPORTED)

  target_link_libraries(nanort::openmp
  INTERFACE
    nanort::core
    OpenMP::OpenMP_CXX
  )

  target_compile_definitions(nanort::openmp
  INTERFACE
    -DNANORT_ENABLE_PARALLEL_BUILD
  )

  nanort_config_message("    --> nanort OpenMP target available (nanort::openmp)")
else()
  nanort_config_message(WARNING "nanort OpenMP target NOT available! (unable to find OpenMP)")
endif()

## Setup a target which uses the "best" available version ##

add_library(nanort::nanort INTERFACE IMPORTED)
if (TARGET nanort::openmp)
  target_link_libraries(nanort::nanort INTERFACE nanort::openmp)
  nanort_config_message("    --> default target (nanort::nanort) uses OpenMP")
elseif (TARGET nanort::threads)
  target_link_libraries(nanort::nanort INTERFACE nanort::threads)
  nanort_config_message("    --> default target (nanort::nanort) uses C++11 threads")
else()
  target_link_libraries(nanort::nanort INTERFACE nanort::core)
  nanort_config_message("    --> default target (nanort::nanort) uses only core nanort")
endif()
