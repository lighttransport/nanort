CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(pbr_surface)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

SET(CMAKE_CXX_STANDARD 11)

include_directories(. ../../ ../common/glm/glm ../common)

add_executable(pbr_surface main.cc pbr_maths.hh)
