CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(pbr_surface)

SET(CMAKE_CXX_STANDARD 11)

include_directories(. ../../ ../common/glm/glm ../common)

add_executable(pbr_surface main.cc pbr_maths.hh)
