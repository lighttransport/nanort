CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(pbr_surface)

include_directories(. ../../ ../common/glm/glm ../common)

add_executable(pbr_surface main.cc pbr_maths.hh)
