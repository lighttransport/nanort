set(BUILD_TARGET "path_tracer")

include_directories(${CMAKE_SOURCE_DIR})

file(GLOB SOURCES "*.cc" "*.h")
add_executable(${BUILD_TARGET} ${SOURCES})

source_group("Source Files" FILES ${SOURCES})
