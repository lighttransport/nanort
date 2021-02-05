set(BUILD_TARGET "path_tracer")

file(GLOB SOURCES "*.cc" "*.h")
add_executable(${BUILD_TARGET} ${SOURCES})
target_link_libraries(${BUILD_TARGET} PRIVATE nanort::nanort tinyexr)

source_group("Source Files" FILES ${SOURCES})
