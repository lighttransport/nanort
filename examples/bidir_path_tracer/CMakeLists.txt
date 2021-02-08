set(BUILD_TARGET "bidir_path_tracer")

file(GLOB SOURCES "*.cc" "*.h")
add_executable(${BUILD_TARGET} ${SOURCES})
target_link_libraries(${BUILD_TARGET} PRIVATE nanort::nanort tinyexr)

source_group("Source Files" FILES ${SOURCES})

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
  set_property(TARGET ${BUILD_TARGET} APPEND PROPERTY LINK_FLAGS "/DEBUG /PROFILE")
endif()
