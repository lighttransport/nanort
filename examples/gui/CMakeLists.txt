
set(BUILD_TARGET "gui")

set(SOURCES
    main.cc
    render.cc
    render-config.cc
    matrix.cc
)

add_executable(${BUILD_TARGET} ${SOURCES})

target_link_libraries(${BUILD_TARGET} PRIVATE nanort::nanort imgui trackball)

source_group("Source Files" FILES ${SOURCES})
