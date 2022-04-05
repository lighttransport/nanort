
set(BUILD_TARGET "gui")

set(SOURCES
    main.cc
    render.cc
    render-config.cc
    matrix.cc
    camera.cc
    render.h
    render-config.h
    matrix.h
    camera.h
)

add_executable(${BUILD_TARGET} ${SOURCES})

target_link_libraries(${BUILD_TARGET} PRIVATE nanort::nanort imgui trackball)

source_group("Source Files" FILES ${SOURCES})
