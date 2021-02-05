set(BUILD_TARGET "nanosg")

set(SOURCES
    main.cc
    render.cc
    render-config.cc
    obj-loader.cc
)

add_executable(${BUILD_TARGET} ${SOURCES})

target_link_libraries( ${BUILD_TARGET}
PRIVATE
    nanort::nanort
    imgui
    trackball
    matrix
    glm
)

source_group("Source Files" FILES ${SOURCES})
