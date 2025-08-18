

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
    ../common/trackball.cc
    ../common/imgui/imgui.cpp
    ../common/imgui/imgui_draw.cpp
    ../common/imgui/imgui_widgets.cpp
    ../common/imgui/imgui_tables.cpp
    ../common/imgui/backends/imgui_impl_glfw.cpp
    ../common/imgui/backends/imgui_impl_opengl3.cpp
)

add_subdirectory(../common/glfw glfw)

add_executable(${BUILD_TARGET} ${SOURCES})

target_include_directories(${BUILD_TARGET} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../common
    ../common/imgui
    ../common/imgui/backends
    ../common/glfw/include
)

if(WIN32)
    target_link_libraries(${BUILD_TARGET} PRIVATE opengl32 glu32)
else()
    find_package(OpenGL REQUIRED)
    target_link_libraries(${BUILD_TARGET} PRIVATE OpenGL::GL)
endif()

target_link_libraries(${BUILD_TARGET} PRIVATE glfw nanort::nanort)
target_compile_definitions(${BUILD_TARGET} PRIVATE 
    IMGUI_IMPL_OPENGL_LOADER_GL3W=0
    STBI_NO_SIMD
)

source_group("Source Files" FILES ${SOURCES})
