

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
    ../common/imgui/imgui_impl_btgui.cpp
    ../common/ThirdPartyLibs/Glew/glew.c
)

if (WIN32)
  list(APPEND SOURCES ../common/OpenGLWindow/Win32OpenGLWindow.cpp ../common/OpenGLWindow/Win32Window.cpp)
else()
endif()

add_executable(${BUILD_TARGET} ${SOURCES})

target_include_directories(${BUILD_TARGET} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../common
    ../common/OpenGLWindow
    ../common/imgui
    ../common/ThirdPartyLibs/Glew
)

if(WIN32)
    target_link_libraries(${BUILD_TARGET} PRIVATE opengl32 glu32)
endif()

target_compile_definitions(${BUILD_TARGET} PRIVATE GLEW_STATIC)
target_link_libraries(${BUILD_TARGET} PRIVATE nanort::nanort)

source_group("Source Files" FILES ${SOURCES})
