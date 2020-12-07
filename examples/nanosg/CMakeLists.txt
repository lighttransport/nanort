cmake_minimum_required(VERSION 3.5.1)

set(BUILD_TARGET "nanosg")

find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

if(WIN32)
    # nothing.
elseif(APPLE)
    find_library(COCOA Cocoa REQUIRED)
else()
    find_package(X11 REQUIRED)
endif()

set(SOURCES
    main.cc
    render.cc
    render-config.cc
    obj-loader.cc
    ../common/trackball.cc
    ../common/matrix.cc
    ../common/imgui/imgui.cpp
    ../common/imgui/imgui_draw.cpp
    ../common/imgui/imgui_impl_btgui.cpp
    ../common/imgui/imgui_widgets.cpp
    ../common/imgui/ImGuizmo.cpp
)

if(WIN32)
    set(SOURCES ${SOURCES}
        ../common/OpenGLWindow/Win32OpenGLWindow.cpp
        ../common/OpenGLWindow/Win32Window.cpp
    )
elseif(APPLE)
    set(SOURCES ${SOURCES} ../common/OpenGLWindow/MacOpenGLWindow.mm)
else()
    set(SOURCES ${SOURCES} ../common/OpenGLWindow/X11OpenGLWindow.cpp)
endif()

set(SOURCES ${SOURCES} ../common/ThirdPartyLibs/Glew/glew.c)

add_executable(${BUILD_TARGET} ${SOURCES})

target_include_directories(${BUILD_TARGET} PRIVATE ${OPENGL_INCLUDE_DIR})
target_include_directories(${BUILD_TARGET} PRIVATE ${X11_INCLUDE_DIR})

target_include_directories(${BUILD_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
target_include_directories(${BUILD_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${BUILD_TARGET} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")
target_include_directories(${BUILD_TARGET} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common/glm")
target_include_directories(${BUILD_TARGET} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common/imgui")

# local Glew
target_include_directories(${BUILD_TARGET} PRIVATE "${CMAKE_SOURCE_DIR}/../common/ThirdPartyLibs/Glew")
target_compile_definitions(${BUILD_TARGET} PRIVATE GLEW_STATIC)


target_link_libraries(
    ${BUILD_TARGET}
    ${OPENGL_LIBRARIES}
    Threads::Threads
)

if(WIN32)
    # nothing.
elseif(APPLE)
    target_link_libraries(
        ${BUILD_TARGET}
        ${COCOA}
    )
else()
    target_link_libraries(
        ${BUILD_TARGET}
        ${X11_LIBRARIES}
    )
endif()

source_group("Source Files" FILES ${SOURCES})
