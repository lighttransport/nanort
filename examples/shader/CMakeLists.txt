cmake_minimum_required(VERSION 3.5.1)

project(shader)
set(BUILD_TARGET "shader")

find_package(OpenGL REQUIRED)

if(WIN32)
    # nothing.
elseif(APPLE)
    find_library(COCOA Cocoa REQUIRED)
else()
    find_package(X11 REQUIRED)
endif()

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${X11_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../common")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../common/imgui")

# local Glew
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../common/ThirdPartyLibs/Glew")
add_definitions("-DGLEW_STATIC")

set(SOURCES
    main.cc
    render.cc
    render-config.cc
    ../common/trackball.cc
    ../common/matrix.cc
    ../common/imgui/imgui.cpp
    ../common/imgui/imgui_color_gradient.cpp
    ../common/imgui/imgui_draw.cpp
    ../common/imgui/imgui_impl_btgui.cpp
    ../common/imgui/imgui_widgets.cpp
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

target_link_libraries(
    ${BUILD_TARGET}
    ${OPENGL_LIBRARIES}
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

# [VisualStudio]
if (WIN32)
  # Set `BUILD_TARGET` as a startup project for VS IDE
  if (CMAKE_VERSION VERSION_GREATER 3.6.0)
    set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${BUILD_TARGET})
  endif ()

  # For easier debugging in VS IDE(cmake 3.8.0 or later required)
  if (CMAKE_VERSION VERSION_GREATER 3.8.0)
    # Set working directory to $lucia git repo root.
    set_target_properties(${BUILD_TARGET} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  endif ()
endif ()


