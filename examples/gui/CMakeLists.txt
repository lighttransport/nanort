set(BUILD_TARGET "gui")

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

include_directories(${CMAKE_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/examples/common")
include_directories(${CMAKE_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/examples/common/imgui")

# local Glew
include_directories(${CMAKE_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/examples/common/ThirdPartyLibs/Glew")
add_definitions("-DGLEW_STATIC")

set(SOURCES
    main.cc
    render.cc
    render-config.cc
    ../common/trackball.cc
    matrix.cc
    ../common/imgui/imgui.cpp
    ../common/imgui/imgui_draw.cpp
    ../common/imgui/imgui_impl_btgui.cpp
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
