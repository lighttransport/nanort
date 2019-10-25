cmake_minimum_required(VERSION 3.5.1)

project(subdrt)

set(BUILD_TARGET "subdrt")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED On)

find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(ZLIB REQUIRED) # ptex require zlib

if(WIN32)
    # nothing.
elseif(APPLE)
    find_library(COCOA Cocoa REQUIRED)
else()
    find_package(X11 REQUIRED)
endif()

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${X11_INCLUDE_DIR})

include_directories(${PROJECT_SOURCE_DIR} "${PROJECT_SOURCE_DIR}/../common")
include_directories(${PROJECT_SOURCE_DIR} "${PROJECT_SOURCE_DIR}/../common/imgui")

# local Glew
include_directories(${PROJECT_SOURCE_DIR} "${PROJECT_SOURCE_DIR}/../common/ThirdPartyLibs/Glew")
include_directories(${PROJECT_SOURCE_DIR} "${PROJECT_SOURCE_DIR}/../common/ThirdPartyLibs/Glew")
add_definitions("-DGLEW_STATIC")

# embree
find_package(embree 3)
include_directories(${EMBREE_INCLUDE_DIRS})
list (APPEND EXT_LIBS ${EMBREE_LIBRARY})


# OpenSubdiv
include_directories(${OSD_DIR}/include)
link_directories(${OSD_DIR}/lib)
list (APPEND EXT_LIBS osdCPU)

# local ptex
# Force set version and SHA
# set(PTEX_SHA "9885965ee4a8664f475d9c00add9d15ce76440f2")
# set(PTEX_VER "v2.3.2-2-g9885965")
include_directories(${PTEX_DIR}/include)
link_directories(${PTEX_DIR}/lib)
list (APPEND EXT_LIBS Ptex)

set(SOURCES
    main.cc
    render.cc
    render-config.cc
    primitive.cc
    ../common/trackball.cc
    ../common/matrix.cc
    ../common/imgui/imgui.cpp
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
target_include_directories(${BUILD_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ptex-aarch64/src/ptex)

target_link_libraries(
    ${BUILD_TARGET}
    ${OPENGL_LIBRARIES}
    ${EXT_LIBS}
    Threads::Threads
    ZLIB::ZLIB
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
