cmake_minimum_required(VERSION 3.0)

set(BUILD_TARGET "point_query")

# Point to $nanort/cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../cmake" ${CMAKE_MODULE_PATH})
find_package(Sanitizers)

find_package(OpenGL REQUIRED)
find_package(Threads)

if(WIN32)
    # nothing.
elseif(APPLE)
    find_library(COCOA Cocoa REQUIRED)
else()
    find_package(X11 REQUIRED)
endif()

# C++11
set(CMAKE_CXX_STANDARD 11)

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${X11_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/../../") # $nanort top directory(where `nanort.h` locates)
include_directories(${CMAKE_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/../common")
include_directories(${CMAKE_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/../common/imgui")

# local Glew
include_directories(${CMAKE_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/../common/ThirdPartyLibs/Glew")
add_definitions("-DGLEW_STATIC")

set(MAIN_SOURCES
    main.cc
    render.cc
    render-config.cc
)

set(EXT_SOURCES
    ../common/trackball.cc
    ../common/matrix.cc
    ../common/imgui/imgui.cpp
    ../common/imgui/imgui_draw.cpp
    ../common/imgui/imgui_widgets.cpp
    ../common/imgui/imgui_impl_btgui.cpp
)



if(WIN32)
    set(EXT_SOURCES ${EXT_SOURCES}
        ../common/OpenGLWindow/Win32OpenGLWindow.cpp
        ../common/OpenGLWindow/Win32Window.cpp
    )
elseif(APPLE)
    set(EXT_SOURCES ${EXT_SOURCES} ../common/OpenGLWindow/MacOpenGLWindow.mm)
else()
    set(EXT_SOURCES ${EXT_SOURCES} ../common/OpenGLWindow/X11OpenGLWindow.cpp)
endif()

set(EXT_SOURCES ${EXT_SOURCES} ../common/ThirdPartyLibs/Glew/glew.c)

add_library(${BUILD_TARGET}_ext ${EXT_SOURCES})

add_executable(${BUILD_TARGET} ${MAIN_SOURCES})
add_sanitizers(${BUILD_TARGET})

# Increase warning level for clang.
# Only apply main source codes.
IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG(-Wzero-as-null-pointer-constant FLAG_ZERO_AS_NULL_POINTER_CONSTANT)
  CHECK_CXX_COMPILER_FLAG(-Wdouble-promotion FLAG_DOUBLE_PROMOTION)
  CHECK_CXX_COMPILER_FLAG(-Wcast-qual FLAG_CAST_QUAL)

  if (FLAG_ZERO_AS_NULL_POINTER_CONSTANT)
    list(APPEND CLANG_OPTIONAL_FLAGS "-Wno-zero-as-null-pointer-constant")
  endif()
  if (FLAG_DOUBLE_PROMOTION)
    list(APPEND CLANG_OPTIONAL_FLAGS "-Wno-double-promotion")
  endif()
  if (FLAG_CAST_QUAL)
    list(APPEND CLANG_OPTIONAL_FLAGS "-Wno-cast-qual")
  endif()
  target_compile_options(${BUILD_TARGET} PRIVATE -Weverything -Wno-system-headers -Werror -Wno-padded -Wno-c++98-compat-pedantic -Wno-documentation -Wno-documentation-unknown-command ${CLANG_OPTIONAL_FLAGS})
ENDIF ()

target_link_libraries(
    ${BUILD_TARGET}
    ${BUILD_TARGET}_ext
    ${OPENGL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
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
