cmake_minimum_required(VERSION 3.5.1)

project(PtexRender)

if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ptex")
  message(FATAL_ERROR "The PTex repository is missing! "
    "You probably did not clone the project. It is possible to recover "
    "by calling \"git clone https://github.com/wdas/ptex\"")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake" ${CMAKE_MODULE_PATH})
find_package(Sanitizers) # Address sanitizer.

# C++11
set (CMAKE_CXX_STANDARD 11)

# Threads
find_package(Threads)

find_package(OpenGL REQUIRED) 

set (PTEXRENDER_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/render.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/render-config.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/matrix.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/trackball.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/imgui/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/imgui/imgui_impl_btgui.cpp
    )

if (WIN32)
   list(APPEND PTEXRENDER_SOURCE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/OpenGLWindow/Win32OpenGLWindow.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/OpenGLWindow/Win32OpenGLWindow.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/OpenGLWindow/Win32Window.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/OpenGLWindow/Win32Window.h
       )
   set(GL_LIBS opengl32 glu32)
elseif (APPLE)
   list(APPEND PTEXRENDER_SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/OpenGLWindow/MacOpenGLWindow.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/OpenGLWindow/MacOpenGLWindow.mm
       )
   find_library(cocoa_library Cocoa)
   set(GL_LIBS ${OPENGL_LIBRARIES} ${cocoa_library})
else()
  # Assume Linux
   list(APPEND PTEXRENDER_SOURCE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/OpenGLWindow/X11OpenGLWindow.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/OpenGLWindow/X11OpenGLWindows.h
       )
   set(GL_LIBS X11)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/imgui
    )

add_executable(ptexrender
    ${PTEXRENDER_SOURCE}
    )

add_sanitizers(ptexrender)

target_link_libraries( ptexrender
    ${GL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS})

# Increase warning level for clang.
#IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#    target_compile_options(ptexrender PRIVATE -Weverything -Wno-system-headers -Werror -Wno-padded -Wno-c++98-compat-pedantic -Wno-documentation -Wno-documentation-unknown-command)
#ENDIF ()
#

# [VisualStudio]
if (WIN32)
  # Set `liselotte` as a startup project for VS IDE
  if (CMAKE_VERSION VERSION_GREATER 3.6.0)
    set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ptexrender)
  endif ()
  
  # For easier debugging in VS IDE(cmake 3.8.0 or later required)
  if (CMAKE_VERSION VERSION_GREATER 3.8.0)
    # Set working directory to $lucia git repo root.
    set_target_properties(ptexrender PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  endif ()
endif ()

# [Android standalone]
if (ANDROID_ABI)
  # NDK
  find_library(log-lib log)
  target_link_libraries(ptexrender ${log-lib})
endif (ANDROID_ABI)
