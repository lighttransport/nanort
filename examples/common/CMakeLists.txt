
## glew ##

add_library(glew STATIC ThirdPartyLibs/Glew/glew.c)
target_compile_definitions(glew PUBLIC -DGLEW_STATIC)
target_include_directories(glew PUBLIC ThirdPartyLibs/Glew)

## tinyexr ##
add_library(tinyexr INTERFACE)
target_include_directories(tinyexr INTERFACE ${CMAKE_CURRENT_LIST_DIR})

## trackball ##

add_library(trackball STATIC trackball.cc)
target_include_directories(trackball PUBLIC ${CMAKE_CURRENT_LIST_DIR})

## windowing ##

set(OpenGL_GL_PREFERENCE "LEGACY")
find_package(OpenGL REQUIRED)

if(WIN32)
    set(SOURCES OpenGLWindow/Win32OpenGLWindow.cpp OpenGLWindow/Win32Window.cpp)
elseif(APPLE)
    find_library(COCOA Cocoa REQUIRED)
    set(SOURCES OpenGLWindow/MacOpenGLWindow.mm)
    set(LIBS ${COCOA})
else()
    find_package(X11 REQUIRED)
    set(SOURCES OpenGLWindow/X11OpenGLWindow.cpp)
    set(LIBS ${X11_LIBRARIES})
    set(INCLUDES ${X11_INCLUDE_DIR})
endif()

add_library(windowing STATIC ${SOURCES})
target_link_libraries(windowing PUBLIC OpenGL::GL glew PRIVATE ${LIBS})
target_include_directories(windowing
PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
PRIVATE
  ${INCLUDES}
)

## ImGui ##

add_library(imgui STATIC
  imgui/imgui.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_impl_btgui.cpp
  imgui/imgui_widgets.cpp
)

target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_LIST_DIR}/imgui)
target_link_libraries(imgui PUBLIC windowing)