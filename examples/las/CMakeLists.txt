cmake_minimum_required(VERSION 3.16)

set(TARGET lasrender)

if (APPLE)
  project(${TARGET} CXX OBJCXX)
else()
  project(${TARGET} CXX)
endif()


option(LASRENDER_USE_PDAL "Use PDAL instead of libLAS" ON)

find_package(OpenGL REQUIRED)
#find_package(GLEW REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/../common/glfw glfw)

set(SOURCES 
   main.cc
   render.cc
   render-config.cc
   ${PROJECT_SOURCE_DIR}/../common/trackball.cc
   ${PROJECT_SOURCE_DIR}/../common/matrix.cc
   ${PROJECT_SOURCE_DIR}/../common/imgui/imgui.cpp
   ${PROJECT_SOURCE_DIR}/../common/imgui/imgui_draw.cpp
   ${PROJECT_SOURCE_DIR}/../common/imgui/imgui_widgets.cpp
   ${PROJECT_SOURCE_DIR}/../common/imgui/imgui_tables.cpp
   ${PROJECT_SOURCE_DIR}/../common/imgui/imgui_impl_glfw.cpp
   ${PROJECT_SOURCE_DIR}/../common/imgui/imgui_impl_opengl3.cpp
  )

if (LASRENDER_USE_PDAL)
  find_package(PDAL REQUIRED CONFIG)
  set(DEP_LIBS ${PDAL_LIBRARIES})
else()
  find_package(las REQUIRED)
  set(DEP_LIBS las)
endif()

add_executable(${TARGET} ${SOURCES} ${UI_SOURCES})
if (APPLE)
  target_link_libraries(${TARGET} "-framework Foundation -framework Cocoa")
elseif(UNIX)
  target_link_libraries(${TARGET} X11)
endif()
target_link_libraries(${TARGET} ${DEP_LIBS} OpenGL::GL glfw ${CMAKE_DL_LIBS})
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/../common/)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/../common/imgui)
target_include_directories(${TARGET} PRIVATE ${GLEW_INCLUDE_DIRS})

if (LASRENDER_USE_PDAL)
  target_include_directories(${TARGET} PRIVATE ${PDAL_INCLUDE_DIRS})
  target_link_directories(${TARGET} PRIVATE ${PDAL_LIBRARY_DIRS})
  target_compile_definitions(${TARGET} PRIVATE LASRENDER_USE_PDAL)
endif()

# [VisualStudio]
if(WIN32)
  #SET_TARGET_PROPERTIES(${TINYUSDZ_TARGET_STATIC} PROPERTIES LINK_FLAGS "/PROFILE")
  # Set ${TINYUSDZ_TARGET_STATIC} as a startup project for VS IDE
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${TARGET})

  # For easier debugging in VS IDE(cmake 3.8.0 or later required) Set working
  # directory to ${TINYUSDZ_TARGET_STATIC} git repo root.
  if(CMAKE_VERSION VERSION_GREATER 3.8.0)
    set_target_properties(
      ${TARGET} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY
                                           "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
endif()

