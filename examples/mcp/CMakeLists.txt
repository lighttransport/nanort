cmake_minimum_required(VERSION 3.5)
project(nanort_mcp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include NanoRT headers
include_directories(../../)
include_directories(../common/)

# Platform-specific settings
if(WIN32)
    add_definitions(-D_WIN32_WINNT=0x0601)
    set(PLATFORM_LIBS ws2_32)
elseif(UNIX)
    set(PLATFORM_LIBS pthread)
endif()

# Common source files
set(COMMON_SOURCES
    mcp_server.cc
    nanort_mcp.cc
)

set(COMMON_HEADERS
    mcp_server.h
    nanort_mcp.h
)

# Stdio version
add_executable(nanort_mcp_stdio
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
    main_stdio.cc
)

target_link_libraries(nanort_mcp_stdio ${PLATFORM_LIBS})

# HTTP version
add_executable(nanort_mcp_http
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
    main_http.cc
)

target_link_libraries(nanort_mcp_http ${PLATFORM_LIBS})

# Installation
install(TARGETS nanort_mcp_stdio nanort_mcp_http
    RUNTIME DESTINATION bin
)

# Add to examples if part of main NanoRT build
if(TARGET nanort_examples)
    add_dependencies(nanort_examples nanort_mcp_stdio nanort_mcp_http)
endif()