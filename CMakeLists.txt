message("\n================= Start configure [cvkit] ==================")

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

if(NOT DEFINED MASTER_PROJECT)
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(CVKIT_MASTER_PROJECT ON)
    else()
        set(CVKIT_MASTER_PROJECT OFF)
    endif()
endif()

project(
    cvkit
    VERSION 1.0.0.0
    LANGUAGES CXX C
    DESCRIPTION "a cv commom library"
    HOMEPAGE_URL "project site")

include(cmake/detect_environment.cmake)
include(cmake/compile_options.cmake)
include(cmake/tool_chain.cmake)

set(CMAKE_C_STANDARD 11) # 设置整个项目的默认 C 标准
set(CMAKE_CXX_STANDARD 17) # 设置整个项目的默认 C++ 标准
set(CMAKE_CXX_EXTENSIONS OFF) # 只启用ISO C++标准的编译器标志，而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 开启防止编译器回退到支持的低版本的C++标准
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 默认为开启，代码位置无关，避免使用动态链接库时需要重定位
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) # 在 Windows 上导出所有库符号
set(CMAKE_MACOSX_RPATH ON) # macOS上动态库使用相对路径
set(CMAKE_DEBUG_POSTFIX "_d")

# 设置默认输出路径
set(CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

if(${CVKIT_MASTER_PROJECT})
    add_subdirectory(demo)
endif()

add_subdirectory(src)

message("================== Stop configure [cvkit] ==================\n")
