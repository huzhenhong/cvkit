set(PROJECT_NAME cvkit)

message("\n================= Start configure ${PROJECT_NAME} ==================")

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(
    ${PROJECT_NAME}
    VERSION 1.0.0.0
    LANGUAGES CXX C
    DESCRIPTION "a cv commom library"
    HOMEPAGE_URL "project site")

if(NOT DEFINED META_PROJECT)
    set(META_PROJECT ${PROJECT_NAME})
    message("config META_PROJECT ${PROJECT_NAME}")

    include(cmake/detect_environment.cmake)
    include(cmake/compile_options.cmake)
    include(cmake/tool_chain.cmake)

    message(${CMAKE_CXX_FLAGS_DEBUG})
    message(${CMAKE_C_FLAGS_DEBUG})
    set(CMAKE_C_STANDARD 11) # 设置整个项目的默认 C 标准
    set(CMAKE_CXX_STANDARD 17) # 设置整个项目的默认 C++ 标准
    set(CMAKE_CXX_EXTENSIONS OFF) # 只启用ISO C++标准的编译器标志，而不使用特定编译器的扩展
    set(CMAKE_CXX_STANDARD_REQUIRED ON) # 开启防止编译器回退到支持的低版本的C++标准
    set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 默认为开启，代码位置无关，避免使用动态链接库时需要重定位
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) # 在 Windows 上导出所有库符号
    set(CMAKE_MACOSX_RPATH ON) # macOS上动态库使用相对路径
    set(CMAKE_DEBUG_POSTFIX "_d")
    message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
    message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

    if(CMAKE_BUILD_TYPE STREQUAL Debug)
        message("Compile in Debug mode.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O0")
    else()
        message("Compile in Release mode.")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
    endif()
    message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
    message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
    # 设置默认输出路径
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

    add_subdirectory(demo)
endif()

add_subdirectory(src)

message("================== Stop configure ${PROJECT_NAME} ==================\n")
