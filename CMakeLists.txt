cmake_minimum_required(VERSION 3.23)

project(
    cvkit
    VERSION 0.1.0
    DESCRIPTION "Computer vision toolkit"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
include(CheckLanguage)
include(FindPackageHandleStandardArgs)

option(BUILD_SHARED_LIBS "Build shared libraries by default" ON)
option(CVKIT_BUILD_EXAMPLES "Build local usage examples" OFF)
option(CVKIT_BUILD_TESTS "Build local tests" OFF)
option(CVKIT_BUILD_BENCHMARKS "Build local benchmarks" OFF)
option(CVKIT_ENABLE_GSTREAMER "Enable GStreamer media backend" ON)
option(CVKIT_ENABLE_GSTREAMER_CUDA "Enable CUDA interop for the GStreamer backend" OFF)
option(CVKIT_ENABLE_FFMPEG "Enable FFmpeg media backend" OFF)
option(CVKIT_ENABLE_ONNXRUNTIME "Enable ONNX Runtime inference backend" ON)
option(CVKIT_ENABLE_OPENVINO "Enable OpenVINO inference backend" OFF)
option(CVKIT_ENABLE_TENSORRT "Enable TensorRT inference backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CVKIT_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/cvkit")
set(CVKIT_RUNTIME_OUTPUT_DIR "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CVKIT_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

set(CVKIT_OPENCV_PREFIX "" CACHE PATH "Optional prefix or CMake package directory for OpenCV")
set(CVKIT_ONNXRUNTIME_PREFIX "" CACHE PATH "Optional prefix or CMake package directory for ONNX Runtime")
set(CVKIT_FFMPEG_PREFIX "" CACHE PATH "Optional prefix or CMake package directory for FFmpeg")
set(CVKIT_OPENVINO_PREFIX "" CACHE PATH "Optional prefix or CMake package directory for OpenVINO")
set(CVKIT_TENSORRT_PREFIX "" CACHE PATH "Optional prefix or CMake package directory for TensorRT")
set(CVKIT_CUDATOOLKIT_ROOT "" CACHE PATH "Optional root path for CUDAToolkit")
set(CVKIT_CUDART_LIBRARY "/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so" CACHE FILEPATH "Optional path to libcudart.so for GStreamer CUDA interop")

if(CVKIT_OPENCV_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${CVKIT_OPENCV_PREFIX}")
endif()

if(CVKIT_ONNXRUNTIME_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${CVKIT_ONNXRUNTIME_PREFIX}")
endif()

if(CVKIT_FFMPEG_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${CVKIT_FFMPEG_PREFIX}")
endif()

if(CVKIT_OPENVINO_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${CVKIT_OPENVINO_PREFIX}")
endif()

if(CVKIT_TENSORRT_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${CVKIT_TENSORRT_PREFIX}")
endif()

find_package(basekit REQUIRED CONFIG)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

if(CVKIT_ENABLE_GSTREAMER)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
    if(CVKIT_ENABLE_GSTREAMER_CUDA)
        if(CVKIT_CUDATOOLKIT_ROOT)
            set(CUDAToolkit_ROOT "${CVKIT_CUDATOOLKIT_ROOT}")
        elseif(EXISTS "${CVKIT_CUDART_LIBRARY}")
            get_filename_component(_cvkit_cudart_libdir "${CVKIT_CUDART_LIBRARY}" DIRECTORY)
            get_filename_component(_cvkit_cuda_target_root "${_cvkit_cudart_libdir}" DIRECTORY)
            get_filename_component(_cvkit_cuda_targets_dir "${_cvkit_cuda_target_root}" DIRECTORY)
            get_filename_component(CUDAToolkit_ROOT "${_cvkit_cuda_targets_dir}" DIRECTORY)
        endif()
        find_package(CUDAToolkit REQUIRED)
    endif()
endif()

if(CVKIT_ENABLE_ONNXRUNTIME)
    find_package(onnxruntime CONFIG REQUIRED)
endif()

if(CVKIT_ENABLE_FFMPEG)
    message(FATAL_ERROR "CVKIT_ENABLE_FFMPEG=ON is reserved but not implemented yet")
endif()

if(CVKIT_ENABLE_OPENVINO)
    message(FATAL_ERROR "CVKIT_ENABLE_OPENVINO=ON is not implemented yet")
endif()

if(CVKIT_ENABLE_TENSORRT)
    if(NOT CUDAToolkit_FOUND)
        if(CVKIT_CUDATOOLKIT_ROOT)
            set(CUDAToolkit_ROOT "${CVKIT_CUDATOOLKIT_ROOT}")
        elseif(EXISTS "${CVKIT_CUDART_LIBRARY}")
            get_filename_component(_cvkit_cudart_libdir "${CVKIT_CUDART_LIBRARY}" DIRECTORY)
            get_filename_component(_cvkit_cuda_target_root "${_cvkit_cudart_libdir}" DIRECTORY)
            get_filename_component(_cvkit_cuda_targets_dir "${_cvkit_cuda_target_root}" DIRECTORY)
            get_filename_component(CUDAToolkit_ROOT "${_cvkit_cuda_targets_dir}" DIRECTORY)
        endif()
        find_package(CUDAToolkit REQUIRED)
    endif()
    find_path(
        CVKIT_TENSORRT_INCLUDE_DIR
        NAMES NvInfer.h
        HINTS
            ${CVKIT_TENSORRT_PREFIX}
            /usr
            /usr/local
        PATH_SUFFIXES
            include
            include/x86_64-linux-gnu
            x86_64-linux-gnu
    )
    find_library(
        CVKIT_TENSORRT_NVINFER_LIBRARY
        NAMES nvinfer
        HINTS
            ${CVKIT_TENSORRT_PREFIX}
            /usr
            /usr/local
        PATH_SUFFIXES
            lib
            lib64
            lib/x86_64-linux-gnu
            x86_64-linux-gnu
    )
    find_library(
        CVKIT_TENSORRT_NVONNXPARSER_LIBRARY
        NAMES nvonnxparser
        HINTS
            ${CVKIT_TENSORRT_PREFIX}
            /usr
            /usr/local
        PATH_SUFFIXES
            lib
            lib64
            lib/x86_64-linux-gnu
            x86_64-linux-gnu
    )
    find_package_handle_standard_args(
        TensorRT
        REQUIRED_VARS
            CVKIT_TENSORRT_INCLUDE_DIR
            CVKIT_TENSORRT_NVINFER_LIBRARY
            CVKIT_TENSORRT_NVONNXPARSER_LIBRARY
    )
endif()

add_subdirectory(src/core)
add_subdirectory(src/media)
add_subdirectory(src/image)
add_subdirectory(src/infer)

if(CVKIT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(CVKIT_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(CVKIT_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cvkitConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cvkitConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cvkitConfig.cmake
    INSTALL_DESTINATION ${CVKIT_CMAKE_INSTALL_DIR}
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/cvkitConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cvkitConfigVersion.cmake
    DESTINATION ${CVKIT_CMAKE_INSTALL_DIR}
)

install(
    EXPORT cvkitTargets
    FILE cvkitTargets.cmake
    NAMESPACE cvkit::
    DESTINATION ${CVKIT_CMAKE_INSTALL_DIR}
)
