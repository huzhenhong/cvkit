include(../cmake/fetch_submodule.cmake)
include(../cmake/add_library.cmake)

file(GLOB_RECURSE H_SRCS *.h)
file(GLOB_RECURSE C_SRCS *.c)
file(GLOB_RECURSE HPP_SRCS *.hpp)
file(GLOB_RECURSE CPP_SRCS *.cpp)

aux_source_directory(img IMG_SRC)
aux_source_directory(video VIDEO_SRC)
aux_source_directory(deploy/yolov8 DEPLOY_YOLO8_SRC)
set(SRCS ${IMG_SRC} ${VIDEO_SRC} ${DEPLOY_YOLO8_SRC})

add_library(cvkit SHARED ${SRCS})
add_library(cvkit::cvkit ALIAS cvkit)

target_link_libraries(cvkit PUBLIC spdlog::spdlog)

add_opencv(cvkit)

if(BUILD_WITH_CUDA OR BUILD_WITH_TRT)
    add_cuda(cvkit)
endif()

if(BUILD_WITH_TRT)
    add_tensorrt(cvkit)
endif()

if(LINUX)
    add_zj_hf_codec(cvkit)
endif()

message("img:" ${CMAKE_CURRENT_SOURCE_DIR}/img)
target_include_directories(cvkit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/img)
target_include_directories(cvkit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/video)
target_include_directories(cvkit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/deploy)

# target_include_directories(cvkit INTERFACE
# "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/img>")
# target_include_directories(cvkit INTERFACE
# "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/video>")

if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR /Users/huzh/Documents/deploy_backends/onnxruntime-osx-x86_64-1.16.1)
endif()

target_include_directories(cvkit PUBLIC ${ONNXRUNTIME_DIR}/include)
if(LINUX)
    target_link_libraries(cvkit PUBLIC "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
elseif(WIN32)
    target_link_libraries(cvkit PUBLIC "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
elseif(APPLE)
    target_link_libraries(cvkit PUBLIC "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
endif()
