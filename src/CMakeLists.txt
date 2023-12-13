include(../cmake/fetch_submodule.cmake)
include(../cmake/add_library.cmake)

set(Target cvkit)

message("\n------------------ Start generate : ${Target} ------------------")

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    message("Compile in Debug mode.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O0")
else()
    message("Compile in Release mode.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
endif()

file(GLOB_RECURSE H_SRCS *.h)
file(GLOB_RECURSE C_SRCS *.c)
file(GLOB_RECURSE HPP_SRCS *.hpp)
file(GLOB_RECURSE CPP_SRCS *.cpp)

aux_source_directory(img IMG_SRC)
aux_source_directory(video VIDEO_SRC)
aux_source_directory(deploy/yolov8 DEPLOY_YOLO8_SRC)
set(${Target}_SRCS ${IMG_SRC} ${VIDEO_SRC} ${DEPLOY_YOLO8_SRC})
# set(${Target}_SRCS ${H_SRCS} ${C_SRCS} ${HPP_SRCS} ${CPP_SRCS})

add_library(${Target} SHARED ${${Target}_SRCS})
add_library(${Target}::${Target} ALIAS ${Target})

target_link_libraries(${Target} PRIVATE spdlog::spdlog)

add_opencv(${Target})

if(BUILD_WITH_CUDA OR BUILD_WITH_TRT)
    add_cuda(${Target})
endif()

if(BUILD_WITH_TRT)
    add_tensorrt(${Target})
endif()

if(LINUX)
    add_zj_hf_codec(${Target})
endif()

message("img:" ${CMAKE_CURRENT_SOURCE_DIR}/img)
target_include_directories(${Target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/img)
target_include_directories(${Target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/video)
target_include_directories(${Target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/deploy)

# target_include_directories(${Target} INTERFACE
# "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/img>")
# target_include_directories(${Target} INTERFACE
# "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/video>")

target_link_libraries(${Target} PUBLIC # basekit::basekit
                                       spdlog::spdlog)

if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR /Users/huzh/Documents/deploy_backends/onnxruntime-osx-x86_64-1.16.1)
endif()

target_include_directories(${Target} PUBLIC ${ONNXRUNTIME_DIR}/include)
if(LINUX)
    target_link_libraries(${Target} PUBLIC "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
elseif(WIN32)
    target_link_libraries(${Target} PUBLIC "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
elseif(APPLE)
    target_link_libraries(${Target} PUBLIC "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
endif()

message("------------------ Stop generate : ${Target} ------------------\n")
