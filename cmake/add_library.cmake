# opencv
function(add_opencv Target)
    set(OPENCV_NEED_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_video)

    if(PLATFORM STREQUAL Cambricon)
        target_include_directories(${Target} PUBLIC /cambricon/third_party/opencv/include/opencv4)
        target_link_directories(${Target} PUBLIC /cambricon/third_party/opencv/lib)
    else()
        find_package(OpenCV REQUIRED)
        if(NOT OpenCV_FOUND)
            message(FATAL_ERROR "opencv not found!")
        else()
            target_include_directories(${Target} PUBLIC ${OpenCV_INCLUDE_DIRS})
            target_link_directories(${Target} PUBLIC ${OpenCV_LIBRARY_DIRS})
            message("opencv include dir : " ${OpenCV_INCLUDE_DIRS})
            message("opencv lib dir : " ${OpenCV_LIBRARY_DIRS})

            find_package(CUDA QUIET)
            if(CUDA_FOUND)
                list(APPEND OPENCV_NEED_LIBS opencv_cudacodec)
            endif()
        endif()
    endif()

    target_link_libraries(${Target} PUBLIC ${OPENCV_NEED_LIBS})
endfunction()

# cuda
function(add_cuda Target)
    find_package(CUDA REQUIRED)
    if(NOT CUDA_FOUND)
        message(FATAL_ERROR "CUDA not found!")
    endif()

    if(CUDA_ENABLE)
        enable_language(CUDA)
    endif()

    target_include_directories(${Target} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_directories(${Target} PUBLIC ${CUDA_LIBRARY_DIRS})
    message("CUDA include dir : " ${CUDA_INCLUDE_DIRS})
    message("CUDA lib dir : " ${CUDA_LIBRARY_DIRS})
endfunction()

# tensorrt
function(add_tensorrt Target)
    if(WIN32)
        set(TRT_ROOT_DIR D:/dev-env/TensorRT-[7-9].[0-9].[0-9].[0-9])
    else()
        file(GLOB TRT_ROOT_DIR "/usr/local/TensorRT-[7-9].[0-9].[0-9].[0-9]")
    endif()

    if(EXISTS TRT_ROOT_DIR)
        target_include_directories(${Target} PUBLIC ${TRT_ROOT_DIR}/include)
        target_link_directories(${Target} PUBLIC ${TRT_ROOT_DIR}/lib)
        message("TensorRT include : " ${TRT_ROOT_DIR}/include)
        message("TensorRT lib : " ${TRT_ROOT_DIR}/lib)
    else()
        message("TensorRT path do not exist")
    endif()

    set(TRT_NEED_LIBS nvinfer_plugin nvinfer nvonnxparser nvparsers)
    target_link_libraries(${Target} PUBLIC ${TRT_NEED_LIBS})
endfunction()

# libzj_hf_codec
function(add_zj_hf_codec Target)
    if(PLATFORM STREQUAL Cambricon)
        string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE_LOWER)
        string(TOLOWER ${PLATFORM} PLATFORM_LOWER)
        string(TOLOWER ${CMAKE_HOST_SYSTEM_PROCESSOR} PROCESSOR_LOWER)
        target_include_directories(${Target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/libzj_hf_codec/include)
        target_link_directories(
            ${Target}
            PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/libzj_hf_codec/lib/${BUILD_TYPE_LOWER}/${PLATFORM_LOWER}/${PROCESSOR_LOWER}
        )
    else()
        target_include_directories(${Target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/libzj_hf_codec/include)
        target_link_directories(
            ${Target}
            PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/libzj_hf_codec/lib/${BUILD_TYPE_LOWER}/${PLATFORM_LOWER}/${PROCESSOR_LOWER}
        )
    endif()

    set(ZJ_HF_CODEC_NEED_LIBS
        swresample
        avcodec
        avutil
        swscale
        avformat
        avfilter
        zj_hf_codec)
    target_link_libraries(${Target} PUBLIC ${ZJ_HF_CODEC_NEED_LIBS})
endfunction()
