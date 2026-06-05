include(GenerateExportHeader)

add_library(cvkit_infer
    executor.cpp
    graph/graph.cpp
    model.cpp
    session.cpp
    registry.cpp
    tile_runner.cpp
    backends/onnxruntime/ort_session.cpp
    backends/tensorrt/trt_session.cpp
    tasks/classification/classification_pipeline.cpp
    tasks/detection/detection_pipeline.cpp
    tasks/detection/yolo/yolo_preprocess_cpu.cpp
    tasks/detection/yolo/yolo_preprocess_cuda.cpp
    tasks/detection/yolo/yolo_postprocess.cpp
    tasks/face_detection/face_detection_pipeline.cpp
    tasks/facemesh/facemesh_pipeline.cpp
    tasks/pose/pose_pipeline.cpp
    tasks/segmentation/segmentation_pipeline.cpp
    tasks/promptable_segmentation/promptable_segmentation_pipeline.cpp
    tasks/promptable_segmentation/promptable_preprocess_cuda.cpp
    utils/debug.cpp
    utils/image_preprocess.cpp
    utils/image_value_utils.cpp
    utils/labels.cpp
    utils/nms.cpp
    utils/bbox.cpp
    utils/opencv_utils.cpp
    utils/tensor_io.cpp
    utils/tensor_layout.cpp
    utils/tensor_value_utils.cpp
)

add_library(cvkit::infer ALIAS cvkit_infer)

generate_export_header(
    cvkit_infer
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/generated/cvkit/infer/infer_export.h
    EXPORT_MACRO_NAME BK_INFER_EXPORT
)

target_include_directories(cvkit_infer
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(cvkit_infer
    PUBLIC
        cvkit::core
        cvkit::image
        basekit::config
        basekit::log
    PRIVATE
        opencv_core
        opencv_imgproc
)

target_compile_features(cvkit_infer PUBLIC cxx_std_17)

if(CVKIT_ENABLE_ONNXRUNTIME)
    target_compile_definitions(cvkit_infer PRIVATE CVKIT_WITH_ONNXRUNTIME=1)
    target_link_libraries(cvkit_infer PRIVATE onnxruntime::onnxruntime)
endif()

if(TARGET CUDA::cudart)
    target_compile_definitions(cvkit_infer PRIVATE CVKIT_WITH_CUDA_RUNTIME=1)
    target_link_libraries(cvkit_infer PRIVATE CUDA::cudart)
endif()

if(CVKIT_ENABLE_CUDA_LANGUAGE)
    target_compile_definitions(cvkit_infer PRIVATE CVKIT_WITH_CUDA_PREPROCESS_KERNEL=1)
    target_sources(cvkit_infer PRIVATE tasks/detection/yolo/yolo_preprocess_cuda_kernel.cu)
    target_sources(cvkit_infer PRIVATE tasks/promptable_segmentation/promptable_preprocess_cuda_kernel.cu)
    set_target_properties(cvkit_infer PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

if(CVKIT_ENABLE_OPENVINO)
    target_compile_definitions(cvkit_infer PRIVATE CVKIT_WITH_OPENVINO=1)
endif()

if(CVKIT_ENABLE_TENSORRT)
    target_compile_definitions(cvkit_infer PRIVATE CVKIT_WITH_TENSORRT=1)
    target_include_directories(cvkit_infer PRIVATE ${CVKIT_TENSORRT_INCLUDE_DIR})
    target_link_libraries(cvkit_infer PRIVATE ${CVKIT_TENSORRT_NVINFER_LIBRARY} ${CVKIT_TENSORRT_NVONNXPARSER_LIBRARY})
endif()

set_target_properties(cvkit_infer PROPERTIES
    OUTPUT_NAME cvkit_infer
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME infer
    RUNTIME_OUTPUT_DIRECTORY ${CVKIT_RUNTIME_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CVKIT_LIBRARY_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CVKIT_LIBRARY_OUTPUT_DIR}
)

install(
    TARGETS cvkit_infer
    EXPORT cvkitTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY
        ${PROJECT_SOURCE_DIR}/include/cvkit/infer/
        ${CMAKE_CURRENT_BINARY_DIR}/generated/cvkit/infer/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cvkit/infer
    FILES_MATCHING
        PATTERN "*.h"
)
