add_executable(cvkit_example_pipeline
    example_graph_utils.cpp
    example_opencv_utils.cpp
    pipeline_media_utils.cpp
    pipeline.cpp
    pipeline_support.cpp
)

add_executable(cvkit_example_video_gpu_detection
    video_gpu_detection.cpp
)

add_executable(cvkit_example_promptable_segmentation
    example_graph_utils.cpp
    example_opencv_utils.cpp
    promptable_segmentation.cpp
    promptable_segmentation_support.cpp
)

add_executable(cvkit_example_classification
    example_graph_utils.cpp
    example_opencv_utils.cpp
    classification.cpp
)

add_executable(cvkit_example_face_detection
    example_graph_utils.cpp
    example_opencv_utils.cpp
    face_detection.cpp
)

add_executable(cvkit_example_segmentation
    example_graph_utils.cpp
    example_opencv_utils.cpp
    segmentation.cpp
)

add_executable(cvkit_example_pose
    example_graph_utils.cpp
    example_opencv_utils.cpp
    pose.cpp
)

add_executable(cvkit_example_facemesh
    example_graph_utils.cpp
    example_opencv_utils.cpp
    facemesh.cpp
)

target_link_libraries(cvkit_example_pipeline
    PRIVATE
        cvkit::media
        cvkit::image
        cvkit::infer
        basekit::cli11
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio
)

target_link_libraries(cvkit_example_video_gpu_detection
    PRIVATE
        cvkit::media
        cvkit::infer
        basekit::cli11
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
)

if(TARGET CUDA::cudart)
    target_compile_definitions(cvkit_example_video_gpu_detection PRIVATE CVKIT_VIDEO_GPU_DETECTION_WITH_CUDA_RUNTIME=1)
    target_link_libraries(cvkit_example_video_gpu_detection PRIVATE CUDA::cudart)
endif()

target_link_libraries(cvkit_example_promptable_segmentation
    PRIVATE
        cvkit::infer
        basekit::cli11
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
)

target_link_libraries(cvkit_example_classification
    PRIVATE
        cvkit::infer
        basekit::cli11
        opencv_core
        opencv_imgcodecs
)

target_link_libraries(cvkit_example_face_detection
    PRIVATE
        cvkit::infer
        basekit::cli11
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
)

target_link_libraries(cvkit_example_segmentation
    PRIVATE
        cvkit::infer
        basekit::cli11
        opencv_core
        opencv_imgcodecs
)

target_link_libraries(cvkit_example_pose
    PRIVATE
        cvkit::infer
        basekit::cli11
        opencv_core
        opencv_imgcodecs
)

target_link_libraries(cvkit_example_facemesh
    PRIVATE
        cvkit::infer
        basekit::cli11
        opencv_core
        opencv_imgcodecs
)

target_compile_features(cvkit_example_pipeline PRIVATE cxx_std_17)
target_compile_features(cvkit_example_video_gpu_detection PRIVATE cxx_std_17)
target_compile_features(cvkit_example_promptable_segmentation PRIVATE cxx_std_17)
target_compile_features(cvkit_example_classification PRIVATE cxx_std_17)
target_compile_features(cvkit_example_face_detection PRIVATE cxx_std_17)
target_compile_features(cvkit_example_segmentation PRIVATE cxx_std_17)
target_compile_features(cvkit_example_pose PRIVATE cxx_std_17)
target_compile_features(cvkit_example_facemesh PRIVATE cxx_std_17)

set_target_properties(cvkit_example_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

set_target_properties(cvkit_example_video_gpu_detection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

set_target_properties(cvkit_example_promptable_segmentation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

set_target_properties(cvkit_example_classification PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

set_target_properties(cvkit_example_face_detection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

set_target_properties(cvkit_example_segmentation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

set_target_properties(cvkit_example_pose PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

set_target_properties(cvkit_example_facemesh PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)
