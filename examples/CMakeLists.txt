add_executable(cvkit_example_pipeline
    example_graph_utils.cpp
    example_opencv_utils.cpp
    pipeline_media_utils.cpp
    pipeline.cpp
    pipeline_support.cpp
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

target_compile_features(cvkit_example_pipeline PRIVATE cxx_std_17)
target_compile_features(cvkit_example_promptable_segmentation PRIVATE cxx_std_17)
target_compile_features(cvkit_example_classification PRIVATE cxx_std_17)

set_target_properties(cvkit_example_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

set_target_properties(cvkit_example_promptable_segmentation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

set_target_properties(cvkit_example_classification PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)
