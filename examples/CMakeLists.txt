add_executable(cvkit_example_pipeline
    pipeline.cpp
    pipeline_support.cpp
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

target_compile_features(cvkit_example_pipeline PRIVATE cxx_std_17)

set_target_properties(cvkit_example_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)
