find_package(benchmark REQUIRED)

add_executable(cvkit_benchmark_image
    image_benchmark.cpp
)

target_link_libraries(cvkit_benchmark_image
    PRIVATE
        cvkit::image
        benchmark::benchmark_main
)

target_compile_features(cvkit_benchmark_image PRIVATE cxx_std_17)
set_target_properties(cvkit_benchmark_image PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)
