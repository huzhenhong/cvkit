find_package(Catch2 REQUIRED)
include(Catch)

add_executable(cvkit_test_core
    test_core.cpp
)

target_link_libraries(cvkit_test_core
    PRIVATE
        cvkit::core
        Catch2::Catch2WithMain
)

target_compile_features(cvkit_test_core PRIVATE cxx_std_17)
set_target_properties(cvkit_test_core PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)
catch_discover_tests(cvkit_test_core)

add_executable(cvkit_test_infer
    test_infer.cpp
)

target_link_libraries(cvkit_test_infer
    PRIVATE
        cvkit::infer
        opencv_imgcodecs
        Catch2::Catch2WithMain
)

if(CVKIT_ENABLE_TENSORRT)
    target_compile_definitions(cvkit_test_infer PRIVATE CVKIT_WITH_TENSORRT=1)
endif()

target_compile_features(cvkit_test_infer PRIVATE cxx_std_17)
set_target_properties(cvkit_test_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)
catch_discover_tests(cvkit_test_infer)
