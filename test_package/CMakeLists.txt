cmake_minimum_required(VERSION 3.23)

project(cvkit_test_package LANGUAGES CXX)

find_package(cvkit REQUIRED CONFIG)

enable_testing()

set(CVKIT_TEST_PACKAGE_RUNTIME_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")

add_executable(cvkit_test_package
    src/main.cpp
)

target_compile_features(cvkit_test_package PRIVATE cxx_std_17)

target_link_libraries(cvkit_test_package
    PRIVATE
        cvkit::media
        cvkit::image
        cvkit::infer
)

set_target_properties(cvkit_test_package PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CVKIT_TEST_PACKAGE_RUNTIME_OUTPUT_DIR}
)

add_test(NAME cvkit_test_package COMMAND ${CVKIT_TEST_PACKAGE_RUNTIME_OUTPUT_DIR}/cvkit_test_package)
set_tests_properties(cvkit_test_package PROPERTIES
    WORKING_DIRECTORY ${CVKIT_TEST_PACKAGE_RUNTIME_OUTPUT_DIR}
)
