include(GenerateExportHeader)

add_library(cvkit_image
    ops.cpp
)

add_library(cvkit::image ALIAS cvkit_image)

generate_export_header(
    cvkit_image
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/generated/cvkit/image/image_export.h
    EXPORT_MACRO_NAME BK_IMAGE_EXPORT
)

target_include_directories(cvkit_image
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(cvkit_image
    PUBLIC
        cvkit::core
    PRIVATE
        opencv_core
        opencv_imgproc
)

target_compile_features(cvkit_image PUBLIC cxx_std_17)

set_target_properties(cvkit_image PROPERTIES
    OUTPUT_NAME cvkit_image
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME image
    RUNTIME_OUTPUT_DIRECTORY ${CVKIT_RUNTIME_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CVKIT_LIBRARY_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CVKIT_LIBRARY_OUTPUT_DIR}
)

install(
    TARGETS cvkit_image
    EXPORT cvkitTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES
    ${PROJECT_SOURCE_DIR}/include/cvkit/image/ops.h
    ${CMAKE_CURRENT_BINARY_DIR}/generated/cvkit/image/image_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cvkit/image
)
