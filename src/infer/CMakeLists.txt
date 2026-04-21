include(GenerateExportHeader)

add_library(cvkit_infer
    model.cpp
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

if(CVKIT_ENABLE_OPENVINO)
    target_compile_definitions(cvkit_infer PRIVATE CVKIT_WITH_OPENVINO=1)
endif()

if(CVKIT_ENABLE_TENSORRT)
    target_compile_definitions(cvkit_infer PRIVATE CVKIT_WITH_TENSORRT=1)
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
    FILES
        ${PROJECT_SOURCE_DIR}/include/cvkit/infer/model.h
        ${CMAKE_CURRENT_BINARY_DIR}/generated/cvkit/infer/infer_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cvkit/infer
)
