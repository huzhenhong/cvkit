include(GenerateExportHeader)

add_library(cvkit_core
    types.cpp
)

add_library(cvkit::core ALIAS cvkit_core)

generate_export_header(
    cvkit_core
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/generated/cvkit/core/core_export.h
    EXPORT_MACRO_NAME BK_CORE_EXPORT
)

target_include_directories(cvkit_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(cvkit_core
    PUBLIC
        basekit::core
)

target_compile_features(cvkit_core PUBLIC cxx_std_17)

set_target_properties(cvkit_core PROPERTIES
    OUTPUT_NAME cvkit_core
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME core
    RUNTIME_OUTPUT_DIRECTORY ${CVKIT_RUNTIME_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CVKIT_LIBRARY_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CVKIT_LIBRARY_OUTPUT_DIR}
)

install(
    TARGETS cvkit_core
    EXPORT cvkitTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES
    ${PROJECT_SOURCE_DIR}/include/cvkit/core/types.h
    ${CMAKE_CURRENT_BINARY_DIR}/generated/cvkit/core/core_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cvkit/core
)
