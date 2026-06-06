include(GenerateExportHeader)

add_library(cvkit_media
    source.cpp
    writer.cpp
)

add_library(cvkit::media ALIAS cvkit_media)

generate_export_header(
    cvkit_media
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/generated/cvkit/media/media_export.h
    EXPORT_MACRO_NAME BK_MEDIA_EXPORT
)

target_include_directories(cvkit_media
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(cvkit_media
    PUBLIC
        cvkit::core
        basekit::log
    PRIVATE
        ${OpenCV_LIBS}
)

target_compile_features(cvkit_media PUBLIC cxx_std_17)

if(CVKIT_ENABLE_GSTREAMER)
    target_compile_definitions(cvkit_media PRIVATE CVKIT_WITH_GSTREAMER=1)
    target_link_libraries(cvkit_media PRIVATE PkgConfig::GST)
endif()

set_target_properties(cvkit_media PROPERTIES
    OUTPUT_NAME cvkit_media
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME media
    RUNTIME_OUTPUT_DIRECTORY ${CVKIT_RUNTIME_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CVKIT_LIBRARY_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CVKIT_LIBRARY_OUTPUT_DIR}
)

install(
    TARGETS cvkit_media
    EXPORT cvkitTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES
    ${PROJECT_SOURCE_DIR}/include/cvkit/media/source.h
    ${PROJECT_SOURCE_DIR}/include/cvkit/media/writer.h
    ${PROJECT_SOURCE_DIR}/include/cvkit/media/options.h
    ${CMAKE_CURRENT_BINARY_DIR}/generated/cvkit/media/media_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cvkit/media
)
