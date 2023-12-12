set(LIB_NAME basekit)

include(FetchContent)

FetchContent_Declare(
    ${LIB_NAME}
    QUIET
    GIT_REPOSITORY https://github.com/huzhenhong/${LIB_NAME}.git
    # URL https://github.com/open-source-parsers/jsoncpp/archive/1.9.4.tar.gz)
    GIT_TAG main
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/ext/${LIB_NAME})
FetchContent_MakeAvailable(${LIB_NAME})
