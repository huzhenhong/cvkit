
option(BUILD_SHARED_LIBS "Build shared instead of static libraries." ON) # 开启后默认构建动态库

# 编译选项
# message("\n------------------ Start configure compile option ------------------")
if(USE_CXX_COMPILER MATCHES "MSVC")
    message("- configure MSVC compile option")
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/source-charset:utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
    # add_compile_options(SCL_SECURE_NO_WARNINGS)     # Calling any one of the potentially unsafe methods in the Standard C++ Library
    # add_compile_options(CRT_SECURE_NO_WARNINGS)     # Calling any one of the potentially unsafe methods in the CRT Library
    # add_compile_options(/MP)                        # build with multiple processes
    # add_compile_options(/W4)                        # warning level 4
    # add_compile_options(/WX)                        # treat warnings as errors
    # add_compile_options(/wd4251)                    # disable warning: 'identifier': class 'type' needs to have dll-interface to be used by clients of class 'type2'
    # add_compile_options(/wd4592)                    # disable warning: 'identifier': symbol will be dynamically initialized (implementation limitation)
    # add_compile_options(/wd4201)                    # disable warning: nonstandard extension used: nameless struct/union (caused by GLM)
    # add_compile_options(/wd4127)                    # disable warning: conditional expression is constant (caused by Qt)

    # $<$<CONFIG:Debug>:
    #     add_compile_options(/RTCc)                  # value is assigned to a smaller data type and results in a data loss
    # >

    # $<$<CONFIG:Release>:
    # /Gw                                             # whole program global optimization
    # /GS-                                            # buffer security check: no
    # /GL                                             # whole program optimization: enable link-time code generation (disables Zi)
    # /GF                                             # enable string pooling
    # >
    # add_compile_options(/Od)
    string(REPLACE "/O2" "/Od" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Od")
    add_definitions(-DNOMINMAX) # 去除windows min max 函数影响
elseif(USE_CXX_COMPILER MATCHES "GNU")
    message("- configure GNU compile option")
    # add_compile_options(-Wall)·
    # add_compile_options(-Wextra)
    # add_compile_options(-Wunused)
    # add_compile_options(-Wreorder)
    # add_compile_options(-Wignored-qualifiers)
    # add_compile_options(-Wmissing-braces)
    # add_compile_options(-Wreturn-type)
    # add_compile_options(-Wswitch)
    # add_compile_options(-Wswitch-default)
    # add_compile_options(-Wuninitialized)
    # add_compile_options(-Wmissing-field-initializers)
    # add_compile_options(
    #     $<$<CXX_COMPILER_ID:GNU>:
    #         -Wmaybe-uninitialized
    #         $<$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,4.8>:
    #             -Wpedantic
    #             -Wreturn-local-addr
    #         >
    #     >)
    # add_compile_options(
    #     $<$<BOOL:${OPTION_COVERAGE_ENABLED}>:
    #         -fprofile-arcs
    #         -ftest-coverage
    #     >)

    # string(REPLACE "-O2" "-O0" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    # add_compile_options(-O0)
    # add_compile_options("$<$<CONFIG:Debug>:-O0>")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
elseif(USE_CXX_COMPILER MATCHES "Clang")
    message("- configure Clang compile option")
    # add_compile_options(
    #     $<$<CXX_COMPILER_ID:Clang>:
    #         -Wpedantic
    #         -Wreturn-stack-address          # gives false positives
    # >)
    add_compile_options(-O0)
    add_compile_options(-g)
endif()

# if(CMAKE_BUILD_TYPE MATCHES "Debug")
#     set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
#     set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
# endif()

# message("CMAKE_CXX_FLAGS_MINSIZEREL: ${CMAKE_CXX_FLAGS_MINSIZEREL}")
# message("CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
# message("CMAKE_CXX_FLAGS_RELWITHDEBINFO: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
# message("------------------ Stop configure compile option ------------------\n")

# 编译宏定义
# message("\n------------------ Start configure compile definitions ------------------")
if(USE_CXX_COMPILER MATCHES "MSVC")
    message("- configure MSVC compile definitions")
    add_compile_definitions(_SCL_SECURE_NO_WARNINGS
    )# Calling any one of the potentially unsafe methods in the Standard C++ Library
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS
    )# Calling any one of the potentially unsafe methods in the CRT Library
elseif(USE_CXX_COMPILER MATCHES "GNU")
    message("- configure GNU compile definitions")

elseif(USE_CXX_COMPILER MATCHES "Clang")
    message("- configure Clang compile definitions")

endif()
# message("------------------ Stop configure compile definitions ------------------\n")

# 链接配置
# message("\n------------------ Start configure link option ------------------")
if(USE_SYSTEM_PLATFORM MATCHES "WINDOWS")
    if(USE_CXX_COMPILER MATCHES "MSVC")
        if(USE_X64)
            link_directories(${CMAKE_SOURCE_DIR}/3rdparty/msvc-64)
            message("- Using msvc-64 to link")
        else()
            link_directories(${CMAKE_SOURCE_DIR}/3rdparty/msvc-32)
            message("- Using msvc-32 to link")
        endif()
    else()
        if(USE_X64)
            link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-win64)
            message("- Using lib-win64 to link")
        else()
            link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-win32)
            message("- Using lib-win32 to link")
        endif()
    endif()
elseif(USE_SYSTEM_PLATFORM MATCHES "LINUX")
    if(USE_X64)
        link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-linux-64)
        message("- Using lib-linux64 to link")
    else()
        link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-linux-32)
        message("- Using lib-linux32 to link")
    endif()
    # add_link_options(pthread)
    # add_link_options(stdc++fs)
elseif(USE_SYSTEM_PLATFORM MATCHES "OSX")
    if(USE_X64)
        link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-macos-64)
        message("- Using lib-macos64 to link")
    else()
        link_directories(${CMAKE_SOURCE_DIR}/3rdparty/lib-macos-32)
        message("- Using lib-macos32 to link")
    endif()
else()
    message("- unkown system platform")
endif()
# message("------------------ Stop configure link option ------------------\n")
