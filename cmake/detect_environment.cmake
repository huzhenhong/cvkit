
# 检测操作系统
if(${CMAKE_VERSION} GREATER_EQUAL 3.22)
    cmake_host_system_information(RESULT DISTRO QUERY DISTRIB_INFO)
    foreach(VAR IN LISTS DISTRO)
        message(STATUS "${VAR}=`${${VAR}}`")
    endforeach()
endif()

# 判断交叉编译
if(CMAKE_CROSSCOMPILING)
	message(STATUS "Cross Comliling!!!,  ARM")
endif()

# message("\n------------------ Start detect system platform ------------------")
message("- host system name is: ${CMAKE_SYSTEM_NAME}")
if(WIN32)
    set(USE_SYSTEM_PLATFORM "WINDOWS")
    add_compile_definitions(IS_WINDOWS)
elseif(APPLE)
    if(IOS)
        set(USE_SYSTEM_PLATFORM "IOS")
    else()
        set(USE_SYSTEM_PLATFORM "OSX")
        add_compile_definitions(IS_OSX)
    endif()
elseif(ANDROID)
    set(USE_SYSTEM_PLATFORM "ANDROID")
elseif(UNIX AND NOT APPLE)
    execute_process(COMMAND bash "-c" "cat /etc/*-release" RESULT_VARIABLE rv OUTPUT_VARIABLE out)
    if (${rv})
    	message(FATAL_ERROR "command execute error")
    else()
        if(out MATCHES ".*Ubuntu.*")
        	option(USE_IS_UBUNTU "is the ubuntu" true)
        elseif(out MATCHES ".*CentOS.*")
            option(USE_IS_CENTOS "is the CentOS" true)
        elseif(out MATCHES ".*FreeBSD.*")
            option(USE_IS_FreeBSD "is the FreeBSD" true)
        else()
        	message(WARNING "Unknown Linux Branch")
            set(USE_SYSTEM_PLATFORM "UNKNOWN")
        endif()
        set(USE_SYSTEM_PLATFORM "LINUX")
        add_compile_definitions(IS_LINUX)
    endif()
else()
    set(USE_SYSTEM_PLATFORM "UNKNOWN")
    message(WARNING "Unknown system")
endif()
# message("------------------ Stop detect system platform ------------------\n")

# 检测编译器
# message("\n------------------ Start detect compiler id ------------------")
message("- compiler id is: ${CMAKE_CXX_COMPILER_ID}")
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(USE_CXX_COMPILER "MSVC")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(USE_CXX_COMPILER "GNU")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    set(USE_CXX_COMPILER "Clang")
else()
    set(USE_CXX_COMPILER "Unknown compiler")
    message(WARNING "Unknown compiler")
endif()
# message("------------------ Stop detect compiler id ------------------\n")


# 检测处理器体系架构
# message("\n------------------ Start detect compiler processor architecture ------------------")
message("- host processor architecture is: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i386")
    set(USE_HOST_SYSTEM_PROCESSOR "i386")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    set(USE_HOST_SYSTEM_PROCESSOR "arm64")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(USE_HOST_SYSTEM_PROCESSOR "x86_64")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "AMD64")
    set(USE_HOST_SYSTEM_PROCESSOR "AMD64")
else()
    message(WARNING "Unknown host processor architecture")
endif()
# message("------------------ Stop detect compiler processor architecture ------------------\n")


# 检测处理器位宽
# message("\n------------------ Start detect processor bit wide ------------------")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(USE_X64 true)
    message("- processor is 64 bits")
else()
    set(USE_X64 false)
    message("- processor is 32 bits")
endif()
# message("------------------ Stop detect processor bit wide ------------------\n")
