if(EXISTS /cambricon)
    set(tools /opt/cambricon/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/)
    set(CMAKE_C_COMPILER ${tools}/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER ${tools}/bin/aarch64-linux-gnu-g++)
endif()
