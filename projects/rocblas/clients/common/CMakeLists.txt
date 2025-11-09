# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(rocblas-clients-common STATIC)
add_library(rocblas::clients-common ALIAS rocblas-clients-common)

target_link_libraries(rocblas-clients-common
    PRIVATE
        hip::device
        hip::host
        roc::rocblas
        rocblas::warnings
)

if(ROCBLAS_ENABLE_ROCM_SMI)
    target_link_libraries(rocblas-clients-common PRIVATE rocm_smi64)
endif()

if(NOT WIN32)
    target_link_libraries(rocblas-clients-common
        PRIVATE
            m
            stdc++fs
    )
endif()

target_include_directories(rocblas-clients-common
    PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clients/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
)

# FLAG: -Wno-c++17-extensions suppresses warnings from AMD Clang about extension usage
target_compile_options(rocblas-clients-common PRIVATE -Wno-c++17-extensions)

target_compile_definitions(rocblas-clients-common
    PRIVATE
        ROCBLAS_INTERNAL_API
        ROCBLAS_NO_DEPRECATED_WARNINGS
        ROCM_USE_FLOAT16
        __HIP_HCC_COMPAT_MODE__=1
)

if(ROCBLAS_ENABLE_TENSILE)
    target_compile_definitions(rocblas-clients-common
        PRIVATE
            BUILD_WITH_TENSILE
            ROCBLAS_TENSILE_LAZY_LOAD=1
            ROCBLAS_TENSILE_SEPARATE_ARCH=1
    )
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
    target_compile_definitions(rocblas-clients-common PRIVATE BUILD_WITH_HIPBLASLT)
endif()

if(NOT ROCBLAS_ENABLE_FORTRAN)
    target_compile_definitions(rocblas-clients-common PRIVATE CLIENTS_NO_FORTRAN)
endif()

if(ROCBLAS_ENABLE_ASAN)
    target_compile_definitions(rocblas-clients-common PRIVATE ASAN_BUILD)
endif()

if(ROCBLAS_ENABLE_BLIS)
    target_compile_definitions(rocblas-clients-common
        PRIVATE
            BLIS_ENABLE_CBLAS
            BLIS_ENABLE_NO_UNDERSCORE_API
    )
endif()

target_sources(rocblas-clients-common
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/argument_model.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/frequency_monitor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/host_alloc.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_arguments.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_parse_data.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_random.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/singletons.cpp"
)

if(ROCBLAS_ENABLE_FORTRAN)
    add_library(rocblas-fortran-client STATIC)
    add_library(rocblas::fortran-client ALIAS rocblas-fortran-client)

    target_link_libraries(rocblas-fortran-client
        PUBLIC
            rocblas::rocblas-fortran-module
    )

    target_include_directories(rocblas-fortran-client
        PRIVATE
            $<TARGET_PROPERTY:rocblas-fortran-module,Fortran_MODULE_DIRECTORY>
    )

    add_dependencies(rocblas-fortran-client rocblas-fortran-module)

    target_sources(rocblas-fortran-client
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran.f90"
    )

    if(ROCBLAS_ENABLE_TENSILE)
        target_sources(rocblas-fortran-client
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran_tensile.f90"
        )
    endif()
endif()
