# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

add_executable(rocblas-bench)

target_sources(rocblas-bench
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/client.cpp"
)

target_link_libraries(rocblas-bench 
    PRIVATE
        roc::rocblas
        rocblas::warnings
        rocblas::testing-common
        hip::device
)

target_include_directories(rocblas-bench
    PRIVATE
        "${PROJECT_SOURCE_DIR}/clients/common"
        "${PROJECT_SOURCE_DIR}/clients/include"
        "${PROJECT_SOURCE_DIR}/library/src/include"
)

target_compile_definitions(rocblas-bench
    PRIVATE
        ROCBLAS_BENCH
        ROCM_USE_FLOAT16
        ROCBLAS_NO_DEPRECATED_WARNINGS
        GOOGLE_TEST
)

if(ROCBLAS_ENABLE_TENSILE)
    add_executable(rocblas-gemm-tune)

    target_sources(rocblas-gemm-tune
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/gemm_tune/gemm_tune_client.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/gemm_tune/gemm_tuners.cpp"
    )

    target_link_libraries(rocblas-gemm-tune
        PRIVATE
            roc::rocblas
            rocblas::warnings
            rocblas::testing-common
            hip::device
    )

    target_include_directories(rocblas-gemm-tune
        PRIVATE
            "${PROJECT_SOURCE_DIR}/clients/include"
            "${PROJECT_SOURCE_DIR}/clients/include/blas3"
    )

    target_compile_definitions(rocblas-gemm-tune
        PRIVATE
            ROCBLAS_BENCH
            ROCM_USE_FLOAT16
            ROCBLAS_NO_DEPRECATED_WARNINGS
            #__HIP_HCC_COMPAT_MODE__=1
            #ROCBLAS_INTERNAL_API
    )
    if(ROCBLAS_ENABLE_TENSILE)
        target_compile_definitions(rocblas-gemm-tune PRIVATE
            BUILD_WITH_TENSILE
            ROCBLAS_TENSILE_SEPARATE_ARCH=1
            ROCBLAS_TENSILE_LAZY_LOAD=1
        )
    endif()
endif()

add_subdirectory(perf_script)
