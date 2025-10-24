# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Set output directory for all benchmarks at directory level
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

# rocBLAS benchmark client
add_executable(rocblas-bench)

target_sources(rocblas-bench
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/client.cpp"
)

# NOTE: All common include paths, compile definitions, compile options, and link libraries
# (rocblas, hip, OpenMP, Threads, BLAS, m, stdc++fs) are provided by the common targets.
# Only target-specific settings below.

target_link_libraries(rocblas-bench
    PRIVATE
        rocblas-clients-helper
        rocblas-clients-testing-common
        rocblas-clients-common
        rocblas-benchmark-common
)

target_compile_definitions(rocblas-bench PRIVATE ROCBLAS_BENCH)

# GEMM tuning client (only built when Tensile is enabled)
if(ROCBLAS_ENABLE_TENSILE)
    add_executable(rocblas-gemm-tune)

    target_sources(rocblas-gemm-tune
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/gemm_tune/gemm_tune_client.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/gemm_tune/gemm_tuners.cpp"
    )

    # NOTE: All common properties provided by common targets.
    target_link_libraries(rocblas-gemm-tune
        PRIVATE
            rocblas-clients-helper
            rocblas-clients-common
            rocblas-benchmark-common
    )

    # Need blas3 specific include for gemm tuning
    target_include_directories(rocblas-gemm-tune
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3"
    )

    target_compile_definitions(rocblas-gemm-tune PRIVATE ROCBLAS_BENCH)
endif()

add_subdirectory(perf_script)
