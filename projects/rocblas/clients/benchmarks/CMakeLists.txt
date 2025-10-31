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
        rocblas::testing-common
        hip::device
)

target_compile_definitions(rocblas-bench PRIVATE ROCBLAS_BENCH)

if(ROCBLAS_ENABLE_TENSILE)
    add_executable(rocblas-gemm-tune)

    target_sources(rocblas-gemm-tune
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/gemm_tune/gemm_tune_client.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/gemm_tune/gemm_tuners.cpp"
    )

    target_link_libraries(rocblas-gemm-tune PRIVATE rocblas::clients-common hip::device)

    target_include_directories(rocblas-gemm-tune
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3"
    )

    target_compile_definitions(rocblas-gemm-tune PRIVATE ROCBLAS_BENCH)
endif()

add_subdirectory(perf_script)
