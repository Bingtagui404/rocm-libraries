# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# rocBLAS benchmark client
add_executable(rocblas-bench)

target_sources(rocblas-bench
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/client.cpp"
)

# NOTE: All common include paths, compile definitions, and compile options are
# provided by rocblas-clients-helper. Only target-specific settings below.

# Link to helper interface to get common client properties
target_link_libraries(rocblas-bench
    PRIVATE
        rocblas-clients-helper
        rocblas-clients-testing-common
        rocblas-clients-common
)

target_compile_definitions(rocblas-bench PRIVATE ROCBLAS_BENCH)

if(ROCBLAS_ENABLE_BLIS)
    target_compile_definitions(rocblas-bench PRIVATE BLIS_ENABLE_NO_UNDERSCORE_API BLIS_ENABLE_CBLAS)
endif()

# Link to BLAS library for reference implementations
if(ROCBLAS_ENABLE_BLIS)
    target_link_libraries(rocblas-bench PRIVATE BLIS::BLIS)
else()
    target_link_libraries(rocblas-bench PRIVATE ${BLAS_LIBRARIES})
endif()

# OpenMP is linked based on whether it was found during configuration
if(OpenMP_CXX_FOUND)
    target_link_libraries(rocblas-bench PRIVATE OpenMP::OpenMP_CXX)
endif()

if(Threads_FOUND)
    target_link_libraries(rocblas-bench PRIVATE Threads::Threads)
endif()

# FLAG: -lm links math library, -lstdc++fs links C++17 filesystem library
if(NOT WIN32)
    target_link_libraries(rocblas-bench PRIVATE m stdc++fs)
endif()

if(ROCBLAS_ENABLE_FORTRAN)
    target_link_libraries(rocblas-bench PRIVATE rocblas-fortran-client)
endif()

set_target_properties(rocblas-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

# GEMM tuning client (only built when Tensile is enabled)
if(ROCBLAS_ENABLE_TENSILE)
    add_executable(rocblas-gemm-tune)

    target_sources(rocblas-gemm-tune
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/gemm_tune/gemm_tune_client.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/gemm_tune/gemm_tuners.cpp"
    )

    # Link to helper interface to get common client properties
    target_link_libraries(rocblas-gemm-tune
        PRIVATE
            rocblas-clients-helper
            rocblas-clients-common
    )

    # Need blas3 specific include for gemm tuning
    # PRIVATE includes don't need BUILD_INTERFACE wrapper
    target_include_directories(rocblas-gemm-tune
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3"
    )

    target_compile_definitions(rocblas-gemm-tune PRIVATE ROCBLAS_BENCH)

    if(ROCBLAS_ENABLE_BLIS)
        target_link_libraries(rocblas-gemm-tune PRIVATE BLIS::BLIS)
    else()
        target_link_libraries(rocblas-gemm-tune PRIVATE ${BLAS_LIBRARIES})
    endif()

    if(OpenMP_CXX_FOUND)
        target_link_libraries(rocblas-gemm-tune PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(Threads_FOUND)
        target_link_libraries(rocblas-gemm-tune PRIVATE Threads::Threads)
    endif()

    if(NOT WIN32)
        target_link_libraries(rocblas-gemm-tune PRIVATE m stdc++fs)
    endif()

    if(ROCBLAS_ENABLE_FORTRAN)
        target_link_libraries(rocblas-gemm-tune PRIVATE rocblas-fortran-client)
    endif()

    set_target_properties(rocblas-gemm-tune PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
    )
endif()

add_subdirectory(perf_script)
