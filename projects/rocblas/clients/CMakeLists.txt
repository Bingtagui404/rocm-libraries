# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# This file builds the rocBLAS clients (benchmarks, tests, samples). It is always included as a
# subdirectory from the parent build - standalone builds are not supported.

# Threading support for clients
if(ROCBLAS_ENABLE_THREADS)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
endif()

# OpenMP is used for parallel reference implementations
if(ROCBLAS_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# Add client-specific CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Note: client-build-options.cmake removed - options now managed in root CMakeLists.txt

if(ROCBLAS_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

if(ROCBLAS_ENABLE_BENCHMARKS OR ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
    # Find BLAS library for reference implementations in tests/benchmarks
    if(ROCBLAS_ENABLE_BLIS)
        find_package(BLIS REQUIRED)
    else()
        find_package(BLAS REQUIRED)
    endif()

    # Find ROCmSMI for monitoring (Linux only)
    if(NOT WIN32 AND ROCBLAS_ENABLE_ROCM_SMI)
        find_package(ROCmSMI REQUIRED)
    endif()

    find_package(GTest REQUIRED)

    add_subdirectory(common)

    if(ROCBLAS_ENABLE_BENCHMARKS)
        add_subdirectory(benchmarks)
    endif()

    if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
        add_subdirectory(gtest)
    endif()
endif()

# Custom targets for copying client support files to staging directory
set(ROCBLAS_CLIENT_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_common.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_template.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_general.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/rocblas_gentest.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_clients_readme.txt"
)

set(ROCBLAS_CLIENT_FILES_STAGED "")
foreach(file ${ROCBLAS_CLIENT_FILES})
    get_filename_component(filename ${file} NAME)
    set(staged_file "${PROJECT_BINARY_DIR}/staging/${filename}")
    add_custom_command(
        OUTPUT "${staged_file}" COMMAND "${CMAKE_COMMAND}" -E copy "${file}" "${staged_file}"
        DEPENDS "${file}"
    )
    list(APPEND ROCBLAS_CLIENT_FILES_STAGED "${staged_file}")
endforeach()

add_custom_target(rocblas-common DEPENDS ${ROCBLAS_CLIENT_FILES_STAGED})
