# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(ROCBLAS_ENABLE_THREADS)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
endif()

if(ROCBLAS_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(ROCBLAS_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

if(ROCBLAS_ENABLE_BENCHMARKS OR ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
    if(ROCBLAS_ENABLE_BLIS)
        find_package(BLIS REQUIRED)
    else()
        find_package(BLAS REQUIRED)
    endif()

    if(NOT WIN32 AND ROCBLAS_ENABLE_ROCM_SMI)
        find_package(ROCmSMI REQUIRED)
    endif()

    find_package(GTest REQUIRED)

    add_subdirectory(common)

    if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
        add_subdirectory(gtest)
    endif()

    if(ROCBLAS_ENABLE_BENCHMARKS)
        add_subdirectory(benchmarks)
    endif()
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_common.yaml"
               "${PROJECT_BINARY_DIR}/staging/rocblas_common.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_template.yaml"
               "${PROJECT_BINARY_DIR}/staging/rocblas_template.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_general.yaml"
               "${PROJECT_BINARY_DIR}/staging/rocblas_general.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/common/rocblas_gentest.py"
               "${PROJECT_BINARY_DIR}/staging/rocblas_gentest.py" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/rocblas_clients_readme.txt"
               "${PROJECT_BINARY_DIR}/staging/rocblas_clients_readme.txt" COPYONLY)
