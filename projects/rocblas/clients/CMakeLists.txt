# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

# This project builds client applications for rocBLAS
if(ROCBLAS_ENABLE_FORTRAN)
  project(rocblas-clients LANGUAGES CXX C Fortran)
else()
  project(rocblas-clients LANGUAGES CXX C)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# OpenMP is used for parallel reference implementations
find_package(OpenMP)

# Add client-specific CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(client-build-options)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required dependencies
find_package(rocblas REQUIRED CONFIG)
find_package(hip REQUIRED CONFIG)

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

# Custom targets for copying client support files
set(ROCBLAS_COMMON "${PROJECT_BINARY_DIR}/staging/rocblas_common.yaml")
add_custom_command(OUTPUT "${ROCBLAS_COMMON}"
                    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_common.yaml" "${ROCBLAS_COMMON}"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_common.yaml")

set(ROCBLAS_TEMPLATE "${PROJECT_BINARY_DIR}/staging/rocblas_template.yaml")
add_custom_command(OUTPUT "${ROCBLAS_TEMPLATE}"
                    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_template.yaml" "${ROCBLAS_TEMPLATE}"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_template.yaml")

set(ROCBLAS_GENERAL_YAML "${PROJECT_BINARY_DIR}/staging/rocblas_general.yaml")
add_custom_command(OUTPUT "${ROCBLAS_GENERAL_YAML}"
                    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_general.yaml" "${ROCBLAS_GENERAL_YAML}"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_general.yaml")

set(ROCBLAS_GENTEST "${PROJECT_BINARY_DIR}/staging/rocblas_gentest.py")
add_custom_command(OUTPUT "${ROCBLAS_GENTEST}"
                    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/common/rocblas_gentest.py" "${ROCBLAS_GENTEST}"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/common/rocblas_gentest.py")

set(ROCBLAS_CLIENTS_README "${PROJECT_BINARY_DIR}/staging/rocblas_clients_readme.txt")
add_custom_command(OUTPUT "${ROCBLAS_CLIENTS_README}"
                    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_clients_readme.txt" "${ROCBLAS_CLIENTS_README}"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_clients_readme.txt")

add_custom_target(rocblas-common DEPENDS "${ROCBLAS_COMMON}" "${ROCBLAS_TEMPLATE}" "${ROCBLAS_GENERAL_YAML}" "${ROCBLAS_GENTEST}" "${ROCBLAS_CLIENTS_README}")
