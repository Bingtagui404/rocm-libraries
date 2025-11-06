# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")

# Prefer ROCm LLVM OpenMP (config-based) located in <rocm install>/llvm/lib/cmake/openmp
find_package(OpenMP CONFIG)

if(OpenMP_FOUND)
    add_library(rocm-omp-shim INTERFACE)
    # TODO: Investigate if this alias causes issues in superbuild
    add_library(OpenMP::OpenMP_CXX ALIAS rocm-omp-shim)

    set_target_properties(rocm-omp-shim
        PROPERTIES
            INTERFACE_COMPILE_OPTIONS -fopenmp
            INTERFACE_LINK_LIBRARIES OpenMP::omp
    )
else()
    message(STATUS "Did not find rocm-omp - falling back to gomp")
    find_package(OpenMP REQUIRED)
endif()

add_library(rocblas-clients-openmp STATIC)
add_library(rocblas::clients-openmp ALIAS rocblas-clients-openmp)

target_link_libraries(rocblas-clients-openmp
    PUBLIC
        OpenMP::OpenMP_CXX
    PRIVATE
        hip::device
        roc::rocblas
)

target_include_directories(rocblas-clients-openmp
    PRIVATE
        "${PROJECT_SOURCE_DIR}/clients/include"
        "${PROJECT_SOURCE_DIR}/library/src/include"
)

target_compile_definitions(rocblas-clients-openmp
    PRIVATE
        ROCBLAS_INTERNAL_API
        ROCBLAS_NO_DEPRECATED_WARNINGS
        ROCM_USE_FLOAT16
        __HIP_HCC_COMPAT_MODE__=1
)

target_sources(rocblas-clients-openmp PRIVATE "${PROJECT_SOURCE_DIR}/clients/common/client_omp.cpp")

add_executable(rocblas-example-openmp)

target_link_libraries(rocblas-example-openmp
    PRIVATE
        hip::host
        roc::rocblas
        rocblas::clients-openmp
        rocblas::warnings
)

target_include_directories(rocblas-example-openmp PRIVATE "${PROJECT_BINARY_DIR}/library/include")

# Uses C++14 to demonstrate OpenMP with C++14 standard
set_target_properties(rocblas-example-openmp PROPERTIES CXX_STANDARD 14)

# NOTE: CXX_EXTENSIONS inherited from global CMAKE_CXX_EXTENSIONS=OFF setting

target_compile_definitions(rocblas-example-openmp PRIVATE ROCBLAS_INTERNAL_API ROCM_USE_FLOAT16)

target_sources(rocblas-example-openmp PRIVATE "${PROJECT_SOURCE_DIR}/clients/samples/example_openmp.cpp")
