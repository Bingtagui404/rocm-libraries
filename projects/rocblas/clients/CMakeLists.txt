# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# samples only requires four source files from clients common so we may want to
# create another object library for those four files.
#target_sources(rocblas-samples-common
#    PRIVATE
#        "${CMAKE_CURRENT_SOURCE_DIR}/../common/singletons.cpp"
#        "${CMAKE_CURRENT_SOURCE_DIR}/../common/client_utility.cpp"
#        "${CMAKE_CURRENT_SOURCE_DIR}/../common/rocblas_random.cpp"
#        "${CMAKE_CURRENT_SOURCE_DIR}/../common/host_alloc.cpp"
#)

if(ROCBLAS_ENABLE_THREADS)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
endif()

if(ROCBLAS_ENABLE_OPENMP)
    # We prefer to find the rocm llvm openmp which is config based
    # and located in <rocm install>/llvm/lib/cmake/openmp
    # which requires adding this path to the prefix path.
    find_package(OpenMP CONFIG)

    # If we find the rocm llvm omp then look for the gnu
    # omp and create an alias so we have uniform treatment.
    if(OpenMP_FOUND)
        add_library(rocm-omp-shim INTERFACE)
        set_target_properties(rocm-omp-shim
            PROPERTIES
                INTERFACE_LINK_LIBRARIES OpenMP::omp
                INTERFACE_COMPILE_OPTIONS -fopenmp
        )
        add_library(OpenMP::OpenMP_CXX ALIAS rocm-omp-shim) # will this be an issue in the super build?
    else()
        message(STATUS "Did not find rocm-omp - falling back to gomp")
        find_package(OpenMP REQUIRED)
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_library(rocblas-clients-helper INTERFACE)
add_library(rocblas::clients-helper ALIAS rocblas-clients-helper)

target_include_directories(rocblas-clients-helper
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/blas1>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/blas2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/blas3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
    # TODO EVALUATE: May be redundant with roc::rocblas link propagation
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    # TODO EVALUATE: Internal rocBLAS headers - needed for some tests accessing internals
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/blas1>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/blas2>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/blas3>
)

target_compile_definitions(rocblas-clients-helper
  INTERFACE
    ROCM_USE_FLOAT16
    ROCBLAS_INTERNAL_API
    ROCBLAS_NO_DEPRECATED_WARNINGS
    rocblas_EXPORTS
)

if(ROCBLAS_ENABLE_TENSILE)
  target_compile_definitions(rocblas-clients-helper INTERFACE BUILD_WITH_TENSILE)
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
  target_compile_definitions(rocblas-clients-helper INTERFACE BUILD_WITH_HIPBLASLT)
endif()

if(NOT ROCBLAS_ENABLE_FORTRAN)
  target_compile_definitions(rocblas-clients-helper INTERFACE CLIENTS_NO_FORTRAN)
endif()

if(ROCBLAS_ENABLE_ASAN)
    target_compile_definitions(rocblas-clients-helper INTERFACE ASAN_BUILD)
endif()

if(ROCBLAS_ENABLE_BLIS)
    target_compile_definitions(rocblas-clients-helper INTERFACE
        BLIS_ENABLE_NO_UNDERSCORE_API
        BLIS_ENABLE_CBLAS)
endif()

# FLAG: -Wno-c++17-extensions suppresses warnings from AMD Clang about extension usage
target_compile_options(rocblas-clients-helper INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wno-c++17-extensions>)

target_link_libraries(rocblas-clients-helper
    INTERFACE
        roc::rocblas
        hip::host
        rocblas::warnings
)

if(ROCBLAS_ENABLE_BENCHMARKS OR ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
    if(ROCBLAS_ENABLE_BLIS)
        find_package(BLIS REQUIRED) # are we doing this in multiple places?
    else()
        find_package(BLAS REQUIRED) # are we doing this in multiple places?
    endif()

    if(ROCBLAS_ENABLE_ROCM_SMI)
        find_package(ROCmSMI REQUIRED)
    endif()

    find_package(GTest REQUIRED)

    add_library(rocblas-testing-common OBJECT)
    add_library(rocblas::testing-common ALIAS rocblas-testing-common)
    set_target_properties(rocblas-testing-common PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_subdirectory(common)

    add_subdirectory(gtest)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_smoke.yaml"
                   "${PROJECT_BINARY_DIR}/staging/rocblas_smoke.yaml" COPYONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_extras.yaml"
                   "${PROJECT_BINARY_DIR}/staging/rocblas_extras.yaml" COPYONLY)
    configure_file("${PROJECT_SOURCE_DIR}/rtest.py"
                   "${PROJECT_BINARY_DIR}/staging/rocblas_rtest.py" COPYONLY)
    configure_file("${PROJECT_SOURCE_DIR}/rtest.xml"
                   "${PROJECT_BINARY_DIR}/staging/rocblas_rtest.xml" COPYONLY)
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
endif()

if(ROCBLAS_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(ROCBLAS_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()
