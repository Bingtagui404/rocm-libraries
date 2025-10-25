# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(rocblas-clients-helper INTERFACE)
add_library(rocblas::clients-helper ALIAS rocblas-clients-helper)

target_include_directories(rocblas-clients-helper
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas1>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    # TODO EVALUATE: May be redundant with roc::rocblas link propagation
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    # TODO EVALUATE: Internal rocBLAS headers - needed for some tests accessing internals
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/blas1>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/blas2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/blas3>
)

target_compile_definitions(rocblas-clients-helper
  INTERFACE
    ROCM_USE_FLOAT16
    ROCBLAS_INTERNAL_API
    ROCBLAS_NO_DEPRECATED_WARNINGS
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
        hip::device
)

add_library(rocblas-clients-common OBJECT)
add_library(rocblas::clients-common ALIAS rocblas-clients-common)

target_link_libraries(rocblas-clients-common PUBLIC rocblas::clients-helper)

target_compile_definitions(rocblas-clients-common PRIVATE GOOGLE_TEST)
target_link_libraries(rocblas-clients-common
  PUBLIC
    $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
    $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main>
)

if(NOT WIN32 AND ROCBLAS_ENABLE_ROCM_SMI)
    target_link_libraries(rocblas-clients-common PUBLIC rocm_smi64)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(rocblas-clients-common PUBLIC OpenMP::OpenMP_CXX)
endif()

if(Threads_FOUND)
    target_link_libraries(rocblas-clients-common PUBLIC Threads::Threads)
endif()

if(NOT WIN32)
    target_link_libraries(rocblas-clients-common PUBLIC m stdc++fs)
endif()

target_sources(rocblas-clients-common
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/singletons.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/client_omp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/client_utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/frequency_monitor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_arguments.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/argument_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_random.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_parse_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_alloc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gtest_helpers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_matrix.cpp"
)

if(ROCBLAS_ENABLE_FORTRAN)
    add_library(rocblas-fortran-wrappers STATIC)
    add_library(rocblas::fortran-wrappers ALIAS rocblas-fortran-wrappers)
    target_sources(rocblas-fortran-wrappers
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran.f90"
    )
    if(ROCBLAS_ENABLE_TENSILE)
        target_sources(rocblas-fortran-wrappers
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran_tensile.f90"
        )
    endif()
    add_dependencies(rocblas-fortran-wrappers rocblas-fortran-module)
    target_include_directories(rocblas-fortran-wrappers
        PRIVATE
            $<TARGET_PROPERTY:rocblas-fortran-module,Fortran_MODULE_DIRECTORY>
    )

    add_library(rocblas-fortran-client INTERFACE)
    target_link_libraries(rocblas-fortran-client INTERFACE
        rocblas::rocblas-fortran-module
        rocblas::fortran-wrappers
    )
    target_include_directories(rocblas-fortran-client
        INTERFACE
            $<TARGET_PROPERTY:rocblas-fortran-module,Fortran_MODULE_DIRECTORY>
    )
endif()
