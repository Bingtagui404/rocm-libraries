# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(ROCBLAS_ENABLE_THREADS)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(ROCBLAS_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

add_subdirectory(common)

if(ROCBLAS_ENABLE_BENCHMARKS OR ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
    if(ROCBLAS_ENABLE_BLIS)
        find_package(BLIS REQUIRED)
    else()
        find_package(BLAS REQUIRED)
        find_package(cblas)
    endif()

    if(ROCBLAS_ENABLE_ROCM_SMI)
        find_package(rocm_smi REQUIRED)
    endif()

    find_package(GTest REQUIRED)

    add_library(rocblas-testing-common STATIC)
    add_library(rocblas::testing-common ALIAS rocblas-testing-common)

    target_link_libraries(rocblas-testing-common
        PRIVATE
            $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
            $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main>
            hip::device
            roc::rocblas
            rocblas::clients-common
            rocblas::warnings
    )

    if(ROCBLAS_ENABLE_BLIS)
        target_link_libraries(rocblas-testing-common PRIVATE BLIS::BLIS)
        target_sources(rocblas-testing-common
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/common/blis_interface.cpp"
        )
    else()
        target_link_libraries(rocblas-testing-common PRIVATE ${BLAS_LIBRARIES})
        if(TARGET cblas)
            target_link_libraries(rocblas-testing-common PRIVATE cblas)
        endif()
        if(BLAS_INCLUDE_DIRS)
            target_include_directories(rocblas-testing-common SYSTEM PRIVATE ${BLAS_INCLUDE_DIRS})
        endif()
    endif()

    if(ROCBLAS_ENABLE_FORTRAN)
        target_link_libraries(rocblas-testing-common PRIVATE rocblas-fortran-client)
    endif()

    target_include_directories(rocblas-testing-common
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/blas1"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/blas2"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/blas3"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/blas_ex"
            "${PROJECT_SOURCE_DIR}/library/src"
            "${PROJECT_SOURCE_DIR}/library/src/include"
    )

    target_compile_definitions(rocblas-testing-common
        PRIVATE
            GOOGLE_TEST
            ROCBLAS_INTERNAL_API
            ROCBLAS_NO_DEPRECATED_WARNINGS
            ROCM_USE_FLOAT16
            __HIP_HCC_COMPAT_MODE__=1
    )

    if(ROCBLAS_ENABLE_TENSILE OR ROCBLAS_ENABLE_HIPBLASLT)
        target_compile_definitions(rocblas-testing-common
            PRIVATE
                ROCBLAS_TENSILE_LAZY_LOAD=1
                ROCBLAS_TENSILE_SEPARATE_ARCH=1
        )
    endif()

    if(ROCBLAS_ENABLE_TENSILE)
        target_compile_definitions(rocblas-testing-common
            PRIVATE
                BUILD_WITH_TENSILE
        )
    endif()

    if(ROCBLAS_ENABLE_HIPBLASLT)
        target_compile_definitions(rocblas-testing-common
            PRIVATE
                BUILD_WITH_HIPBLASLT
        )
        target_link_libraries(rocblas-testing-common PRIVATE roc::hipblaslt)
    endif()

    if(ROCBLAS_ENABLE_ASAN)
        rocblas_target_configure_sanitizers(rocblas-testing-common PRIVATE)
    endif()

    target_sources(rocblas-testing-common
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/common/cblas_interface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/common/gtest_helpers.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/common/client_utility.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/common/rocblas_matrix.cpp"
    )

    add_subdirectory(common/blas1)
    add_subdirectory(common/blas2)
    add_subdirectory(common/blas3)
    add_subdirectory(common/blas_ex)
    add_subdirectory(gtest)

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_smoke.yaml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_smoke.yaml" COPYONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_extras.yaml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_extras.yaml" COPYONLY)
    configure_file("${PROJECT_SOURCE_DIR}/rtest.py" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_rtest.py" COPYONLY)
    configure_file("${PROJECT_SOURCE_DIR}/rtest.xml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_rtest.xml" COPYONLY)
endif()

if(ROCBLAS_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(ROCBLAS_ENABLE_OPENMP)
    add_subdirectory(openmp-support)

    if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
        target_link_libraries(rocblas-test PRIVATE rocblas::clients-openmp)
    endif()

    if(ROCBLAS_ENABLE_BENCHMARKS)
        target_link_libraries(rocblas-bench PRIVATE rocblas::clients-openmp)
        if(ROCBLAS_ENABLE_TENSILE)
            target_link_libraries(rocblas-gemm-tune PRIVATE rocblas::clients-openmp)
        endif()
    endif()
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_common.yaml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_common.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_template.yaml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_template.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_general.yaml" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_general.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/common/rocblas_gentest.py" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_gentest.py" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/rocblas_clients_readme.txt" "${PROJECT_BINARY_DIR}/clients/staging/rocblas_clients_readme.txt" COPYONLY)

if(NOT ROCBLAS_ENABLE_FORTRAN)
  target_compile_definitions(rocblas-bench PRIVATE CLIENTS_NO_FORTRAN)
  target_compile_definitions(rocblas-test PRIVATE CLIENTS_NO_FORTRAN)
  target_compile_definitions(rocblas-testing-common PRIVATE CLIENTS_NO_FORTRAN)
  target_compile_definitions(rocblas-clients-common PRIVATE CLIENTS_NO_FORTRAN)
  # need to add others to the list that may require this
endif()

# Install rules for test support files
if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
  rocm_install(
    FILES
      "${PROJECT_BINARY_DIR}/clients/staging/rocblas_smoke.yaml"
      "${PROJECT_BINARY_DIR}/clients/staging/rocblas_extras.yaml"
      "${PROJECT_BINARY_DIR}/clients/staging/rocblas_rtest.xml"
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT tests
  )
  rocm_install(
    PROGRAMS "${PROJECT_BINARY_DIR}/clients/staging/rocblas_rtest.py"
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT tests
  )
endif()
