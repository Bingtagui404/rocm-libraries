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
        find_package(BLIS REQUIRED) # are we doing this in multiple places?
    else()
        find_package(BLAS REQUIRED) # are we doing this in multiple places?
    endif()

    if(ROCBLAS_ENABLE_ROCM_SMI)
        find_package(ROCmSMI REQUIRED)
    endif()

    find_package(GTest REQUIRED)

    add_library(rocblas-testing-common STATIC)
    add_library(rocblas::testing-common ALIAS rocblas-testing-common)

    target_link_libraries(rocblas-testing-common
        PRIVATE
            rocblas::warnings
            roc::rocblas
            $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
            $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main>
            rocblas::clients-common
            hip::device
    )
    if(ROCBLAS_ENABLE_BLIS)
        target_link_libraries(rocblas-testing-common PRIVATE BLIS::BLIS)
        target_sources(rocblas-testing-common
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/clients/common/blis_interface.cpp"
        )
    else()
        target_link_libraries(rocblas-testing-common PRIVATE ${BLAS_LIBRARIES})
        if(BLAS_INCLUDE_DIRS)
            target_include_directories(rocblas-testing-common SYSTEM PRIVATE ${BLAS_INCLUDE_DIRS}) # should we be doing this?
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
            __HIP_HCC_COMPAT_MODE__=1
            #ROCBLAS_INTERNAL_API
            ROCM_USE_FLOAT16
            ROCBLAS_NO_DEPRECATED_WARNINGS
            GOOGLE_TEST
    )

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

if(ROCBLAS_ENABLE_OPENMP)
    add_subdirectory(openmp-support)
    if(ROCBLAS_BUILD_TESTING)
        target_link_libraries(rocblas-test PRIVATE rocblas::clients-openmp)
    endif()
    if(ROCBLAS_ENABLE_BENCHMARKS)
        target_link_libraries(rocblas-bench PRIVATE rocblas::clients-openmp)
    endif()
endif()
