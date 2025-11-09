# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")

add_executable(rocblas-test)

target_link_libraries(rocblas-test
    PRIVATE
        $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
        $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main>
        hip::device
        roc::rocblas
        rocblas::testing-common
        rocblas::warnings
)

target_include_directories(rocblas-test
    PRIVATE
        "${PROJECT_SOURCE_DIR}/clients/common"
        "${PROJECT_SOURCE_DIR}/clients/include"
        "${PROJECT_SOURCE_DIR}/clients/include/blas1"
        "${PROJECT_SOURCE_DIR}/clients/include/blas_ex"
        "${PROJECT_SOURCE_DIR}/library/src"
        "${PROJECT_SOURCE_DIR}/library/src/include"
)

target_compile_definitions(rocblas-test
    PRIVATE
        GOOGLE_TEST
        ROCBLAS_INTERNAL_API
        ROCBLAS_NO_DEPRECATED_WARNINGS
        ROCM_USE_FLOAT16
        __HIP_HCC_COMPAT_MODE__=1
)

if(ROCBLAS_ENABLE_TENSILE)
    target_sources(rocblas-test
        PRIVATE
            # Keep multiheaded_gtest.cpp first to initialize TensileHost
            "${CMAKE_CURRENT_SOURCE_DIR}/multiheaded_gtest.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/atomics_mode_gtest.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/get_solutions_gtest.cpp"
    )

    target_compile_definitions(rocblas-test
        PRIVATE
            BUILD_WITH_TENSILE
            ROCBLAS_TENSILE_LAZY_LOAD=1
            ROCBLAS_TENSILE_SEPARATE_ARCH=1
    )
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
    target_compile_definitions(rocblas-test
        PRIVATE
            BUILD_WITH_HIPBLASLT
    )
endif()

target_sources(rocblas-test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest_main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/general_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/set_get_pointer_mode_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/set_get_atomics_mode_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/logging_mode_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ostream_threadsafety_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/set_get_vector_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/set_get_matrix_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/asum_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/axpy_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/copy_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/dot_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/iamaxmin_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/nrm2_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/rot_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/scal_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas1/swap_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/axpy_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/dot_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/nrm2_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/rot_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/scal_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/trsv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/gbmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/gemv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/hbmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/hemv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/her_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/her2_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/hpmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/hpr_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/hpr2_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/trmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/tpmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/tbmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/tbsv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/tpsv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/ger_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/geru_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/gerc_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/spr_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/spr2_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/syr_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/syr2_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/sbmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/spmv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas2/symv_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/gemm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/symm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/hemm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/trsm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/trtri_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/trmm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/syrk_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/syrkx_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/syr2k_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/herk_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/herkx_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/her2k_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/dgmm_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas3/geam_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/gemm_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/gemmt_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/syrk_ex_gtest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/blas_ex/geam_ex_gtest.cpp"
)

if(WIN32)
    # FLAG: Suppress experimental filesystem deprecation warning on Windows
    target_compile_definitions(rocblas-test
        PRIVATE
            _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
    )

    # Copy required DLLs to staging directory for Windows execution
    file(GLOB third_party_dlls
        LIST_DIRECTORIES OFF
        CONFIGURE_DEPENDS
        "${OPENBLAS_DIR}/bin/*.dll"
        "${HIP_DIR}/bin/amd*.dll"
        "${HIP_DIR}/bin/hiprt*.dll"
        "${HIP_DIR}/bin/hipinfo.exe"
        "${CMAKE_SOURCE_DIR}/rtest.*"
        "C:/Windows/System32/libomp140*.dll"
    )
    foreach(file_i ${third_party_dlls})
        add_custom_command(TARGET rocblas-test POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy "${file_i}" "${PROJECT_BINARY_DIR}/clients/staging/")
    endforeach()
endif()

set_target_properties(rocblas-test
    PROPERTIES
        IMPORT_PREFIX ""
        IMPORT_SUFFIX ".lib"
        LINKER_LANGUAGE CXX
)

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
set(ROCBLAS_TEST_DATA "${PROJECT_BINARY_DIR}/clients/staging/rocblas_gtest.data")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_command(
    OUTPUT "${ROCBLAS_TEST_DATA}"
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/clients/common/rocblas_gentest.py"
            -I "${PROJECT_SOURCE_DIR}/clients/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest.yaml"
            -o "${ROCBLAS_TEST_DATA}"
    DEPENDS 
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest.yaml"
        "${PROJECT_SOURCE_DIR}/clients/common/rocblas_gentest.py"
    COMMENT "Generating test data: ${ROCBLAS_TEST_DATA}"
    VERBATIM
)

add_custom_target(rocblas-test-data ALL
    DEPENDS "${ROCBLAS_TEST_DATA}"
)

add_dependencies(rocblas-test rocblas-test-data)
