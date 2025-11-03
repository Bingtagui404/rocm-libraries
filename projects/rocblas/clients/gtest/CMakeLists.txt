# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

add_executable(rocblas-test)

target_link_libraries(rocblas-test
    PRIVATE
        hip::device
        roc::rocblas
        rocblas::warnings
        rocblas::testing-common
        $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
        $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main>
)

target_include_directories(rocblas-test
    PRIVATE
        "${PROJECT_SOURCE_DIR}/clients/include"
        "${PROJECT_SOURCE_DIR}/clients/include/blas1"
        "${PROJECT_SOURCE_DIR}/clients/include/blas_ex"
        "${PROJECT_SOURCE_DIR}/clients/common"
        "${PROJECT_SOURCE_DIR}/library/src"
        "${PROJECT_SOURCE_DIR}/library/src/include")

target_compile_definitions(rocblas-test
    PRIVATE
        ROCBLAS_NO_DEPRECATED_WARNINGS
        ROCM_USE_FLOAT16
        GOOGLE_TEST
)

if(ROCBLAS_ENABLE_TENSILE)
  target_sources(rocblas-test
    PRIVATE
      # Keep multiheaded_gtest.cpp first to initialize TensileHost
      "${CMAKE_CURRENT_SOURCE_DIR}/multiheaded_gtest.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/atomics_mode_gtest.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/get_solutions_gtest.cpp"
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
            COMMAND "${CMAKE_COMMAND}" -E copy "${file_i}" "${PROJECT_BINARY_DIR}/staging/")
    endforeach()
endif()

set_target_properties(rocblas-test PROPERTIES
    IMPORT_PREFIX ""
    IMPORT_SUFFIX ".lib"
    LINKER_LANGUAGE CXX
)

target_link_libraries(rocblas-test PRIVATE rocblas::testing-common)

set(yaml_files
    "${CMAKE_CURRENT_SOURCE_DIR}/asum_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/atomics_mode_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/aux_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/axpy_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/blas1_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/blas2_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/blas3_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/copy_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/dgmm_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/dot_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/gbmv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/geam_ex_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/geam_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/gemm_batched_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/gemm_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/gemm_strided_batched_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/gemmt_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/gemv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/general_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/ger_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/geruc_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/get_solutions_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/hbmv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/hemm_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/hemv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/her_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/her2_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/her2k_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/herk_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/herkx_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/hpmv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/hpr_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/hpr2_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/iamax_iamin_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/known_bugs.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/logging_mode_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/multiheaded_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/nrm2_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/ostream_threadsafety_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/rot_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/sbmv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/scal_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/set_get_atomics_mode_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/set_get_matrix_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/set_get_pointer_mode_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/set_get_vector_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/spmv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/spr_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/spr2_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/swap_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/symm_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/symv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/syr_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/syr2_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/syr2k_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/syrk_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/syrkx_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/tbmv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/tbsv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/tpmv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/tpsv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/trmm_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/trmv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/trsm_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/trsv_gtest.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/trtri_gtest.yaml"
)

set(ROCBLAS_TEST_DATA "${PROJECT_BINARY_DIR}/staging/rocblas_gtest.data")
# ../common/rocblas_gentest.py really belongs in this directory.
# were these files copied into the binary tree previously and if so do we need to continue doing that?
add_custom_command(
    OUTPUT "${ROCBLAS_TEST_DATA}"
    COMMAND "${Python3_EXECUTABLE}" "../common/rocblas_gentest.py" -I "../include" "rocblas_gtest.yaml" -o "${ROCBLAS_TEST_DATA}"
    #DEPENDS "../common/rocblas_gentest.py" "../include/rocblas_common.yaml" # ${yaml_files} if files alrady exist then adding them here is not very helpful this is typically for generated files or targets to ensure the command doesn't run until depedencies are satisfied
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_target(rocblas-test-data DEPENDS "${ROCBLAS_TEST_DATA}")
add_dependencies(rocblas-test rocblas-test-data)

# do we actually use this custom propert? review build system and docs to see if it is used.
set_target_properties(rocblas-test-data 
    PROPERTIES
      ROCBLAS_INSTALL_FILES "${ROCBLAS_TEST_DATA}"
)
