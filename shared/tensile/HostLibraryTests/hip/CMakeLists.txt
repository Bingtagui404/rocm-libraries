# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(test_yaml rocblas_sgemm_asm_single_kernel.yaml)
file(COPY ${test_yaml} DESTINATION .)

add_custom_target(
    test_code_object
    COMMAND python "${PROJECT_SOURCE_DIR}/Tensile/Tensile.py" ${test_yaml} code_object
    DEPENDS ${test_yaml}
)

target_sources(tensile-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/HipSolutionAdapter_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CopyTensor_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RunGEMMKernel_test.cpp"
)

set(TENSILE_CONFIG_DIR "${PROJECT_SOURCE_DIR}/configs")
set(TENSILE_TEST_DATA_DIR "${PROJECT_BINARY_DIR}/data")

set(tcl_command
    ${TENSILE_PYTHON_COMMAND} "${PROJECT_SOURCE_DIR}/../Tensile/bin/TensileCreateLibrary"
    --no-merge-files --no-short-file-names --verbose=0 --code-object-version=default
    --cxx-compiler=amdclang++ --cmake-cxx-compiler="${CMAKE_CXX_COMPILER}" --jobs=8
)

set(lite_command ${tcl_command} --embed-library=test_kernels_lite --embed-library-key=kernels_lite
                 "${TENSILE_CONFIG_DIR}/lite_configs" "${TENSILE_TEST_DATA_DIR}/test_kernels_lite" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_kernels_lite/library/test_kernels_lite.cpp"
  COMMAND ${lite_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_kernels_lite/library/test_kernels_lite.cpp"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/..")

set(lite2_command ${tcl_command} --embed-library=test_kernels_lite_2 --embed-library-key=kernels_lite_2
                "${TENSILE_CONFIG_DIR}/lite_configs" "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2/library/test_kernels_lite_2.cpp"
  COMMAND ${lite2_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2/library/test_kernels_lite_2.cpp"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/..")

set(lite_mixed_command ${tcl_command} --embed-library=test_kernels_lite_mixed --embed-library-key=kernels_lite_mixed
                "${TENSILE_CONFIG_DIR}/lite_configs_mixed" "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed/library/test_kernels_lite_mixed.cpp"
  COMMAND ${lite_mixed_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed/library/test_kernels_lite_mixed.cpp"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/..")

set(tile_aware_command ${tcl_command} --embed-library=test_tile_aware_selection --embed-library-key=tile_aware_selection
                       "${TENSILE_CONFIG_DIR}/tile_aware_selection" "${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection/library/test_tile_aware_selection.cpp"
  COMMAND ${tile_aware_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection/library/test_tile_aware_selection.cpp"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/..")

add_library(test_kernels_lite STATIC
  "${TENSILE_TEST_DATA_DIR}/test_kernels_lite/library/test_kernels_lite.cpp")
target_link_libraries(test_kernels_lite PRIVATE roc::tensile-host hip::device)
target_compile_definitions(test_kernels_lite
  PRIVATE
    __HIP_HCC_COMPAT_MODE__=1
    TENSILE_STATIC_ONLY
)

add_library(test_kernels_lite2 STATIC
  "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2/library/test_kernels_lite_2.cpp")
target_link_libraries(test_kernels_lite2 PRIVATE roc::tensile-host hip::device)
target_compile_definitions(test_kernels_lite2
  PRIVATE
    __HIP_HCC_COMPAT_MODE__=1
    TENSILE_STATIC_ONLY
)

add_library(test_kernels_lite_mixed STATIC
  "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed/library/test_kernels_lite_mixed.cpp")
target_link_libraries(test_kernels_lite_mixed PRIVATE roc::tensile-host hip::device)
target_compile_definitions(test_kernels_lite_mixed
  PRIVATE
    __HIP_HCC_COMPAT_MODE__=1
    TENSILE_STATIC_ONLY
)

add_library(test_tile_aware_selection STATIC
  "${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection/library/test_tile_aware_selection.cpp")
target_link_libraries(test_tile_aware_selection PRIVATE roc::tensile-host hip::device)
target_compile_definitions(test_tile_aware_selection
  PRIVATE
    __HIP_HCC_COMPAT_MODE__=1
    TENSILE_STATIC_ONLY
)
