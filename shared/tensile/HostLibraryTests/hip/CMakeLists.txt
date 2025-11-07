# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(test_yaml rocblas_sgemm_asm_single_kernel.yaml)
file(COPY ${test_yaml} DESTINATION .)

add_custom_target(
    test_code_object
    COMMAND python "${PROJECT_SOURCE_DIR}/Tensile/Tensile.py" ${test_yaml} code_object
    DEPENDS ${test_yaml}
)

target_sources(TensileTests
    PRIVATE
        hip/HipSolutionAdapter_test.cpp
        hip/CopyTensor_test.cpp
        hip/RunGEMMKernel_test.cpp
)

target_link_libraries(TensileTests
    PRIVATE
        hip::host
        test_kernels_lite
        test_kernels_lite_2
        test_kernels_lite_mixed
        test_tile_aware_selection
        TensileClient
)

set(TENSILE_CONFIG_DIR "${PROJECT_SOURCE_DIR}/HostLibraryTests/configs")
set(TENSILE_TEST_DATA_DIR "${PROJECT_SOURCE_DIR}/HostLibraryTests/data")

set(tcl_command
    ${TENSILE_PYTHON_COMMAND} "${PROJECT_SOURCE_DIR}/Tensile/bin/TensileCreateLibrary"
    --no-merge-files --no-short-file-names --verbose=0 --code-object-version=default
    --cxx-compiler=amdclang++ --cmake-cxx-compiler="${CMAKE_CXX_COMPILER}" --jobs=8
)

set(lite_command ${tcl_command} --embed-library=test_kernels_lite --embed-library-key=kernels_lite
                 "${TENSILE_CONFIG_DIR}/lite_configs" "${TENSILE_TEST_DATA_DIR}/test_kernels_lite" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_kernels_lite/library/test_kernels_lite.cpp"
  COMMAND ${lite_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_kernels_lite/library/test_kernels_lite.cpp"
  WORKING_DIRECTORY "${TENSILE_SOURCE_TREE}/../..")

set(lite2_command ${tcl_command} --embed-library=test_kernels_lite_2 --embed-library-key=kernels_lite_2
                "${TENSILE_CONFIG_DIR}/lite_configs" "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2/library/test_kernels_lite_2.cpp"
  COMMAND ${lite2_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2/library/test_kernels_lite_2.cpp"
  WORKING_DIRECTORY "${TENSILE_SOURCE_TREE}/../..")

set(lite_mixed_command ${tcl_command} --embed-library=test_kernels_lite_mixed --embed-library-key=kernels_lite_mixed
                "${TENSILE_CONFIG_DIR}/lite_configs_mixed" "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed/library/test_kernels_lite_mixed.cpp"
  COMMAND ${lite_mixed_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed/library/test_kernels_lite_mixed.cpp"
  WORKING_DIRECTORY "${TENSILE_SOURCE_TREE}/../..")

set(tile_aware_command ${tcl_command} --embed-library=test_tile_aware_selection --embed-library-key=tile_aware_selection
                       "${TENSILE_CONFIG_DIR}/tile_aware_selection" "${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection" HIP)

add_custom_command(
  COMMENT "Generating ${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection/library/test_tile_aware_selection.cpp"
  COMMAND ${tile_aware_command}
  OUTPUT "${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection/library/test_tile_aware_selection.cpp"
  WORKING_DIRECTORY "${TENSILE_SOURCE_TREE}/../..")

add_library(test-device-kernels OBJECT
  "${TENSILE_TEST_DATA_DIR}/test_kernels_lite/library/test_kernels_lite.cpp"
  "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_2/library/test_kernels_lite_2.cpp"
  "${TENSILE_TEST_DATA_DIR}/test_kernels_lite_mixed/library/test_kernels_lite_mixed.cpp"
  "${TENSILE_TEST_DATA_DIR}/test_tile_aware_selection/library/test_tile_aware_selection.cpp")
target_link_libraries(test-device-kernels PRIVATE tensile-host)
target_compile_definitions(test-device-kernels PRIVATE __HIP_HCC_COMPAT_MODE__=1)
target_compile_features(test-device-kernels PUBLIC cxx_std_17)