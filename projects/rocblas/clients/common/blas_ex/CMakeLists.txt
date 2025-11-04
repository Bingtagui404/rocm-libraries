# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocblas-testing-common
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/common_axpy_ex.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_dot_ex.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_nrm2_ex.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_rot_ex.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_scal_ex.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_gemm_ex.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_trsm_ex.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_gemmt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_syrk_ex.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_geam_ex.cpp"
)
