# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocblas-clients-testing-common
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/common_gemm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_symm_hemm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_trsm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_trtri.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_trmm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_syrk.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_syr2k.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_herk.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_her2k.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_dgmm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_geam.cpp"
)
