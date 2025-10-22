# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocblas-clients-testing-common
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/common_trsv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_gbmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_gemv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_hbmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_hemv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_her.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_her2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_hpmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_hpr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_hpr2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_trmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_tpmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_tbmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_tbsv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_tpsv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_ger.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_spr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_spr2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_syr.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_syr2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_sbmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_spmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_symv.cpp"
)
