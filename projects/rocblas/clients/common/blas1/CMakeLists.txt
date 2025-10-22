# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocblas-clients-testing-common
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/common_asum.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_axpy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_copy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_dot.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_iamax_iamin.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_nrm2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_rot.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_scal.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_swap.cpp"
)
