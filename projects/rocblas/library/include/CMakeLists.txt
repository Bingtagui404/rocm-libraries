# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-version.h" @ONLY)

generate_export_header(
  rocblas EXPORT_FILE_NAME
  "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-export.h" BASE_NAME
  rocblas)

target_sources(
  rocblas
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas_bfloat16.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-auxiliary.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-functions.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-macros.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-beta.h"
    "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-version.h"
    "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-export.h")
