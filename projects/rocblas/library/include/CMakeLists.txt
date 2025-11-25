# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Configure generated headers
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-version.h.in"
    "${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-version.h"
)

generate_export_header(rocblas
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-export.h"
    BASE_NAME rocblas
)

# All rocBLAS public and internal headers (that actually exist)
target_sources(rocblas
    PUBLIC
        # Public API headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rocblas.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-auxiliary.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-functions.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-types.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas_bfloat16.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas_xfloat32.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-export.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-version.h>
        
        # Install interface
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas/rocblas.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal/rocblas-auxiliary.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal/rocblas-functions.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal/rocblas-types.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal/rocblas_bfloat16.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal/rocblas_xfloat32.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal/rocblas-export.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal/rocblas-version.h>
    PRIVATE
        # Internal headers
        "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-beta.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas_block_sizes.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-complex-types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-macros.h"
)
