# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Configure generated headers
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/internal/rocblas-version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-version.h"
)

generate_export_header(rocblas
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-export.h"
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

set(ROCBLAS_PROTO_TEMPLATES "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas-exported-proto.hpp")
set(ROCBLAS_LIBRARY_SOURCE_DIR "${PROJECT_SOURCE_DIR}/library/src")
add_custom_command(OUTPUT ${ROCBLAS_PROTO_TEMPLATES}
  COMMAND ${Python3_EXECUTABLE} "${ROCBLAS_LIBRARY_SOURCE_DIR}/template-proto.py" ${CMAKE_CURRENT_SOURCE_DIR} "${ROCBLAS_LIBRARY_SOURCE_DIR}/blas3/Tensile" "${ROCBLAS_LIBRARY_SOURCE_DIR}/blas3" "${ROCBLAS_LIBRARY_SOURCE_DIR}/blas2" "${ROCBLAS_LIBRARY_SOURCE_DIR}/blas1" "${ROCBLAS_LIBRARY_SOURCE_DIR}/src64/blas3" "${ROCBLAS_LIBRARY_SOURCE_DIR}/src64/blas2" "${ROCBLAS_LIBRARY_SOURCE_DIR}/src64/blas1" > ${ROCBLAS_PROTO_TEMPLATES}
  COMMENT "Generating prototypes from ${ROCBLAS_LIBRARY_SOURCE_DIR}."
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(rocblas_proto_templates ALL DEPENDS ${ROCBLAS_PROTO_TEMPLATES})
add_dependencies(rocblas rocblas_proto_templates)

# generate rocblas_device_malloc.hpp header for device memory allocation
configure_file("${ROCBLAS_LIBRARY_SOURCE_DIR}/include/rocblas_device_malloc.hpp" "${CMAKE_CURRENT_BINARY_DIR}/rocblas/internal/rocblas_device_malloc.hpp")

