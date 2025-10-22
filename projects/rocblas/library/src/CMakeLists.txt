# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Organize rocBLAS library sources
# Note: Full BLAS level subdirectory organization (blas1/, blas2/, blas3/, blas_ex/)
# will be added in future iterations. For now, we add core sources to get builds working.

# Core rocBLAS helper library sources
target_sources(rocblas-helper
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/buildinfo.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/check_numerics_matrix.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/check_numerics_vector.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/handle.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/logging.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_auxiliary.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_ostream.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
)

# Main rocBLAS library backend integration sources
# TODO: BLAS level sources (blas1/, blas2/, blas3/, blas_ex/) will be organized
# into subdirectories following the same pattern

# For now, we need at least one source file for the rocblas target
# The actual BLAS functions are in subdirectories which we'll organize later
if(ROCBLAS_ENABLE_TENSILE)
    target_sources(rocblas
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/tensile_host.cpp"
    )
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
    target_sources(rocblas
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_host.cpp"
    )
endif()

# ILP64 variant sources
add_subdirectory(src64)
