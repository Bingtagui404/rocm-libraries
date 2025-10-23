# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# BLAS Extended (EX) implementation sources
# These sources are added to the rocblas target (not rocblas64)

target_sources(rocblas
    PRIVATE
        # AXPY_EX
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_axpy_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_axpy_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_axpy_strided_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_axpy_ex_kernels.cpp"
        
        # DOT_EX
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dot_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dot_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dot_strided_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dot_ex_kernels.cpp"
        
        # NRM2_EX
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_nrm2_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_nrm2_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_nrm2_strided_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_nrm2_ex_kernels.cpp"
        
        # ROT_EX
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rot_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rot_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rot_strided_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rot_ex_kernels.cpp"
        
        # SCAL_EX
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_scal_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_scal_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_scal_strided_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_scal_ex_kernels.cpp"
        
        # GEAM_EX
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_geam_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_geam_ex_kernels.cpp"
        
        # GEMM_EX
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemm_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemm_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemm_strided_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemm_ex_kernels.cpp"
        
        # GEMMT
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemmt.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemmt_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemmt_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemmt_kernels.cpp"
        
        # TRSV_EX
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsv_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsv_batched_ex.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsv_strided_batched_ex.cpp"
)
