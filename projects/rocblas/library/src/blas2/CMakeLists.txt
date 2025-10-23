# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# BLAS Level 2 implementation sources
# These sources are added to the rocblas target (not rocblas64)

target_sources(rocblas
    PRIVATE
        # GBMV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gbmv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gbmv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gbmv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gbmv_kernels.cpp"
        
        # GEMV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemv_kernels.cpp"
        
        # GER
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_ger.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_ger_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_ger_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_ger_kernels.cpp"
        
        # HBMV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hbmv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hbmv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hbmv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hbmv_kernels.cpp"
        
        # HEMV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hemv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hemv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hemv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hemv_symv_kernels.cpp"
        
        # HER
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_her.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_her_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_her_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_her_kernels.cpp"
        
        # HER2
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_her2.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_her2_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_her2_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_her2_kernels.cpp"
        
        # HPMV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpmv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpmv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpmv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpmv_kernels.cpp"
        
        # HPR
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpr.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpr_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpr_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpr_kernels.cpp"
        
        # HPR2
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpr2.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpr2_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpr2_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hpr2_kernels.cpp"
        
        # SBMV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_sbmv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_sbmv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_sbmv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_sbmv_kernels.cpp"
        
        # SPMV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spmv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spmv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spmv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spmv_kernels.cpp"
        
        # SPR
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spr.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spr_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spr_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spr_kernels.cpp"
        
        # SPR2
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spr2.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spr2_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spr2_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_spr2_kernels.cpp"
        
        # SYMV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_symv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_symv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_symv_strided_batched.cpp"
        
        # SYR
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr_kernels.cpp"
        
        # SYR2
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr2.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr2_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr2_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr2_kernels.cpp"
        
        # TBMV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tbmv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tbmv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tbmv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tbmv_kernels.cpp"
        
        # TBSV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tbsv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tbsv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tbsv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tbsv_kernels.cpp"
        
        # TPMV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tpmv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tpmv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tpmv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tpmv_kernels.cpp"
        
        # TPSV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tpsv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tpsv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tpsv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_tpsv_kernels.cpp"
        
        # TRMV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trmv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trmv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trmv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trmv_kernels.cpp"
        
        # TRSV
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsv.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsv_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsv_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsv_kernels.cpp"
)
