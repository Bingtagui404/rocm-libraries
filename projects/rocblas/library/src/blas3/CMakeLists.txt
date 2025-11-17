# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# BLAS Level 3 implementation sources
# These sources are added to the rocblas target (not rocblas64)

target_sources(rocblas
    PRIVATE
        # DGMM
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dgmm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dgmm_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dgmm_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dgmm_kernels.cpp"
        
        # GEAM
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_geam.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_geam_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_geam_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_geam_kernels.cpp"
        
        # GEMM
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemm_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_gemm_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/gemm_templates.cpp"
        
        # HEMM
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hemm_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_hemm_strided_batched.cpp"
        
        # HER2K
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_her2k.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_her2k_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_her2k_strided_batched.cpp"
        
        # HERK
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_herk.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_herk_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_herk_strided_batched.cpp"
        
        # HERKX
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_herkx.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_herkx_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_herkx_strided_batched.cpp"
        
        # SYMM
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_symm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_symm_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_symm_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_symm_hemm_kernels.cpp"
        
        # SYR2K
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr2k.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr2k_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr2k_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syr2k_her2k_kernels.cpp"
        
        # SYRK
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syrk.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syrk_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syrk_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syrk_herk_kernels.cpp"
        
        # SYRKX
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syrkx.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syrkx_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syrkx_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_syrkx_herkx_kernels.cpp"
        
        # TRMM
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trmm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trmm_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trmm_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trmm_kernels.cpp"
        
        # TRSM
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsm_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsm_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsm_kernels.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trsm_batched_kernels.cpp"
        
        # TRTRI
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trtri.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trtri_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trtri_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_trtri_kernels.cpp"
)
