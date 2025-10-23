# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# BLAS Level 1 implementation sources
# These sources are added to the rocblas target (not rocblas64)

target_sources(rocblas
    PRIVATE
        # ASUM
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_asum.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_asum_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_asum_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_asum_nrm2_kernels.cpp"
        
        # AXPY
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_axpy.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_axpy_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_axpy_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_axpy_kernels.cpp"
        
        # COPY
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_copy.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_copy_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_copy_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_copy_kernels.cpp"
        
        # DOT
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dot.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dot_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dot_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_dot_kernels.cpp"
        
        # IAMAX
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_iamax.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_iamax_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_iamax_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_iamax_iamin_kernels.cpp"
        
        # IAMIN
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_iamin.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_iamin_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_iamin_strided_batched.cpp"
        
        # NRM2
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_nrm2.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_nrm2_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_nrm2_strided_batched.cpp"
        
        # ROT
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rot.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rot_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rot_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rot_kernels.cpp"
        
        # ROTG
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotg.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotg_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotg_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotg_kernels.cpp"
        
        # ROTM
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotm_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotm_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotm_kernels.cpp"
        
        # ROTMG
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotmg.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotmg_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotmg_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_rotmg_kernels.cpp"
        
        # SCAL
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_scal.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_scal_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_scal_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_scal_kernels.cpp"
        
        # SWAP
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_swap.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_swap_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_swap_strided_batched.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_swap_kernels.cpp"
)
