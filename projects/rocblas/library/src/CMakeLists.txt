# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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

if(ROCBLAS_ENABLE_TENSILE OR ROCBLAS_ENABLE_HIPBLASLT)
    # Create internal object library for tensile/hipblaslt_host.cpp
    add_library(rocblas-gemm-host OBJECT)
    add_library(rocblas::tensile-host ALIAS rocblas-gemm-host)
    
    # Explicit includes for rocblas-gemm-host
    target_include_directories(rocblas-gemm-host
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/library/src"
            "${PROJECT_SOURCE_DIR}/library/include"
            "${PROJECT_SOURCE_DIR}/library/include/internal"
            "${PROJECT_BINARY_DIR}/library/include"
            "${PROJECT_BINARY_DIR}/library/include/rocblas"
            "${PROJECT_BINARY_DIR}/library/include/rocblas/internal"
    )

    target_compile_definitions(rocblas-gemm-host
        PRIVATE
            ROCM_USE_FLOAT16
            ROCBLAS_INTERNAL_API
            ROCBLAS_BETA_FEATURES_API
    )

    # Conditional feature-specific definitions
    if(ROCBLAS_ENABLE_TENSILE)
        target_compile_definitions(rocblas-gemm-host PRIVATE
            BUILD_WITH_TENSILE
            ROCBLAS_TENSILE_SEPARATE_ARCH=1
            ROCBLAS_TENSILE_LAZY_LOAD=1
        )
        target_link_libraries(rocblas-gemm-host PUBLIC roc::tensile-host)
    elseif(ROCBLAS_ENABLE_HIPBLASLT)
        target_compile_definitions(rocblas-gemm-host PRIVATE BUILD_WITH_HIPBLASLT)
        target_link_libraries(rocblas-gemm-host PRIVATE roc::hipblaslt)
    endif()
    
    target_link_libraries(rocblas-gemm-host
        PRIVATE
            rocblas::warnings
            hip::device
            hip::host
    )

    # NOTE: C++ standard inherited from global CMAKE_CXX_STANDARD=17 setting
    # NOTE: CXX_EXTENSIONS inherited from global CMAKE_CXX_EXTENSIONS=OFF setting
    
    set_target_properties(rocblas-gemm-host PROPERTIES
        # POSITION_INDEPENDENT_CODE required for OBJECT library linking into shared library
        POSITION_INDEPENDENT_CODE ON
        # CXX_VISIBILITY_PRESET hidden
        # VISIBILITY_INLINES_HIDDEN ON
    )
    
    # Link the object library into rocblas
    target_link_libraries(rocblas PRIVATE rocblas-gemm-host)

   if(ROCBLAS_ENABLE_HIPBLASLT)
        target_sources(rocblas
            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_host.cpp"
        )
    elseif(ROCBLAS_ENABLE_TENSILE)
        target_sources(rocblas-gemm-host
            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tensile_host.cpp"
        )
    endif()
endif()

# BLAS implementation sources organized by level
add_subdirectory(blas1)
add_subdirectory(blas2)
add_subdirectory(blas3)
add_subdirectory(blas_ex)

# ILP64 variant sources
add_subdirectory(src64)
