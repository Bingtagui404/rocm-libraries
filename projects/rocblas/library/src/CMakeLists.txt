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
    # Create internal object library for tensile/hipblaslt_host.cpp with legacy compatibility flag
    # 
    # RATIONALE: __HIP_HCC_COMPAT_MODE__=1 was transitively provided in the legacy build system
    # (not explicitly defined in any rocBLAS CMakeLists.txt files). Investigation of the develop
    # branch confirmed this flag was pulled in through dependencies or legacy toolchain files.
    # 
    # DECISION: We scope this flag to tensile host code only via an internal object library to:
    #   1. Avoid unnecessary global propagation of a legacy compatibility flag
    #   2. Maintain compatibility with Tensile integration that may depend on this flag
    #   3. Provide clear separation of concerns and easier future removal if needed
    #   4. Use proper CMake target-based properties instead of source file properties
    #
    add_library(rocblas-gemm-host OBJECT)
    add_library(rocblas::tensile-host ALIAS rocblas-gemm-host)
        
    target_link_libraries(rocblas-gemm-host
        PRIVATE rocblas-interface hip::device hip::host
    )

    # Static library must be position-independent for linking into shared library
    set_target_properties(rocblas-gemm-host PROPERTIES
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
