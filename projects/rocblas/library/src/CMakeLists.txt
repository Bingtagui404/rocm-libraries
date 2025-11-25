# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
    add_library(rocblas-gemm-host OBJECT)
    add_library(rocblas::tensile-host ALIAS rocblas-gemm-host)

    target_link_libraries(rocblas-gemm-host
        PRIVATE
            hip::device
            hip::host
            rocblas::warnings
    )

    target_include_directories(rocblas-gemm-host
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${PROJECT_BINARY_DIR}/include"
            "${PROJECT_BINARY_DIR}/include/rocblas"
            "${PROJECT_BINARY_DIR}/include/rocblas/internal"
            "${PROJECT_SOURCE_DIR}/library/include"
            "${PROJECT_SOURCE_DIR}/library/include/internal"
    )

    target_compile_definitions(rocblas-gemm-host
        PRIVATE
            ROCBLAS_BETA_FEATURES_API
            ROCBLAS_INTERNAL_API
            ROCM_USE_FLOAT16
    )

    if(ROCBLAS_ENABLE_TENSILE)
        target_link_libraries(rocblas-gemm-host PRIVATE roc::tensile-host)

        target_compile_definitions(rocblas-gemm-host
            PRIVATE
                BUILD_WITH_TENSILE
                ROCBLAS_TENSILE_LAZY_LOAD=1
                ROCBLAS_TENSILE_SEPARATE_ARCH=1
        )
    endif()

    if(ROCBLAS_ENABLE_HIPBLASLT)
        target_link_libraries(rocblas-gemm-host PRIVATE roc::hipblaslt)
        target_compile_definitions(rocblas-gemm-host PRIVATE BUILD_WITH_HIPBLASLT)
    endif()

    # NOTE: C++ standard inherited from global CMAKE_CXX_STANDARD=17 setting
    # NOTE: CXX_EXTENSIONS inherited from global CMAKE_CXX_EXTENSIONS=OFF setting

    # Compiler flags for rocblas-gemm-host
    target_compile_options(rocblas-gemm-host PRIVATE --offload-compress)
    target_compile_definitions(rocblas-gemm-host PRIVATE rocblas_EXPORTS)

    # POSITION_INDEPENDENT_CODE required for OBJECT library linking into shared library
    set_target_properties(rocblas-gemm-host
        PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            POSITION_INDEPENDENT_CODE ON
            VISIBILITY_INLINES_HIDDEN ON
    )

    target_link_libraries(rocblas PRIVATE rocblas-gemm-host)

    if(ROCBLAS_ENABLE_TENSILE)
        target_sources(rocblas-gemm-host PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tensile_host.cpp")
        if(ROCBLAS_ENABLE_ASAN)
            rocblas_target_configure_sanitizers(rocblas-gemm-host PRIVATE)
        endif()
    endif()

    if(ROCBLAS_ENABLE_HIPBLASLT)
        target_sources(rocblas PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_host.cpp")
        # Link hipBLASLt to main rocblas target (not just rocblas-gemm-host)
        target_link_libraries(rocblas PRIVATE roc::hipblaslt)
    endif()
endif()

add_subdirectory(blas1)
add_subdirectory(blas2)
add_subdirectory(blas3)
add_subdirectory(blas_ex)
add_subdirectory(src64)

# generate rocblas_device_malloc.hpp header for device memory allocation
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_device_malloc.hpp" "${PROJECT_BINARY_DIR}/include/rocblas/rocblas_device_malloc.hpp")

set(ROCBLAS_PROTO_TEMPLATES "${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-exported-proto.hpp")

add_custom_command(OUTPUT ${ROCBLAS_PROTO_TEMPLATES}
  COMMAND ${Python3_EXECUTABLE} 
    "${CMAKE_CURRENT_SOURCE_DIR}/template-proto.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    "${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/"
    "${CMAKE_CURRENT_SOURCE_DIR}/blas3/"
    "${CMAKE_CURRENT_SOURCE_DIR}/blas2/"
    "${CMAKE_CURRENT_SOURCE_DIR}/blas1/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/" > ${ROCBLAS_PROTO_TEMPLATES}
  COMMENT "Generating prototypes from ${CMAKE_CURRENT_SOURCE_DIR}."
)
add_custom_target(rocblas_proto_templates ALL DEPENDS ${ROCBLAS_PROTO_TEMPLATES})
