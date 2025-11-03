# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(rocblas64 OBJECT)
add_library(rocblas::64 ALIAS rocblas64)

# Explicit includes for rocblas64
target_include_directories(rocblas64
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/internal"
        "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas"
        "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

# Explicit definitions for rocblas64
target_compile_definitions(rocblas64
    PRIVATE
        ROCM_USE_FLOAT16
        ROCBLAS_INTERNAL_API
        ROCBLAS_BETA_FEATURES_API
        ROCBLAS_INTERNAL_ILP64
        rocblas_EXPORTS
)

# Conditional Tensile/hipBLASLt definitions
if(ROCBLAS_ENABLE_TENSILE)
    target_compile_definitions(rocblas64 PRIVATE
        BUILD_WITH_TENSILE
        ROCBLAS_TENSILE_SEPARATE_ARCH=1
        ROCBLAS_TENSILE_LAZY_LOAD=1
    )
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
    target_compile_definitions(rocblas64 PRIVATE BUILD_WITH_HIPBLASLT)
endif()

set_target_properties(rocblas64 PROPERTIES COMPILE_FLAGS "--offload-compress" )

set_target_properties(rocblas64 PROPERTIES 
    # POSITION_INDEPENDENT_CODE required for OBJECT library linking into shared library
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden # do we need these?
    VISIBILITY_INLINES_HIDDEN ON # do we need these?
)

# NOTE: C++ standard inherited from global CMAKE_CXX_STANDARD=17 setting
# NOTE: CXX_EXTENSIONS inherited from global CMAKE_CXX_EXTENSIONS=OFF setting

add_library(rocblas-helper OBJECT)
add_library(rocblas::helper ALIAS rocblas-helper)

# Explicit includes for rocblas-helper (no longer rely on interface propagation)
target_include_directories(rocblas-helper
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/internal"
        "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas"
        "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

target_compile_definitions(rocblas-helper
    PRIVATE
        ROCM_USE_FLOAT16
        ROCBLAS_BETA_FEATURES_API
        ROCBLAS_INTERNAL_API
        rocblas_EXPORTS
)

if(ROCBLAS_ENABLE_TENSILE)
    target_compile_definitions(rocblas-helper PRIVATE
        BUILD_WITH_TENSILE
        ROCBLAS_TENSILE_SEPARATE_ARCH=1
        ROCBLAS_TENSILE_LAZY_LOAD=1
    )
endif()

set_target_properties(rocblas-helper PROPERTIES COMPILE_FLAGS "--offload-compress" )

if(ROCBLAS_ENABLE_HIPBLASLT)
    target_compile_definitions(rocblas-helper PRIVATE BUILD_WITH_HIPBLASLT)
endif()

target_link_libraries(rocblas-helper
    PRIVATE
        rocblas::warnings
        hip::host
        hip::device
)

# NOTE: C++ standard inherited from global CMAKE_CXX_STANDARD=17 setting
# NOTE: CXX_EXTENSIONS inherited from global CMAKE_CXX_EXTENSIONS=OFF setting

set_target_properties(rocblas-helper PROPERTIES 
    # POSITION_INDEPENDENT_CODE required for OBJECT library linking into shared library
    POSITION_INDEPENDENT_CODE ON
    # CXX_VISIBILITY_PRESET hidden
    # VISIBILITY_INLINES_HIDDEN ON
)

# Links for rocblas64 (all PRIVATE - OBJECT libraries don't need PUBLIC links)
target_link_libraries(rocblas64
    PRIVATE
        rocblas::warnings
        hip::device
        hip::host
)

if(ROCBLAS_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    add_library(rocblas SHARED)
else()
    add_library(rocblas STATIC)
endif()

add_library(roc::rocblas ALIAS rocblas)
rocm_set_soversion(rocblas ${rocblas_SOVERSION})

if(ROCBLAS_ENABLE_ASAN)
    rocblas_target_configure_sanitizers(rocblas PRIVATE)
    rocblas_target_configure_sanitizers(rocblas64 PRIVATE)
endif()

target_include_directories(rocblas
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/internal>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

set_target_properties(rocblas
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/staging
)
if(ROCBLAS_ENABLE_TENSILE)
    target_compile_definitions(rocblas PRIVATE
        BUILD_WITH_TENSILE
        ROCBLAS_TENSILE_SEPARATE_ARCH=1
        ROCBLAS_TENSILE_LAZY_LOAD=1
    )
endif()

if(ROCBLAS_ENABLE_MARKER)
    target_link_libraries(rocblas PRIVATE ${ROCTX64_LIBRARY})
endif()

target_link_libraries(rocblas
    PRIVATE
        rocblas::helper
        rocblas::64
        rocblas::warnings
        hip::device
        hip::host
)

target_compile_definitions(rocblas
    PRIVATE
        ROCM_USE_FLOAT16
        ROCBLAS_INTERNAL_API
        ROCBLAS_BETA_FEATURES_API
)

set_target_properties(rocblas PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Enable variable-length array (VLA) errors to catch potential stack issues
# need to make sure we have this flag everywhere it was previously
target_compile_options(rocblas PRIVATE -Werror=vla --offload-compress)

# Prevent symbols from static libraries (like Tensile) from being re-exported
# This linker flag has no CMake property equivalent and must be specified directly
target_link_options(rocblas PRIVATE -Xlinker --exclude-libs=ALL)

if(ROCBLAS_ENABLE_FORTRAN)
    add_library(rocblas-fortran-module STATIC)
    add_library(rocblas::rocblas-fortran-module ALIAS rocblas-fortran-module)
    target_sources(
        rocblas-fortran-module PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_module.f90"
    )
    set_target_properties(
        rocblas-fortran-module
        PROPERTIES Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules"
                   POSITION_INDEPENDENT_CODE ON
                   # CXX_VISIBILITY_PRESET hidden
                   # VISIBILITY_INLINES_HIDDEN ON
    )

    target_include_directories(
        rocblas-fortran-module PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/modules>
                                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas>
    )
endif()

# Organize library components
add_subdirectory(include)
add_subdirectory(src)

# TODO: Fix the source code that uses inconsistent include paths and then remove the
# `configure_file` calls below. This is improper practice and allows clients to use wrong (or at
# least inconsistent) include paths. Refer to file: testing_gemm_batched_ex.hpp, which uses the
# following include: #include <rocblas/internal/rocblas-beta.h>
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas-beta.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal" COPYONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas.h" "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas"
    COPYONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/include/rocblas_device_malloc.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas" COPYONLY
)
