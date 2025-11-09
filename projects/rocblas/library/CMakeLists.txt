# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(rocblas64 OBJECT)
add_library(rocblas::64 ALIAS rocblas64)

target_link_libraries(rocblas64
    PRIVATE
        hip::device
        hip::host
        rocblas::warnings
)

target_include_directories(rocblas64
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas"
        "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/internal"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

target_compile_definitions(rocblas64
    PRIVATE
        ROCBLAS_BETA_FEATURES_API
        ROCBLAS_INTERNAL_API
        ROCBLAS_INTERNAL_ILP64
        ROCM_USE_FLOAT16
        rocblas_EXPORTS
)

if(ROCBLAS_ENABLE_TENSILE)
    target_compile_definitions(rocblas64
        PRIVATE
            BUILD_WITH_TENSILE
            ROCBLAS_TENSILE_LAZY_LOAD=1
            ROCBLAS_TENSILE_SEPARATE_ARCH=1
    )
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
    target_compile_definitions(rocblas64 PRIVATE BUILD_WITH_HIPBLASLT)
endif()

# NOTE: C++ standard inherited from global CMAKE_CXX_STANDARD=17 setting
# NOTE: CXX_EXTENSIONS inherited from global CMAKE_CXX_EXTENSIONS=OFF setting

set_target_properties(rocblas64
    PROPERTIES
        COMPILE_FLAGS "--offload-compress"
        CXX_VISIBILITY_PRESET hidden
        # POSITION_INDEPENDENT_CODE required for OBJECT library linking into shared library
        POSITION_INDEPENDENT_CODE ON
        VISIBILITY_INLINES_HIDDEN ON
)

add_library(rocblas-helper OBJECT)
add_library(rocblas::helper ALIAS rocblas-helper)

target_link_libraries(rocblas-helper
    PRIVATE
        hip::device
        hip::host
        rocblas::warnings
)

target_include_directories(rocblas-helper
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas"
        "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/internal"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

target_compile_definitions(rocblas-helper
    PRIVATE
        ROCBLAS_BETA_FEATURES_API
        ROCBLAS_INTERNAL_API
        ROCM_USE_FLOAT16
        rocblas_EXPORTS
)

if(ROCBLAS_ENABLE_TENSILE)
    target_compile_definitions(rocblas-helper
        PRIVATE
            BUILD_WITH_TENSILE
            ROCBLAS_TENSILE_LAZY_LOAD=1
            ROCBLAS_TENSILE_SEPARATE_ARCH=1
    )
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
    target_compile_definitions(rocblas-helper PRIVATE BUILD_WITH_HIPBLASLT)
endif()

# NOTE: C++ standard inherited from global CMAKE_CXX_STANDARD=17 setting
# NOTE: CXX_EXTENSIONS inherited from global CMAKE_CXX_EXTENSIONS=OFF setting

# POSITION_INDEPENDENT_CODE required for OBJECT library linking into shared library
set_target_properties(rocblas-helper
    PROPERTIES
        COMPILE_FLAGS "--offload-compress"
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VISIBILITY_INLINES_HIDDEN ON
)

if(ROCBLAS_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    add_library(rocblas SHARED)
else()
    add_library(rocblas STATIC)
endif()

add_library(roc::rocblas ALIAS rocblas)

if(ROCBLAS_ENABLE_ASAN)
    rocblas_target_configure_sanitizers(rocblas PRIVATE)
    rocblas_target_configure_sanitizers(rocblas64 PRIVATE)
endif()

target_link_libraries(rocblas
    PRIVATE
        hip::device
        hip::host
        rocblas::64
        rocblas::helper
        rocblas::warnings
)

target_include_directories(rocblas
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/internal>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

target_compile_definitions(rocblas
    PRIVATE
        ROCBLAS_BETA_FEATURES_API
        ROCBLAS_INTERNAL_API
        ROCM_USE_FLOAT16
)

if(ROCBLAS_ENABLE_TENSILE)
    target_compile_definitions(rocblas
        PRIVATE
            BUILD_WITH_TENSILE
            ROCBLAS_TENSILE_LAZY_LOAD=1
            ROCBLAS_TENSILE_SEPARATE_ARCH=1
    )
endif()

# FLAG: -Werror=vla catches variable-length array usage to prevent stack issues
target_compile_options(rocblas
    PRIVATE
        --offload-compress
        -Werror=vla
)

# Prevent symbols from static libraries (like Tensile) from being re-exported
target_link_options(rocblas PRIVATE -Xlinker --exclude-libs=ALL)

set_target_properties(rocblas
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/staging
        VISIBILITY_INLINES_HIDDEN ON
)

if(ROCBLAS_ENABLE_MARKER)
    target_link_libraries(rocblas PRIVATE ${ROCTX64_LIBRARY})
endif()

if(ROCBLAS_ENABLE_FORTRAN)
    add_library(rocblas-fortran-module STATIC)
    add_library(rocblas::rocblas-fortran-module ALIAS rocblas-fortran-module)

    target_sources(rocblas-fortran-module PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_module.f90")

    target_include_directories(rocblas-fortran-module
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/modules>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rocblas>
    )

    set_target_properties(rocblas-fortran-module
        PROPERTIES
            Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules"
            POSITION_INDEPENDENT_CODE ON
    )
endif()

add_subdirectory(include)
add_subdirectory(src)

# TODO: Fix source code to use consistent include paths, then remove these configure_file calls
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas-beta.h" "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas/internal" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas.h" "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/include/rocblas_device_malloc.hpp" "${CMAKE_CURRENT_BINARY_DIR}/include/rocblas" COPYONLY)
