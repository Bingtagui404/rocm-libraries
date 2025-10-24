# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ==============================================================================
# Backward Compatibility Shims for rocBLAS Build Options
# ==============================================================================
# This file maps legacy build option names to modern project-specific options.
# It provides a transition period for users to update their build scripts while
# maintaining backward compatibility.
#
# Users can suppress deprecation warnings with: -DCMAKE_WARN_DEPRECATED=OFF
#
# Legacy → Modern Mappings:
# ------------------------
# BUILD_CLIENTS_TESTS       → ROCBLAS_BUILD_TESTING
# BUILD_CLIENTS_BENCHMARKS  → ROCBLAS_ENABLE_BENCHMARKS
# BUILD_CLIENTS_SAMPLES     → ROCBLAS_ENABLE_SAMPLES
# BUILD_CLIENTS             → ROCBLAS_ENABLE_CLIENT
# BUILD_WITH_TENSILE        → ROCBLAS_ENABLE_TENSILE
# BUILD_WITH_HIPBLASLT      → ROCBLAS_ENABLE_HIPBLASLT
# AMDGPU_TARGETS            → GPU_TARGETS
# BUILD_SHARED_LIBS         → ROCBLAS_BUILD_SHARED_LIBS
# BUILD_ADDRESS_SANITIZER   → ROCBLAS_ENABLE_ASAN
# BUILD_CODE_COVERAGE       → ROCBLAS_BUILD_COVERAGE
# BUILD_VERBOSE             → CMAKE_VERBOSE_MAKEFILE
# SKIP_LIBRARY              → ROCBLAS_ENABLE_HOST (inverted: SKIP_LIBRARY=ON → ROCBLAS_ENABLE_HOST=OFF)
#
# Modern Options (no legacy equivalent):
# --------------------------------------
# ROCBLAS_ENABLE_HOST       - Build rocBLAS host library
# ROCBLAS_ENABLE_OPENMP     - Enable OpenMP for parallel reference implementations
# ROCBLAS_ENABLE_THREADS    - Enable threading support in clients
# ROCBLAS_ENABLE_MARKER     - Enable rocTracer marker support
# ROCBLAS_ENABLE_BLIS       - Enable BLIS support
# ROCBLAS_ENABLE_FORTRAN    - Build Fortran clients
# ROCBLAS_ENABLE_ROCM_SMI   - Require rocm_smi
# ROCBLAS_ENABLE_SUPERBUILD - Build as ROCm BLAS subproject
# CMAKE_VERBOSE_MAKEFILE    - Enable verbose Makefile output
# ==============================================================================

# Helper macro for deprecation warnings using native CMake mechanism
macro(_rocblas_deprecation_warning old_var new_var)
    message(DEPRECATION
        "The option '${old_var}' is deprecated and will be removed in a future release.\n"
        "Please use '${new_var}' instead.\n"
        "To suppress these warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
endmacro()

# Helper macro for conflict detection
macro(_rocblas_check_conflict old_var new_var)
    # If new variable is defined, unset any cached legacy variable to avoid conflicts
    # This allows smooth migration from legacy to modern options without requiring build dir removal
    if(DEFINED ${new_var})
        unset(${old_var} CACHE)
    endif()

    # Only check for actual conflicts if both are actively being set
    if(DEFINED ${old_var} AND DEFINED ${new_var})
        if(NOT "${${old_var}}" STREQUAL "${${new_var}}")
            message(FATAL_ERROR
                "Conflicting options detected:\n"
                "  ${old_var}=${${old_var}} (deprecated)\n"
                "  ${new_var}=${${new_var}}\n"
                "Please remove ${old_var} from your build configuration and use only ${new_var}.")
        endif()
    endif()
endmacro()

# Consolidated function for mapping legacy options to modern equivalents
# Usage: shim_mapping(<legacy_var> <modern_var> <description> [<type>])
# Type defaults to BOOL if not specified
function(shim_mapping legacy_var modern_var description)
    set(var_type "${ARGV3}")
    if(NOT var_type)
        set(var_type "BOOL")
    endif()

    if(DEFINED ${legacy_var})
        _rocblas_check_conflict(${legacy_var} ${modern_var})
        if(NOT DEFINED ${modern_var})
            set(${modern_var} ${${legacy_var}} CACHE ${var_type} "${description}" FORCE)
            _rocblas_deprecation_warning(${legacy_var} ${modern_var})
            list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "${legacy_var}=${${legacy_var}}")
            list(APPEND _ROCBLAS_CURRENT_OPTIONS "${modern_var}=${${modern_var}}")
            # Propagate lists to parent scope
            set(_ROCBLAS_LEGACY_OPTIONS_USED "${_ROCBLAS_LEGACY_OPTIONS_USED}" PARENT_SCOPE)
            set(_ROCBLAS_CURRENT_OPTIONS "${_ROCBLAS_CURRENT_OPTIONS}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ==============================================================================
# Apply Legacy Option Mappings
# ==============================================================================

# Initialize tracking lists for migration guidance
set(_ROCBLAS_LEGACY_OPTIONS_USED "")
set(_ROCBLAS_CURRENT_OPTIONS "")

# Apply mappings using consolidated function
shim_mapping(BUILD_CLIENTS_TESTS ROCBLAS_BUILD_TESTING "Build test client; master switch.")
shim_mapping(BUILD_CLIENTS_BENCHMARKS ROCBLAS_ENABLE_BENCHMARKS "Build benchmark client.")
shim_mapping(BUILD_CLIENTS_SAMPLES ROCBLAS_ENABLE_SAMPLES "Build client samples.")
shim_mapping(BUILD_WITH_TENSILE ROCBLAS_ENABLE_TENSILE "Build Tensile GEMM device libraries.")
shim_mapping(AMDGPU_TARGETS GPU_TARGETS "AMD GFX targets to cross-compile" STRING)
shim_mapping(BUILD_ADDRESS_SANITIZER ROCBLAS_ENABLE_ASAN "Build with address sanitizer enabled.")
shim_mapping(BUILD_CODE_COVERAGE ROCBLAS_BUILD_COVERAGE "Build tests with coverage enabled.")

# Special case: BUILD_CLIENTS (only map if ON)
if(DEFINED BUILD_CLIENTS AND BUILD_CLIENTS AND NOT DEFINED ROCBLAS_ENABLE_CLIENT)
    _rocblas_check_conflict(BUILD_CLIENTS ROCBLAS_ENABLE_CLIENT)
    set(ROCBLAS_ENABLE_CLIENT ON CACHE BOOL "Build rocBLAS clients." FORCE)
    _rocblas_deprecation_warning(BUILD_CLIENTS ROCBLAS_ENABLE_CLIENT)
    list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "BUILD_CLIENTS=${BUILD_CLIENTS}")
    list(APPEND _ROCBLAS_CURRENT_OPTIONS "ROCBLAS_ENABLE_CLIENT=${ROCBLAS_ENABLE_CLIENT}")
endif()

# Additional mappings with special handling
shim_mapping(BUILD_VERBOSE CMAKE_VERBOSE_MAKEFILE "Enable verbose output from Makefile builds.")
shim_mapping(BUILD_SHARED_LIBS ROCBLAS_BUILD_SHARED_LIBS "Build the rocBLAS shared or static library.")

# Map SKIP_LIBRARY → ROCBLAS_ENABLE_HOST (inverted logic)
if(DEFINED SKIP_LIBRARY)
    _rocblas_check_conflict(SKIP_LIBRARY ROCBLAS_ENABLE_HOST)
    if(NOT DEFINED ROCBLAS_ENABLE_HOST)
        if(SKIP_LIBRARY)
            set(ROCBLAS_ENABLE_HOST OFF CACHE BOOL "Build rocBLAS host library." FORCE)
        else()
            set(ROCBLAS_ENABLE_HOST ON CACHE BOOL "Build rocBLAS host library." FORCE)
        endif()
        _rocblas_deprecation_warning(SKIP_LIBRARY ROCBLAS_ENABLE_HOST)
        list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "SKIP_LIBRARY=${SKIP_LIBRARY}")
        list(APPEND _ROCBLAS_CURRENT_OPTIONS "ROCBLAS_ENABLE_HOST=${ROCBLAS_ENABLE_HOST}")
    endif()
endif()

# ==============================================================================
# Display Migration Guidance
# ==============================================================================

if(_ROCBLAS_LEGACY_OPTIONS_USED)
    # Format the current options with -D prefix for each
    string(REPLACE ";" " -D" _formatted_current_options "${_ROCBLAS_CURRENT_OPTIONS}")
    set(_formatted_current_options "-D${_formatted_current_options}")

    message(DEPRECATION
        "\n"
        "  Legacy build options detected:\n"
        "    ${_ROCBLAS_LEGACY_OPTIONS_USED}\n"
        "\n"
        "  To use current options, run:\n"
        "    cmake ${_formatted_current_options} ..\n"
        "\n"
        "  To suppress warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...\n"
    )
endif()
