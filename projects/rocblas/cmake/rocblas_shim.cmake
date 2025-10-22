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
# AMDGPU_TARGETS            → GPU_TARGETS
# BUILD_SHARED_LIBS         → ROCBLAS_BUILD_SHARED_LIBS
# BUILD_ADDRESS_SANITIZER   → ROCBLAS_ENABLE_ASAN
# BUILD_CODE_COVERAGE       → ROCBLAS_BUILD_COVERAGE
# BUILD_VERBOSE             → (deprecated, no replacement)
# SKIP_LIBRARY              → (deprecated, no replacement)
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

# ==============================================================================
# Apply Legacy Option Mappings
# ==============================================================================

# Initialize tracking lists for migration guidance
set(_ROCBLAS_LEGACY_OPTIONS_USED "")
set(_ROCBLAS_CURRENT_OPTIONS "")

# Map BUILD_CLIENTS_TESTS → ROCBLAS_BUILD_TESTING
if(DEFINED BUILD_CLIENTS_TESTS)
    _rocblas_check_conflict(BUILD_CLIENTS_TESTS ROCBLAS_BUILD_TESTING)
    if(NOT DEFINED ROCBLAS_BUILD_TESTING)
        set(ROCBLAS_BUILD_TESTING ${BUILD_CLIENTS_TESTS} CACHE BOOL 
            "Build test client; master switch." FORCE)
        _rocblas_deprecation_warning(BUILD_CLIENTS_TESTS ROCBLAS_BUILD_TESTING)
        list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "BUILD_CLIENTS_TESTS=${BUILD_CLIENTS_TESTS}")
        list(APPEND _ROCBLAS_CURRENT_OPTIONS "ROCBLAS_BUILD_TESTING=${ROCBLAS_BUILD_TESTING}")
    endif()
endif()

# Map BUILD_CLIENTS_BENCHMARKS → ROCBLAS_ENABLE_BENCHMARKS
if(DEFINED BUILD_CLIENTS_BENCHMARKS)
    _rocblas_check_conflict(BUILD_CLIENTS_BENCHMARKS ROCBLAS_ENABLE_BENCHMARKS)
    if(NOT DEFINED ROCBLAS_ENABLE_BENCHMARKS)
        set(ROCBLAS_ENABLE_BENCHMARKS ${BUILD_CLIENTS_BENCHMARKS} CACHE BOOL 
            "Build benchmark client." FORCE)
        _rocblas_deprecation_warning(BUILD_CLIENTS_BENCHMARKS ROCBLAS_ENABLE_BENCHMARKS)
        list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "BUILD_CLIENTS_BENCHMARKS=${BUILD_CLIENTS_BENCHMARKS}")
        list(APPEND _ROCBLAS_CURRENT_OPTIONS "ROCBLAS_ENABLE_BENCHMARKS=${ROCBLAS_ENABLE_BENCHMARKS}")
    endif()
endif()

# Map BUILD_CLIENTS_SAMPLES → ROCBLAS_ENABLE_SAMPLES
if(DEFINED BUILD_CLIENTS_SAMPLES)
    _rocblas_check_conflict(BUILD_CLIENTS_SAMPLES ROCBLAS_ENABLE_SAMPLES)
    if(NOT DEFINED ROCBLAS_ENABLE_SAMPLES)
        set(ROCBLAS_ENABLE_SAMPLES ${BUILD_CLIENTS_SAMPLES} CACHE BOOL 
            "Build client samples." FORCE)
        _rocblas_deprecation_warning(BUILD_CLIENTS_SAMPLES ROCBLAS_ENABLE_SAMPLES)
        list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "BUILD_CLIENTS_SAMPLES=${BUILD_CLIENTS_SAMPLES}")
        list(APPEND _ROCBLAS_CURRENT_OPTIONS "ROCBLAS_ENABLE_SAMPLES=${ROCBLAS_ENABLE_SAMPLES}")
    endif()
endif()

# Map BUILD_CLIENTS → ROCBLAS_ENABLE_CLIENT
if(DEFINED BUILD_CLIENTS)
    _rocblas_check_conflict(BUILD_CLIENTS ROCBLAS_ENABLE_CLIENT)
    if(NOT DEFINED ROCBLAS_ENABLE_CLIENT)
        if(BUILD_CLIENTS)
            set(ROCBLAS_ENABLE_CLIENT ON CACHE BOOL 
                "Build rocBLAS clients." FORCE)
            _rocblas_deprecation_warning(BUILD_CLIENTS ROCBLAS_ENABLE_CLIENT)
            list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "BUILD_CLIENTS=${BUILD_CLIENTS}")
            list(APPEND _ROCBLAS_CURRENT_OPTIONS "ROCBLAS_ENABLE_CLIENT=${ROCBLAS_ENABLE_CLIENT}")
        endif()
    endif()
endif()

# Map BUILD_WITH_TENSILE → ROCBLAS_ENABLE_TENSILE
if(DEFINED BUILD_WITH_TENSILE)
    _rocblas_check_conflict(BUILD_WITH_TENSILE ROCBLAS_ENABLE_TENSILE)
    if(NOT DEFINED ROCBLAS_ENABLE_TENSILE)
        set(ROCBLAS_ENABLE_TENSILE ${BUILD_WITH_TENSILE} CACHE BOOL 
            "Build Tensile GEMM device libraries." FORCE)
        _rocblas_deprecation_warning(BUILD_WITH_TENSILE ROCBLAS_ENABLE_TENSILE)
        list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "BUILD_WITH_TENSILE=${BUILD_WITH_TENSILE}")
        list(APPEND _ROCBLAS_CURRENT_OPTIONS "ROCBLAS_ENABLE_TENSILE=${ROCBLAS_ENABLE_TENSILE}")
    endif()
endif()

# Map AMDGPU_TARGETS → GPU_TARGETS
if(DEFINED AMDGPU_TARGETS)
    _rocblas_check_conflict(AMDGPU_TARGETS GPU_TARGETS)
    if(NOT DEFINED GPU_TARGETS)
        set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING 
            "AMD GFX targets to cross-compile" FORCE)
        _rocblas_deprecation_warning(AMDGPU_TARGETS GPU_TARGETS)
        list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "AMDGPU_TARGETS=\"${AMDGPU_TARGETS}\"")
        list(APPEND _ROCBLAS_CURRENT_OPTIONS "GPU_TARGETS=\"${GPU_TARGETS}\"")
    endif()
endif()

# Map BUILD_ADDRESS_SANITIZER → ROCBLAS_ENABLE_ASAN
if(DEFINED BUILD_ADDRESS_SANITIZER)
    _rocblas_check_conflict(BUILD_ADDRESS_SANITIZER ROCBLAS_ENABLE_ASAN)
    if(NOT DEFINED ROCBLAS_ENABLE_ASAN)
        set(ROCBLAS_ENABLE_ASAN ${BUILD_ADDRESS_SANITIZER} CACHE BOOL 
            "Build with address sanitizer enabled." FORCE)
        _rocblas_deprecation_warning(BUILD_ADDRESS_SANITIZER ROCBLAS_ENABLE_ASAN)
        list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "BUILD_ADDRESS_SANITIZER=${BUILD_ADDRESS_SANITIZER}")
        list(APPEND _ROCBLAS_CURRENT_OPTIONS "ROCBLAS_ENABLE_ASAN=${ROCBLAS_ENABLE_ASAN}")
    endif()
endif()

# Map BUILD_CODE_COVERAGE → ROCBLAS_BUILD_COVERAGE
if(DEFINED BUILD_CODE_COVERAGE)
    _rocblas_check_conflict(BUILD_CODE_COVERAGE ROCBLAS_BUILD_COVERAGE)
    if(NOT DEFINED ROCBLAS_BUILD_COVERAGE)
        set(ROCBLAS_BUILD_COVERAGE ${BUILD_CODE_COVERAGE} CACHE BOOL 
            "Build tests with coverage enabled." FORCE)
        _rocblas_deprecation_warning(BUILD_CODE_COVERAGE ROCBLAS_BUILD_COVERAGE)
        list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "BUILD_CODE_COVERAGE=${BUILD_CODE_COVERAGE}")
        list(APPEND _ROCBLAS_CURRENT_OPTIONS "ROCBLAS_BUILD_COVERAGE=${ROCBLAS_BUILD_COVERAGE}")
    endif()
endif()

# Deprecated options with no replacement
if(DEFINED BUILD_VERBOSE)
    message(DEPRECATION 
        "The option 'BUILD_VERBOSE' is deprecated and has no replacement.\n"
        "Verbose build information should be controlled via CMAKE_VERBOSE_MAKEFILE.\n"
        "To suppress: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
    list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "BUILD_VERBOSE=${BUILD_VERBOSE}")
endif()

if(DEFINED SKIP_LIBRARY)
    message(DEPRECATION 
        "The option 'SKIP_LIBRARY' is deprecated and has no replacement.\n"
        "This was an anti-pattern. Use proper CMake configuration options instead.\n"
        "To suppress: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
    list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "SKIP_LIBRARY=${SKIP_LIBRARY}")
endif()

# Map BUILD_SHARED_LIBS → ROCBLAS_BUILD_SHARED_LIBS (for clarity)
if(DEFINED BUILD_SHARED_LIBS AND NOT DEFINED ROCBLAS_BUILD_SHARED_LIBS)
    set(ROCBLAS_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL 
        "Build the rocBLAS shared or static library." FORCE)
    message(DEPRECATION 
        "Using BUILD_SHARED_LIBS for rocBLAS. "
        "Consider using ROCBLAS_BUILD_SHARED_LIBS for clarity in multi-project builds.\n"
        "To suppress: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
    list(APPEND _ROCBLAS_LEGACY_OPTIONS_USED "BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}")
    list(APPEND _ROCBLAS_CURRENT_OPTIONS "ROCBLAS_BUILD_SHARED_LIBS=${ROCBLAS_BUILD_SHARED_LIBS}")
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
