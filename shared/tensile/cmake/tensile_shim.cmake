# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ==============================================================================
# Backward Compatibility Shims for Tensile Build Options
# ==============================================================================
# This file maps legacy build option names to modern project-specific options.
# It provides a transition period for users to update their build scripts while
# maintaining backward compatibility.
#
# Users can suppress deprecation warnings with: -DCMAKE_WARN_DEPRECATED=OFF
#
# Legacy → Modern Mappings:
# ------------------------
# Tensile_ARCHITECTURE              → GPU_TARGETS
# TENSILE_BUILD_CLIENT              → TENSILE_ENABLE_CLIENT
# TENSILE_USE_MSGPACK               → TENSILE_ENABLE_MSGPACK
# TENSILE_USE_LLVM                  → TENSILE_ENABLE_LLVM
# Tensile_LIBRARY_FORMAT            → TENSILE_ENABLE_MSGPACK/TENSILE_ENABLE_LLVM
# Tensile_CPU_THREADS               → TENSILE_JOBS
# Tensile_CODE_OBJECT_VERSION       → TENSILE_CODE_OBJECT_VERSION
# Tensile_MERGE_FILES               → TENSILE_MERGE_FILES
# Tensile_SHORT_FILENAMES           → TENSILE_SHORT_FILENAMES
# Tensile_PRINT_DEBUG               → TENSILE_VERBOSITY
# Tensile_SEPARATE_ARCHITECTURES    → TENSILE_SEPARATE_ARCHITECTURES
# Tensile_LAZY_LIBRARY_LOADING      → TENSILE_LAZY_LIBRARY_LOADING
# Tensile_GENERATE_PACKAGE          → TENSILE_GENERATE_PACKAGE
# Tensile_KEEP_BUILD_TMP            → TENSILE_KEEP_BUILD_TMP
# Tensile_VERBOSE                   → TENSILE_VERBOSITY
# Tensile_NO_ENUMERATE              → TENSILE_NO_ENUMERATE
# ==============================================================================

# Helper macro for deprecation warnings using native CMake mechanism
macro(_tensile_deprecation_warning old_var new_var)
    message(DEPRECATION
        "The option '${old_var}' is deprecated and will be removed in a future release.\n"
        "Please use '${new_var}' instead.\n"
        "To suppress these warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
endmacro()

# Helper macro for conflict detection
macro(_tensile_check_conflict old_var new_var)
    # Only check for conflicts if both variables are defined.
    if(DEFINED ${new_var} AND DEFINED ${old_var})
        # A conflict only occurs if both variables have non-empty values and they differ.
        # Empty strings are treated as "not set" for conflict purposes.
        if("${${old_var}}" AND "${${new_var}}")
            if(NOT "${${old_var}}" STREQUAL "${${new_var}}")
                message(FATAL_ERROR
                    "Conflicting options detected:\n"
                    "  ${old_var}=${${old_var}} (deprecated)\n"
                    "  ${new_var}=${${new_var}}\n"
                    "Please remove ${old_var} from your build configuration and use only ${new_var}.")
            endif()
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
        _tensile_check_conflict(${legacy_var} ${modern_var})
        if(NOT DEFINED ${modern_var})
            set(${modern_var} ${${legacy_var}} CACHE ${var_type} "${description}" FORCE)
            _tensile_deprecation_warning(${legacy_var} ${modern_var})
            list(APPEND _TENSILE_LEGACY_OPTIONS_USED "${legacy_var}=${${legacy_var}}")
            list(APPEND _TENSILE_CURRENT_OPTIONS "${modern_var}=${${modern_var}}")
            # Propagate lists to parent scope
            set(_TENSILE_LEGACY_OPTIONS_USED "${_TENSILE_LEGACY_OPTIONS_USED}" PARENT_SCOPE)
            set(_TENSILE_CURRENT_OPTIONS "${_TENSILE_CURRENT_OPTIONS}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ==============================================================================
# Apply Legacy Option Mappings
# ==============================================================================

# Initialize tracking lists for migration guidance
set(_TENSILE_LEGACY_OPTIONS_USED "")
set(_TENSILE_CURRENT_OPTIONS "")

# Apply mappings using consolidated function
shim_mapping(Tensile_ARCHITECTURE GPU_TARGETS "AMD GFX targets to cross-compile" STRING)
shim_mapping(TENSILE_BUILD_CLIENT TENSILE_ENABLE_CLIENT "Build client app")
shim_mapping(TENSILE_USE_MSGPACK TENSILE_ENABLE_MSGPACK "Enable MessagePack support")
shim_mapping(TENSILE_USE_LLVM TENSILE_ENABLE_LLVM "Use llvm yaml library")
shim_mapping(Tensile_CPU_THREADS TENSILE_JOBS "Number of parallel jobs" STRING)
shim_mapping(Tensile_CODE_OBJECT_VERSION TENSILE_CODE_OBJECT_VERSION "Code object version" STRING)
shim_mapping(Tensile_MERGE_FILES TENSILE_MERGE_FILES "Merge files")
shim_mapping(Tensile_SHORT_FILENAMES TENSILE_SHORT_FILENAMES "Short filenames")
shim_mapping(Tensile_PRINT_DEBUG TENSILE_VERBOSITY "Debug verbosity" STRING)
shim_mapping(Tensile_SEPARATE_ARCHITECTURES TENSILE_SEPARATE_ARCHITECTURES "Separate architectures")
shim_mapping(Tensile_LAZY_LIBRARY_LOADING TENSILE_LAZY_LIBRARY_LOADING "Lazy loading")
shim_mapping(Tensile_GENERATE_PACKAGE TENSILE_GENERATE_PACKAGE "Generate Tensile package")
shim_mapping(Tensile_KEEP_BUILD_TMP TENSILE_KEEP_BUILD_TMP "Keep temporary build files")
shim_mapping(Tensile_VERBOSE TENSILE_VERBOSITY "Verbosity level" STRING)
shim_mapping(Tensile_NO_ENUMERATE TENSILE_NO_ENUMERATE "Disable GPU enumeration")

# Map Tensile_LIBRARY_FORMAT → TENSILE_ENABLE_MSGPACK/TENSILE_ENABLE_LLVM
if(DEFINED Tensile_LIBRARY_FORMAT)
    _tensile_deprecation_warning(Tensile_LIBRARY_FORMAT
                                  "TENSILE_ENABLE_MSGPACK and TENSILE_ENABLE_LLVM")
    list(APPEND _TENSILE_LEGACY_OPTIONS_USED "Tensile_LIBRARY_FORMAT=${Tensile_LIBRARY_FORMAT}")
    if(Tensile_LIBRARY_FORMAT MATCHES "msgpack")
        if(NOT DEFINED TENSILE_ENABLE_MSGPACK)
            set(TENSILE_ENABLE_MSGPACK ON CACHE BOOL "Enable MessagePack support" FORCE)
        endif()
        if(NOT DEFINED TENSILE_ENABLE_LLVM)
            set(TENSILE_ENABLE_LLVM OFF CACHE BOOL "Use llvm yaml library" FORCE)
        endif()
        list(APPEND _TENSILE_CURRENT_OPTIONS "TENSILE_ENABLE_MSGPACK=ON")
        list(APPEND _TENSILE_CURRENT_OPTIONS "TENSILE_ENABLE_LLVM=OFF")
    elseif(Tensile_LIBRARY_FORMAT MATCHES "yaml")
        if(NOT DEFINED TENSILE_ENABLE_LLVM)
            set(TENSILE_ENABLE_LLVM ON CACHE BOOL "Use llvm yaml library" FORCE)
        endif()
        if(NOT DEFINED TENSILE_ENABLE_MSGPACK)
            set(TENSILE_ENABLE_MSGPACK OFF CACHE BOOL "Enable MessagePack support" FORCE)
        endif()
        list(APPEND _TENSILE_CURRENT_OPTIONS "TENSILE_ENABLE_LLVM=ON")
        list(APPEND _TENSILE_CURRENT_OPTIONS "TENSILE_ENABLE_MSGPACK=OFF")
    endif()
endif()

# ==============================================================================
# Display Migration Guidance
# ==============================================================================

if(_TENSILE_LEGACY_OPTIONS_USED)
    # Format the current options with -D prefix for each
    string(REPLACE ";" " -D" _formatted_current_options "${_TENSILE_CURRENT_OPTIONS}")
    set(_formatted_current_options "-D${_formatted_current_options}")

    message(DEPRECATION
        "\n"
        "  Legacy build options detected:\n"
        "    ${_TENSILE_LEGACY_OPTIONS_USED}\n"
        "\n"
        "  To use current options, run:\n"
        "    cmake ${_formatted_current_options} ..\n"
        "\n"
        "  To suppress warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...\n"
    )
endif()
