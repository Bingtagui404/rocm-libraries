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
# Tensile_ARCHITECTURE      → GPU_TARGETS
# TENSILE_BUILD_CLIENT      → TENSILE_ENABLE_CLIENT
# TENSILE_USE_MSGPACK       → TENSILE_ENABLE_MSGPACK
# TENSILE_USE_LLVM          → TENSILE_ENABLE_LLVM
# Tensile_LIBRARY_FORMAT    → TENSILE_ENABLE_MSGPACK/TENSILE_ENABLE_LLVM
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

# Special case for BUILD_SHARED_LIBS → TENSILE_BUILD_SHARED_LIBS
shim_mapping(BUILD_SHARED_LIBS TENSILE_BUILD_SHARED_LIBS "Build shared library")

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
