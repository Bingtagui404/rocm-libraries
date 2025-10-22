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
set(_TENSILE_LEGACY_OPTIONS_USED "")
set(_TENSILE_CURRENT_OPTIONS "")

# Map Tensile_ARCHITECTURE → GPU_TARGETS
if(DEFINED Tensile_ARCHITECTURE)
    _tensile_check_conflict(Tensile_ARCHITECTURE GPU_TARGETS)
    if(NOT DEFINED GPU_TARGETS)
        set(GPU_TARGETS "${Tensile_ARCHITECTURE}" CACHE STRING 
            "AMD GFX targets to cross-compile" FORCE)
        _tensile_deprecation_warning(Tensile_ARCHITECTURE GPU_TARGETS)
        list(APPEND _TENSILE_LEGACY_OPTIONS_USED "Tensile_ARCHITECTURE=\"${Tensile_ARCHITECTURE}\"")
        list(APPEND _TENSILE_CURRENT_OPTIONS "GPU_TARGETS=\"${GPU_TARGETS}\"")
    endif()
endif()

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

# Map TENSILE_BUILD_CLIENT → TENSILE_ENABLE_CLIENT
if(DEFINED TENSILE_BUILD_CLIENT)
    _tensile_check_conflict(TENSILE_BUILD_CLIENT TENSILE_ENABLE_CLIENT)
    if(NOT DEFINED TENSILE_ENABLE_CLIENT)
        set(TENSILE_ENABLE_CLIENT ${TENSILE_BUILD_CLIENT} CACHE BOOL 
            "Build client app" FORCE)
        _tensile_deprecation_warning(TENSILE_BUILD_CLIENT TENSILE_ENABLE_CLIENT)
        list(APPEND _TENSILE_LEGACY_OPTIONS_USED "TENSILE_BUILD_CLIENT=${TENSILE_BUILD_CLIENT}")
        list(APPEND _TENSILE_CURRENT_OPTIONS "TENSILE_ENABLE_CLIENT=${TENSILE_ENABLE_CLIENT}")
    endif()
endif()

# Map TENSILE_USE_MSGPACK → TENSILE_ENABLE_MSGPACK
if(DEFINED TENSILE_USE_MSGPACK)
    _tensile_check_conflict(TENSILE_USE_MSGPACK TENSILE_ENABLE_MSGPACK)
    if(NOT DEFINED TENSILE_ENABLE_MSGPACK)
        set(TENSILE_ENABLE_MSGPACK ${TENSILE_USE_MSGPACK} CACHE BOOL 
            "Enable MessagePack support" FORCE)
        _tensile_deprecation_warning(TENSILE_USE_MSGPACK TENSILE_ENABLE_MSGPACK)
        list(APPEND _TENSILE_LEGACY_OPTIONS_USED "TENSILE_USE_MSGPACK=${TENSILE_USE_MSGPACK}")
        list(APPEND _TENSILE_CURRENT_OPTIONS "TENSILE_ENABLE_MSGPACK=${TENSILE_ENABLE_MSGPACK}")
    endif()
endif()

# Map TENSILE_USE_LLVM → TENSILE_ENABLE_LLVM
if(DEFINED TENSILE_USE_LLVM)
    _tensile_check_conflict(TENSILE_USE_LLVM TENSILE_ENABLE_LLVM)
    if(NOT DEFINED TENSILE_ENABLE_LLVM)
        set(TENSILE_ENABLE_LLVM ${TENSILE_USE_LLVM} CACHE BOOL 
            "Use llvm yaml library" FORCE)
        _tensile_deprecation_warning(TENSILE_USE_LLVM TENSILE_ENABLE_LLVM)
        list(APPEND _TENSILE_LEGACY_OPTIONS_USED "TENSILE_USE_LLVM=${TENSILE_USE_LLVM}")
        list(APPEND _TENSILE_CURRENT_OPTIONS "TENSILE_ENABLE_LLVM=${TENSILE_ENABLE_LLVM}")
    endif()
endif()

# Map BUILD_SHARED_LIBS → TENSILE_BUILD_SHARED_LIBS (for clarity in multi-project builds)
# Note: BUILD_SHARED_LIBS is still valid CMake, but project-specific is clearer
if(DEFINED BUILD_SHARED_LIBS AND NOT DEFINED TENSILE_BUILD_SHARED_LIBS)
    set(TENSILE_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL 
        "Build shared library" FORCE)
    message(DEPRECATION 
        "Using BUILD_SHARED_LIBS for Tensile. "
        "Consider using TENSILE_BUILD_SHARED_LIBS for clarity in multi-project builds.\n"
        "To suppress: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
    list(APPEND _TENSILE_LEGACY_OPTIONS_USED "BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}")
    list(APPEND _TENSILE_CURRENT_OPTIONS "TENSILE_BUILD_SHARED_LIBS=${TENSILE_BUILD_SHARED_LIBS}")
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
