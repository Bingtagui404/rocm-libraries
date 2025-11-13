# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ==============================================================================
# rocBLAS Supported GPU Architectures
# ==============================================================================
# This file defines the GPU architectures supported by rocBLAS.
#
# Architecture support is managed through version control (Git branches/tags)
# rather than build-time conditionals. Each branch/tag in version control
# defines the appropriate architecture list for that ROCm release.
#
# ASAN builds use a restricted set of architectures with xnack+ support.
# ==============================================================================

# Architecture support is managed through version control (Git branches/tags)
# rather than conditional logic based on ROCM_PLATFORM_VERSION.
# This simplifies the build system and aligns with ROCm best practices.
if(NOT ROCBLAS_ENABLE_ASAN)
    # Standard build: all supported GPU architectures
    set(SUPPORTED_TARGETS 
        "gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-"
        "gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102"
        "gfx942;gfx950;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201"
    )
else()
    # Address sanitizer builds: architectures with xnack+ support
    set(SUPPORTED_TARGETS 
        "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+"
    )
endif()

function(rocblas_validate_gpu_targets targets output_var)
    # Expand gfx90a to xnack variants for backward compatibility
    set(target_list_expanded "")
    foreach(target IN LISTS targets)
        if(target STREQUAL "gfx90a")
            list(APPEND target_list_expanded "gfx90a:xnack-")
            list(APPEND target_list_expanded "gfx90a:xnack+")
        else()
            list(APPEND target_list_expanded "${target}")
        endif()
    endforeach()

    set(supported_list ${SUPPORTED_TARGETS})
    set(target_list ${target_list_expanded})

    string(REGEX REPLACE ";" " " supported_flat "${supported_list}")
    string(REGEX REPLACE " +" ";" supported_list "${supported_flat}")

    string(REGEX REPLACE ";" " " target_flat "${target_list}")
    string(REGEX REPLACE " +" ";" target_list "${target_flat}")

    foreach(target IN LISTS target_list)
        list(FIND supported_list "${target}" idx)
        if(idx EQUAL -1)
            message(FATAL_ERROR "Unsupported GPU target: ${target}\nSupported targets are: ${supported_list}")
        endif()
    endforeach()
    
    # Return the expanded target list
    set(${output_var} ${target_list} PARENT_SCOPE)
endfunction()

function(rocblas_get_base_architectures output_var)
    set(${output_var} ${SUPPORTED_TARGETS} PARENT_SCOPE)
endfunction()

function(rocblas_get_supported_architectures output_var)
    set(${output_var} ${SUPPORTED_TARGETS} PARENT_SCOPE)
endfunction()
