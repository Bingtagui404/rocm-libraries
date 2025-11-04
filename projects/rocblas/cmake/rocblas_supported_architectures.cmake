# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ==============================================================================
# rocBLAS Supported GPU Architectures
# ==============================================================================
# This file defines the GPU architectures supported by rocBLAS.
# Architecture support is versioned to align with ROCm releases.
#
# The appropriate target list is selected based on ROCM_PLATFORM_VERSION.
# If ROCM_PLATFORM_VERSION is not set, defaults to the latest (7.1).
# ==============================================================================

if(NOT ROCBLAS_ENABLE_ASAN)
    # Standard build target lists by ROCm version
    # Common architecture groups to avoid repetition
    set(GFX9_ARCHS "gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-")
    set(GFX10_11_ARCHS "gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102")
    
    set(TARGET_LIST_ROCM_5.6 "gfx803;${GFX9_ARCHS};${GFX10_11_ARCHS}")
    set(TARGET_LIST_ROCM_5.7 "gfx803;${GFX9_ARCHS};gfx942;${GFX10_11_ARCHS}")
    set(TARGET_LIST_ROCM_6.0 "${GFX9_ARCHS};gfx942;${GFX10_11_ARCHS}")
    set(TARGET_LIST_ROCM_6.3 "${GFX9_ARCHS};gfx942;${GFX10_11_ARCHS};gfx1151;gfx1200;gfx1201")
    set(TARGET_LIST_ROCM_7.0 "${GFX9_ARCHS};gfx942;gfx950;${GFX10_11_ARCHS};gfx1150;gfx1151;gfx1200;gfx1201")
    set(TARGET_LIST_ROCM_7.1 "${GFX9_ARCHS};gfx942;gfx950;${GFX10_11_ARCHS};gfx1103;gfx1150;gfx1151;gfx1200;gfx1201")
else()
    # Address sanitizer build target lists by ROCm version (require xnack+)
    # Build incrementally to avoid repetition
    set(ASAN_BASE "gfx908:xnack+;gfx90a:xnack+")
    set(ASAN_WITH_942 "${ASAN_BASE};gfx942:xnack+")
    set(ASAN_WITH_950 "${ASAN_WITH_942};gfx950:xnack+")
    
    set(TARGET_LIST_ROCM_5.6 "${ASAN_BASE}")
    set(TARGET_LIST_ROCM_5.7 "${ASAN_WITH_942}")
    set(TARGET_LIST_ROCM_6.0 "${ASAN_WITH_942}")
    set(TARGET_LIST_ROCM_6.3 "${ASAN_WITH_942}")
    set(TARGET_LIST_ROCM_7.0 "${ASAN_WITH_950}")
    set(TARGET_LIST_ROCM_7.1 "${ASAN_WITH_950}")
endif()

# Select appropriate target list based on ROCm platform version
if(ROCM_PLATFORM_VERSION)
    if(${ROCM_PLATFORM_VERSION} VERSION_LESS 5.7.0)
        set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_5.6}")
    elseif(${ROCM_PLATFORM_VERSION} VERSION_LESS 6.0.0)
        set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_5.7}")
    elseif(${ROCM_PLATFORM_VERSION} VERSION_LESS 6.3.0)
        set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_6.0}")
    elseif(${ROCM_PLATFORM_VERSION} VERSION_LESS 7.0.0)
        set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_6.3}")
    elseif(${ROCM_PLATFORM_VERSION} VERSION_LESS 7.1.0)
        set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_7.0}")
    else()
        set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_7.1}")
    endif()
    message(STATUS "ROCm Platform Version: ${ROCM_PLATFORM_VERSION}, using corresponding supported GPU targets")
else()
    message(STATUS "ROCM_PLATFORM_VERSION is not set, using latest supported GPU targets (ROCm 7.1)")
    set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_7.1}")
endif()

function(rocblas_validate_gpu_targets targets)
    set(supported_list ${SUPPORTED_TARGETS})
    set(target_list ${targets})

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
endfunction()

function(rocblas_get_base_architectures output_var)
    set(${output_var} ${SUPPORTED_TARGETS} PARENT_SCOPE)
endfunction()

function(rocblas_get_supported_architectures output_var)
    set(${output_var} ${SUPPORTED_TARGETS} PARENT_SCOPE)
endfunction()
