# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# SageAttention V3 preprocessing benchmark — gfx950 only.
# Exercises the four-kernel pipeline:
#   k_mean → preprocess Q/K → V tile-transpose → delta_s GEMM

if(NOT "gfx950" IN_LIST GPU_TARGETS)
    message(STATUS "Skipping 51_sageattn_v3_preprocess: requires gfx950")
    return()
endif()

set(EXAMPLE_SAGEATTN_V3_PREPROCESS "tile_example_sageattn_v3_preprocess")

set_source_files_properties(
    ${CMAKE_CURRENT_LIST_DIR}/sageattn_v3_preprocess.cpp
    PROPERTIES LANGUAGE HIP)

add_executable(${EXAMPLE_SAGEATTN_V3_PREPROCESS} EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_LIST_DIR}/sageattn_v3_preprocess.cpp)

set_property(TARGET ${EXAMPLE_SAGEATTN_V3_PREPROCESS}
    PROPERTY HIP_ARCHITECTURES "gfx950")

target_include_directories(${EXAMPLE_SAGEATTN_V3_PREPROCESS}
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${PROJECT_SOURCE_DIR}/include)

target_compile_definitions(${EXAMPLE_SAGEATTN_V3_PREPROCESS} PRIVATE
    CK_USE_NATIVE_MX_SUPPORT=1
    CK_USE_GFX950=1
    CK_USE_XDL=1)

target_compile_options(${EXAMPLE_SAGEATTN_V3_PREPROCESS} PRIVATE
    -Wno-global-constructors
    -Wno-undef
    -Wno-covered-switch-default
    -Wno-missing-field-initializers)

set_target_properties(${EXAMPLE_SAGEATTN_V3_PREPROCESS} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
