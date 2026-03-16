# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(GEMM_ROWCOL_DATATYPE "fp8;bf8" CACHE STRING "List of datatypes for GEMM RowCol (semicolon-separated)")
set(GEMM_ROWCOL_LAYOUT "rcr;rrr;ccr;crr" CACHE STRING "List of layout for GEMM RowCol (semicolon-separated)")
set(GEMM_ROWCOL_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
option(ENABLE_CCACHE_GEMM_ROWCOL "Enable ccache for GEMM RowCol ops compilation" OFF)

# Store the directory path for use in functions
set(GEMM_ROWCOL_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Function to create individual GEMM RowCol targets
function(create_individual_gemm_rowcol_target datatype layout trait tile_config config_json)
    if(NOT GEMM_ROWCOL_GPU_TARGETS_INDIVIDUAL)
        message(WARNING "Skipping individual GEMM RowCol target ${datatype}_${layout}_${trait}_${tile_config}: No supported GPU targets")
        return()
    endif()

    # Parse tile configuration: format is tile_mxtile_nxtile_k_warp_mxwarp_nxwarp_k_warp_tile_mxwarp_tile_nxwarp_tile_k
    string(REPLACE "_" ";" config_groups ${tile_config})
    list(GET config_groups 0 tile_dims)
    list(GET config_groups 1 warp_dims)
    list(GET config_groups 2 warp_tile_dims)

    string(REPLACE "x" ";" tile_parts ${tile_dims})
    list(GET tile_parts 0 tile_m)
    list(GET tile_parts 1 tile_n)
    list(GET tile_parts 2 tile_k)

    string(REPLACE "x" ";" warp_parts ${warp_dims})
    list(GET warp_parts 0 warp_m)
    list(GET warp_parts 1 warp_n)
    list(GET warp_parts 2 warp_k)

    string(REPLACE "x" ";" warp_tile_parts ${warp_tile_dims})
    list(GET warp_tile_parts 0 warp_tile_m)
    list(GET warp_tile_parts 1 warp_tile_n)
    list(GET warp_tile_parts 2 warp_tile_k)

    set(target_name "benchmark_gemm_rowcol_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Generate the single instance header for this kernel
    set(instance_header "${working_path}/gemm_rowcol_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")

    # Add custom command to generate the header file at build time
    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${GEMM_ROWCOL_SOURCE_DIR}/gemm_rowcol_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${config_json}
                --gen_single
                --kernel_name "gemm_rowcol_${datatype}_${layout}_${trait}_${tile_config}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
                --gpu_target "${GEMM_ROWCOL_GPU_TARGETS_INDIVIDUAL}"
        DEPENDS ${GEMM_ROWCOL_SOURCE_DIR}/gemm_rowcol_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    # Create the executable
    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${GEMM_ROWCOL_SOURCE_DIR}/gemm_rowcol_benchmark_single.cpp
        ${instance_header}
    )

    # Set GPU architectures
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GEMM_ROWCOL_GPU_TARGETS_INDIVIDUAL})

    # Set compile definitions
    target_compile_definitions(${target_name} PRIVATE
        GEMM_ROWCOL_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${GEMM_ROWCOL_SOURCE_DIR}
        ${working_path}
    )

    # Compile options
    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header}
    )

    # Add to collection targets
    add_dependencies(benchmark_gemm_rowcol_all ${target_name})
    add_dependencies(benchmark_gemm_rowcol_${datatype} ${target_name})
    add_dependencies(benchmark_gemm_rowcol_${layout} ${target_name})
    add_dependencies(benchmark_gemm_rowcol_${datatype}_${layout} ${target_name})

    # Add to pipeline-specific targets
    string(REPLACE "_" ";" trait_parts ${trait})
    list(GET trait_parts 0 pipeline)

    add_dependencies(benchmark_gemm_rowcol_${pipeline}_pipeline ${target_name})
endfunction()

# Function to build individual GEMM RowCol targets
function(build_individual_gemm_rowcol_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Choose config file (priority: env var > cmake var > default)
    if(DEFINED ENV{GEMM_ROWCOL_CONFIG_FILE} AND NOT "$ENV{GEMM_ROWCOL_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{GEMM_ROWCOL_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE "  Using config from environment variable: ${config_filename}")
    elseif(NOT "${GEMM_ROWCOL_CONFIG_FILE}" STREQUAL "")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${GEMM_ROWCOL_CONFIG_FILE}")
        message(VERBOSE "  Using custom config: ${GEMM_ROWCOL_CONFIG_FILE}")
    else()
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE "  Using default config for layout ${layout}")
    endif()

    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    # Create working directory
    file(MAKE_DIRECTORY ${working_path})

    # List kernels
    message(VERBOSE "  Listing RowCol kernel configurations for ${datatype} ${layout}...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/gemm_rowcol_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --gpu_target "${GEMM_ROWCOL_GPU_TARGETS_INDIVIDUAL}"
                --list_kernels
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list RowCol kernels for ${datatype} ${layout}: ${list_error}")
    endif()

    # Read kernel count
    if(EXISTS ${working_path}/gemm_rowcol_kernel_count.txt)
        file(READ ${working_path}/gemm_rowcol_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} RowCol kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found")
    endif()

    # Read kernel list and create targets
    if(EXISTS ${working_path}/gemm_rowcol_kernel_list.txt)
        file(STRINGS ${working_path}/gemm_rowcol_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            create_individual_gemm_rowcol_target("${datatype}" "${layout}" "${trait_combo}" "${tile_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found")
    endif()
endfunction()

# Main build logic
message(VERBOSE "=== Starting Tile Engine GEMM RowCol Configuration ===")
message(VERBOSE "GEMM_ROWCOL_DATATYPE: ${GEMM_ROWCOL_DATATYPE}")
message(VERBOSE "GEMM_ROWCOL_LAYOUT: ${GEMM_ROWCOL_LAYOUT}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

# Filter GPU targets to supported architectures
set(GEMM_ROWCOL_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND GEMM_ROWCOL_GPU_TARGETS_INDIVIDUAL ${target})
        message(VERBOSE "  Adding GPU target: ${target}")
    endif()
endforeach()

# Skip build if no matching targets found
if(NOT GEMM_ROWCOL_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine GEMM RowCol build: No supported GPU targets (gfx90a, gfx942, gfx950) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building individual GEMM RowCol targets for GPU targets: ${GEMM_ROWCOL_GPU_TARGETS_INDIVIDUAL}")

    # Enable compiler cache if requested
    if(ENABLE_CCACHE_GEMM_ROWCOL)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(VERBOSE "Using ccache for faster compilation")
        endif()
    endif()

    # Create master collection target
    add_custom_target(benchmark_gemm_rowcol_all)

    # Create datatype collection targets
    foreach(dt IN LISTS GEMM_ROWCOL_DATATYPE)
        add_custom_target(benchmark_gemm_rowcol_${dt})
    endforeach()

    # Create layout collection targets
    foreach(l IN LISTS GEMM_ROWCOL_LAYOUT)
        add_custom_target(benchmark_gemm_rowcol_${l})
    endforeach()

    # Create combined collection targets
    foreach(dt IN LISTS GEMM_ROWCOL_DATATYPE)
        foreach(l IN LISTS GEMM_ROWCOL_LAYOUT)
            add_custom_target(benchmark_gemm_rowcol_${dt}_${l})
        endforeach()
    endforeach()

    # Create pipeline-specific collection targets
    set(GEMM_ROWCOL_PIPELINES "compv3")
    foreach(pipeline IN LISTS GEMM_ROWCOL_PIPELINES)
        add_custom_target(benchmark_gemm_rowcol_${pipeline}_pipeline)
    endforeach()

    # Build individual targets for each datatype/layout combination
    foreach(dt IN LISTS GEMM_ROWCOL_DATATYPE)
        foreach(l IN LISTS GEMM_ROWCOL_LAYOUT)
            build_individual_gemm_rowcol_targets(${dt} ${l})
        endforeach()
    endforeach()
endif()
