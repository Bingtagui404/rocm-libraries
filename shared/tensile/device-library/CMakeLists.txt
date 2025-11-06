# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

list(JOIN GPU_TARGETS "$<SEMICOLON>" TENSILE_GPU_TARGETS_SEMI_ESCAPED)
set(TENSILE_VERBOSITY "" CACHE STRING "Verbosity level for the TensileCreateLibrary command")
set(TENSILE_JOBS "" CACHE STRING "Number of jobs to run in parallel")
set(TENSILE_CODE_OBJECT_VERSION "V4" CACHE STRING "Path to directory containing logic files")
set(TENSILE_LOGIC_DIR
    "${tensile_SOURCE_DIR}/../../projects/rocblas/library/src/blas3/Tensile/Logic/asm_full"
    CACHE STRING "Path to directory containing logic files"
)
set(ROCBLAS_TENSILE_LIBPATH "${PROJECT_BINARY_DIR}/Tensile"
    CACHE STRING "Path to output the device gemm libraries."
)
file(MAKE_DIRECTORY "${ROCBLAS_TENSILE_LIBPATH}/library")

option(TENSILE_MERGE_FILES "Merge kernel and solution files" ON)
option(TENSILE_SEPARATE_ARCHITECTURES "Use GPU architecture-specific files" ON)
option(TENSILE_LAZY_LIBRARY_LOADING "Load kernels on demand" ON)
option(TENSILE_SHORT_FILENAMES "Use short file names" OFF)

if(TENSILE_MERGE_FILES)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--merge-files")
endif()
if(TENSILE_SEPARATE_ARCHITECTURES)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--separate-architectures")
endif()
if(TENSILE_LAZY_LIBRARY_LOADING)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--lazy-library-loading")
endif()
if(NOT TENSILE_SHORT_FILENAMES)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--no-short-file-names")
endif()

set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS}
                       "--architecture=${TENSILE_GPU_TARGETS_SEMI_ESCAPED}"
)
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--cxx-compiler=${CMAKE_CXX_COMPILER}")
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--code-object-version=${TENSILE_CODE_OBJECT_VERSION}")
if(TENSILE_VERBOSITY OR TENSILE_VERBOSITY STREQUAL "0")
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--verbose=${TENSILE_VERBOSITY}")
endif()
if(TENSILE_JOBS)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--jobs=${TENSILE_JOBS}")
endif()

set(TENSILE_CREATE_LIBRARY_COMMAND
    ${TENSILE_PYTHON_COMMAND} "${PROJECT_SOURCE_DIR}/Tensile/bin/TensileCreateLibrary"
    ${TENSILE_BUILD_OPTS}
    ${TENSILE_LOGIC_DIR}
    "${ROCBLAS_TENSILE_LIBPATH}"
    HIP
)

set(output_stamp "${CMAKE_CURRENT_BINARY_DIR}/Tensile.stamp")
add_custom_command(
    OUTPUT "${output_stamp}"
    COMMENT "Building device libraries to ${ROCBLAS_TENSILE_LIBPATH} ..."
    COMMAND ${TENSILE_CREATE_LIBRARY_COMMAND}
    COMMAND ${CMAKE_COMMAND} -E touch "${output_stamp}"
    VERBATIM
    USES_TERMINAL
)

block(SCOPE_FOR VARIABLES)
list(JOIN TENSILE_CREATE_LIBRARY_COMMAND " " FORMATTED_TCL)
message(VERBOSE "Device lib build command: ${FORMATTED_TCL}")
endblock()

add_custom_target(tensile-device-libraries ALL DEPENDS "${output_stamp}")

add_dependencies(tensile-host tensile-device-libraries)
