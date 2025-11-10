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
option(TENSILE_GENERATE_PACKAGE "Generate Tensile package" OFF)
option(TENSILE_KEEP_BUILD_TMP "Keep temporary build files" OFF)
option(TENSILE_NO_ENUMERATE "Disable enumerating GPU targets" OFF)

set(TENSILE_LIBRARY_FORMAT "msgpack" CACHE STRING "Tensile library format (msgpack or yaml)")
set(TENSILE_EMBED_LIBRARY "" CACHE STRING "Library name to embed in source")
set(TENSILE_EMBED_KEY "" CACHE STRING "Key for embedded library")

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
if(TENSILE_GENERATE_PACKAGE)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--package-library")
endif()
if(TENSILE_KEEP_BUILD_TMP)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--keep-build-tmp")
endif()
if(TENSILE_NO_ENUMERATE)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--no-enumerate")
endif()

set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS}
                       "--architecture=${TENSILE_GPU_TARGETS_SEMI_ESCAPED}"
)
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--cxx-compiler=${CMAKE_CXX_COMPILER}")
set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--code-object-version=${TENSILE_CODE_OBJECT_VERSION}")

if(TENSILE_LIBRARY_FORMAT)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--library-format=${TENSILE_LIBRARY_FORMAT}")
endif()
if(TENSILE_EMBED_LIBRARY)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--embed-library=${TENSILE_EMBED_LIBRARY}")
endif()
if(TENSILE_EMBED_KEY)
    set(TENSILE_BUILD_OPTS ${TENSILE_BUILD_OPTS} "--embed-library-key=${TENSILE_EMBED_KEY}")
endif()
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

add_custom_target(tensile-gemm-libraries ALL DEPENDS "${output_stamp}")

add_dependencies(tensile-host tensile-gemm-libraries)
