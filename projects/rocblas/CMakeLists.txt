# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

# CMP0135: Set the timestamps of extracted contents to the time of extraction.
# This policy ensures that extracted archive contents have consistent timestamps,
# which is important for reproducible builds and proper dependency tracking.
# Without this, timestamps might reflect the original archive creation time,
# potentially causing issues with build systems that rely on file modification times.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(rocblas_shim)

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

project(
    rocblas
    LANGUAGES C CXX)

include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(GNUInstallDirs)

include(rocblas_target_configure_sanitizers)
include(rocblas_supported_architectures)
include(query_git_hash)
include(FetchROCmCMake)

#query_git_hash(rocblas "${CMAKE_CURRENT_SOURCE_DIR}")
rocm_setup_version(VERSION "5.2.0")

if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING OR ROCBLAS_ENABLE_BENCHMARKS OR ROCBLAS_ENABLE_SAMPLES)
    if(NOT DEFINED ROCBLAS_ENABLE_CLIENT OR NOT ROCBLAS_ENABLE_CLIENT)
        set(ROCBLAS_ENABLE_CLIENT ON CACHE BOOL "Build rocBLAS clients." FORCE)
    endif()
endif()

set(HIP_RUNTIME_MINIMUM "4.5.0" CACHE STRING "Minimum HIP runtime version required")

set(GPU_TARGETS "" CACHE STRING "AMD GFX targets to cross-compile")

if(NOT GPU_TARGETS OR GPU_TARGETS STREQUAL "all")
    rocblas_get_base_architectures(GPU_TARGETS)
else()
    rocblas_validate_gpu_targets("${GPU_TARGETS}")
endif()

message(STATUS "Building for GPU targets: ${GPU_TARGETS}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if(WIN32)
    set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix." FORCE)
  else()
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
  endif()
endif()

option(ROCBLAS_ENABLE_SUPERBUILD "Build as ROCm BLAS subproject." OFF)

option(CMAKE_EXPORT_COMPILE_COMMANDS "Export compile_commands.json for clang tooling support" ON)

option(ROCBLAS_ENABLE_CLIENT "Build rocBLAS clients." OFF)
option(ROCBLAS_ENABLE_TENSILE "Build Tensile GEMM device libraries." ON)
option(ROCBLAS_ENABLE_HIPBLASLT "Build rocBLAS host library with hipBLASLt backend." OFF)
set(ROCBLAS_TENSILE_LIBPATH "${PROJECT_BINARY_DIR}/Tensile" CACHE STRING "Path to output the device gemm libraries.")

if(ROCBLAS_ENABLE_CLIENT)
  option(ROCBLAS_ENABLE_BENCHMARKS "Build benchmark client." ON)
  option(ROCBLAS_ENABLE_SAMPLES "Build client samples." ON)
  option(ROCBLAS_ENABLE_FORTRAN "Build Fortran clients." ON)
  cmake_dependent_option(ROCBLAS_ENABLE_ROCM_SMI "Require rocm_smi." ON "NOT WIN32" OFF)
endif()

if(ROCBLAS_ENABLE_TENSILE)
  if(WIN32)
    set(ROCBLAS_TENSILE_INSTALL_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}rocblas/bin" CACHE PATH "Path to install tree tensile library")
  else()
    set(ROCBLAS_TENSILE_INSTALL_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/rocblas/library" CACHE PATH "Path to tensile library")
  endif()
  set(ROCBLAS_TENSILE_SUBDIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/tensile" CACHE STRING "Path to Tensile subdirectory")
endif()
set(ROCBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

option(ROCBLAS_ENABLE_HOST "Build rocBLAS host library." ON)
option(ROCBLAS_BUILD_SHARED_LIBS "Build the rocBLAS shared or static library." ON)
set(BUILD_SHARED_LIBS ${ROCBLAS_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries")
option(ROCBLAS_BUILD_TESTING "Build test client; master switch." OFF)
option(ROCBLAS_BUILD_COVERAGE "Build tests with coverage enabled." OFF)
set(ROCBLAS_COVERAGE_GTEST_FILTER "*quick*:*pre_checkin*:*ILP64*-*known_bug*" CACHE STRING "GTest filter for coverage tests.")

# I don't know that we can build with BLIS OFF
option(ROCBLAS_ENABLE_BLIS "Enable BLIS support." OFF)
option(ROCBLAS_ENABLE_MARKER "Enable rocTracer marker support." ON)
option(ROCBLAS_ENABLE_OPENMP "Enable OpenMP for parallel reference implementations in clients." ON)
option(ROCBLAS_ENABLE_THREADS "Enable threading support in clients." ON)
option(ROCBLAS_ENABLE_ASAN "Build with address sanitizer enabled." OFF)
option(ROCBLAS_ENABLE_THEROCK "Use find_package to pull in 1st party deps." OFF)

option(CMAKE_VERBOSE_MAKEFILE "Enable verbose output from Makefile builds." ON)

if(ROCBLAS_ENABLE_FORTRAN AND ROCBLAS_ENABLE_ASAN)
  message(FATAL_ERROR "Cannot enable fortran clients and address sanitizers.")
endif()

if(ROCBLAS_BUILD_COVERAGE OR BUILD_CODE_COVERAGE)
  if(ROCBLAS_ENABLE_FORTRAN)
    message(FATAL_ERROR "Cannot enable fortran clients and code coverage.")
  endif()
endif()

if(ROCBLAS_ENABLE_FORTRAN)
    enable_language(Fortran)
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard version")
option(CMAKE_CXX_STANDARD_REQUIRED "" ON)
option(CMAKE_CXX_EXTENSIONS "Enable c++ extenstions (default: OFF)" OFF)


set(CMAKE_SKIP_BUILD_RPATH FALSE CACHE BOOL "Skip build RPATH")
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE CACHE BOOL "Build with install RPATH")
# need to account for openmp_LIB_DIR - not sure what cases that would be different from lib
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../llvm/lib" CACHE STRING "Install RPATH")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE CACHE BOOL "Use link path for RPATH")

mark_as_advanced(CMAKE_SKIP_BUILD_RPATH)
mark_as_advanced(CMAKE_BUILD_WITH_INSTALL_RPATH)
mark_as_advanced(CMAKE_INSTALL_RPATH)
mark_as_advanced(CMAKE_INSTALL_RPATH_USE_LINK_PATH)

# Interface library for common warning suppression flags
add_library(rocblas-warnings INTERFACE)
add_library(rocblas::warnings ALIAS rocblas-warnings)
target_compile_options(rocblas-warnings INTERFACE -Wno-unused-result -Wno-c++17-extensions)

find_package(hip REQUIRED)

if(ROCBLAS_ENABLE_HIPBLASLT)
    if(NOT ROCBLAS_ENABLE_SUPERBUILD)
        find_package(hipblaslt 1.0.0 REQUIRED)
    endif()
    rocm_package_add_dependencies(DEPENDS "hipblaslt >= 1.0.0")
endif()

if(ROCBLAS_ENABLE_ROCM_SMI)
  find_package(rocm_smi REQUIRED)
endif()

if(ROCBLAS_ENABLE_HOST)
    if(ROCBLAS_ENABLE_TENSILE)
        if(NOT ROCBLAS_ENABLE_SUPERBUILD)
            if(NOT EXISTS "${ROCBLAS_TENSILE_SUBDIR_PATH}")
                message(FATAL_ERROR
                    "Tensile subdirectory not found at: ${ROCBLAS_TENSILE_SUBDIR_PATH}\n"
                    "Please ensure the Tensile repository is cloned at this location, or\n"
                    "set ROCBLAS_TENSILE_SUBDIR_PATH to the correct path.\n"
                    "Example: -DROCBLAS_TENSILE_SUBDIR_PATH=/path/to/tensile")
            endif()
            message(STATUS "Adding Tensile from subdirectory: ${ROCBLAS_TENSILE_SUBDIR_PATH}")
            # Force Tensile to build as static library
            block(SCOPE_FOR VARIABLES)
                set(BUILD_SHARED_LIBS OFF)
                set(TENSILE_BUILD_SHARED_LIBS OFF CACHE BOOL "Build Tensile as static library" FORCE)
                add_subdirectory("${ROCBLAS_TENSILE_SUBDIR_PATH}" "${CMAKE_CURRENT_BINARY_DIR}/_tensile" EXCLUDE_FROM_ALL)
            endblock()
        elseif(ROCBLAS_ENABLE_THEROCK)
            find_package(tensile REQUIRED)
        endif()
    endif()

    add_subdirectory(library)

    if(ROCBLAS_ENABLE_MARKER)
        include(FindPackageHandleStandardArgs)

        find_path(ROCTRACER_INCLUDE_DIR "roctracer/roctx.h")
        find_library(ROCTX_LIBRARIES roctx64)

        find_package_handle_standard_args(ROCTRACER
            REQUIRED_VARS ROCTRACER_INCLUDE_DIR rocTracer
            FAIL_MESSAGE "Could not find ROCTracer library or headers"
        )
        target_link_libraries(rocblas PRIVATE ${ROCTX_LIBRARIES})
    else()
        target_compile_definitions(rocblas PRIVATE DISABLE_ROCTX)
    endif()
endif()

# Set up client packaging components when any client is enabled
# This ensures dependency variables and parent components are registered
# before child components (tests, benchmarks, samples) reference them
if(ROCBLAS_ENABLE_CLIENT OR ROCBLAS_BUILD_TESTING OR BUILD_TESTING OR ROCBLAS_ENABLE_BENCHMARKS OR ROCBLAS_ENABLE_SAMPLES)
  add_subdirectory(clients)
  # Set up OS-specific dependencies for client components
  if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
  endif()

  set(OPENMP_RPM "libgomp")
  set(OPENMP_DEB "libomp-dev")
  set(GFORTRAN_RPM "libgfortran4")
  set(GFORTRAN_DEB "libgfortran4")

  if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel")
    if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
      set(GFORTRAN_RPM "libgfortran")
    endif()
  elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION
                                          VERSION_GREATER_EQUAL "20.04")
    set(GFORTRAN_DEB "libgfortran5")
  elseif(CLIENTS_OS STREQUAL "sles")
    set(OPENMP_RPM "libgomp1")
  elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
    set(GFORTRAN_RPM "gfortran")
  endif()

  # Register parent components BEFORE child components reference them
  # Set up core library packaging components
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(clients-common)

  # Register child components with proper dependencies
  if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
    rocm_package_setup_client_component(
      tests
      DEPENDS
        COMPONENT clients-common
        DEB "${OPENMP_DEB}"
        RPM "${OPENMP_RPM}"
    )
  endif()

  if(ROCBLAS_ENABLE_BENCHMARKS)
    rocm_package_setup_client_component(
      benchmarks
      DEPENDS
        COMPONENT clients-common
        DEB "${OPENMP_DEB}"
        RPM "${OPENMP_RPM}"
    )
  endif()

  if(ROCBLAS_ENABLE_SAMPLES)
    rocm_package_setup_client_component(
      samples
      DEPENDS
        COMPONENT clients-common
        DEB "${OPENMP_DEB}"
        RPM "${OPENMP_RPM}"
    )
  endif()

  # Add Fortran dependencies if Fortran clients are enabled
  if(ROCBLAS_ENABLE_FORTRAN)
    rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_DEB}")
    rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_DEB}")
  endif()
endif()

if(ROCBLAS_ENABLE_HOST)
    rocm_install_targets(
      TARGETS rocblas
      INCLUDE
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}"
    )
    if(ROCBLAS_ENABLE_TENSILE)
      rocm_install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Tensile/library/"
        DESTINATION "${ROCBLAS_TENSILE_INSTALL_DIR}"
        COMPONENT runtime
      )
    endif()
    if(ROCBLAS_ENABLE_FORTRAN)
      rocm_install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/library/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas"
        COMPONENT devel
        FILES_MATCHING
          PATTERN "*.h"
          PATTERN "*.f90")
    else()
      rocm_install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/library/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas"
        COMPONENT devel
        FILES_MATCHING
          PATTERN "*.h")
    endif()
    rocm_export_targets(
      TARGETS roc::rocblas
      DEPENDS PACKAGE hip
      NAMESPACE roc::
    )
    set(rocblas_SOVERSION 5.2)
    rocm_set_soversion(rocblas ${rocblas_SOVERSION})
endif()

set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "\${CPACK_PACKAGING_INSTALL_PREFIX}")

# Work around code object stripping failure if using /usr/bin/strip
find_program(LLVM_STRIP 
    NAMES llvm-strip
    PATH_SUFFIXES lib/llvm/bin llvm/bin bin
)
if(LLVM_STRIP)
    set(CPACK_RPM_SPEC_MORE_DEFINE "%define __strip ${LLVM_STRIP}")
else()
    message(WARNING "llvm-strip not found, using default strip tool. This may cause stripping to fail during packaging.")
endif()

# cmake-format: off

rocm_install(
  FILES
  "${CMAKE_CURRENT_BINARY_DIR}/library/include/rocblas/internal/rocblas-export.h"
  "${CMAKE_CURRENT_BINARY_DIR}/library/include/rocblas/internal/rocblas-version.h"
  DESTINATION
  "${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal"
  COMPONENT devel)

# For one off inclusion of files required in the install tree
rocm_install(
  FILES "${PROJECT_SOURCE_DIR}/library/src/include/rocblas_device_malloc.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas/internal"
  COMPONENT devel)

# cmake-format: on
# Stage and install common client files to clients-common component
if(ROCBLAS_ENABLE_CLIENT OR ROCBLAS_BUILD_TESTING OR BUILD_TESTING OR ROCBLAS_ENABLE_BENCHMARKS OR ROCBLAS_ENABLE_SAMPLES)

  if(ROCBLAS_ENABLE_BENCHMARKS)
    rocm_install(TARGETS rocblas-bench COMPONENT benchmarks DESTINATION
                 ${CMAKE_INSTALL_BINDIR})

    if(ROCBLAS_ENABLE_TENSILE)
      rocm_install(TARGETS rocblas-gemm-tune COMPONENT benchmarks DESTINATION
                   ${CMAKE_INSTALL_BINDIR})
    endif()
  endif()

  if(ROCBLAS_BUILD_TESTING OR BUILD_TESTING)
    rocm_install(TARGETS rocblas-test COMPONENT tests DESTINATION
                 ${CMAKE_INSTALL_BINDIR})

    rocm_install(
      FILES "${PROJECT_BINARY_DIR}/clients/staging/rocblas_gtest.data"
      DESTINATION "${CMAKE_INSTALL_BINDIR}"
      COMPONENT tests
    )

    if(TARGET rocblas-test-files)
      get_target_property(TEST_FILES rocblas-test-files ROCBLAS_INSTALL_FILES)
      get_target_property(TEST_PROGRAMS rocblas-test-files ROCBLAS_INSTALL_PROGRAMS)

      if(TEST_FILES)
        rocm_install(
          FILES ${TEST_FILES}
          DESTINATION "${CMAKE_INSTALL_BINDIR}"
          COMPONENT tests
        )
      endif()

      if(TEST_PROGRAMS)
        rocm_install(
          PROGRAMS ${TEST_PROGRAMS}
          DESTINATION "${CMAKE_INSTALL_BINDIR}"
          COMPONENT tests
        )
      endif()
    endif()
  endif()

  if(ROCBLAS_ENABLE_SAMPLES)
    rocm_install(
      TARGETS
      rocblas-example-c-dgeam
      rocblas-example-gemv-graph-capture
      rocblas-example-hip-complex-her2
      rocblas-example-scal-multiple-strided-batch
      rocblas-example-scal-template
      rocblas-example-sgemm
      rocblas-example-sgemm-multiple-strided-batch
      rocblas-example-sgemm-strided-batched
      rocblas-example-solver
      rocblas-example-sscal
      # `rocblas-example-header-check` is not installed because it is used to check
      # a C++ version guard exists, and thereby serves as a compile time check.
      COMPONENT
      samples
      DESTINATION
      ${CMAKE_INSTALL_BINDIR})

    if(ROCBLAS_ENABLE_TENSILE)
      rocm_install(TARGETS rocblas-example-user-driven-tuning COMPONENT samples
                   DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    if(OpenMP_CXX_FOUND)
      rocm_install(TARGETS rocblas-example-openmp COMPONENT samples
                   DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    if(ROCBLAS_ENABLE_FORTRAN)
      rocm_install(
        TARGETS
        rocblas-example-fortran-axpy
        rocblas-example-fortran-gemv
        rocblas-example-fortran-scal
        COMPONENT
        samples
        DESTINATION
        ${CMAKE_INSTALL_BINDIR})
    endif()
  endif()

  rocm_install(
    FILES
      "${PROJECT_BINARY_DIR}/clients/staging/rocblas_common.yaml"
      "${PROJECT_BINARY_DIR}/clients/staging/rocblas_template.yaml"
      "${PROJECT_BINARY_DIR}/clients/staging/rocblas_general.yaml"
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT clients-common
  )

  rocm_install(
    PROGRAMS
      "${PROJECT_BINARY_DIR}/clients/staging/rocblas_gentest.py"
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT clients-common
  )

  rocm_install(
    FILES
      "${PROJECT_BINARY_DIR}/clients/staging/rocblas_clients_readme.txt"
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT clients
  )
endif()

if( HIP_RUNTIME MATCHES ".*cuda.*" )
  rocm_package_add_dependencies(DEPENDS "hip-nvcc >= 3.5.0")
else()
    if(ROCBLAS_ENABLE_ASAN)
      set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan")
    else()
      set(DEPENDS_HIP_RUNTIME "hip-runtime-amd")
    endif()

    rocm_package_add_dependencies(SHARED_DEPENDS "${DEPENDS_HIP_RUNTIME} >= ${HIP_RUNTIME_MINIMUM}")
    rocm_package_add_deb_dependencies(STATIC_DEPENDS "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
    rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")
endif()

rocm_create_package(
  NAME
  rocblas
  DESCRIPTION "AMD library for BLAS in ROCm. Implemented in HIP and optimized for AMD GPUs"
  MAINTAINER "rocBLAS Maintainer <rocblas-maintainer@amd.com>"
  LDCONFIG
  LDCONFIG_DIR
  ${ROCBLAS_CONFIG_DIR})

if(ROCBLAS_BUILD_COVERAGE OR BUILD_CODE_COVERAGE)
  message(STATUS "Configuring code coverage target")
  target_compile_options(rocblas PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(rocblas PRIVATE -fprofile-instr-generate)

  set(coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")
  set(coverage_test_arguments
    "--precompile=rocblas-test-precompile.db"
  )
  if(ROCBLAS_COVERAGE_GTEST_FILTER)
    list(APPEND coverage_test_arguments "--gtest_filter=${ROCBLAS_COVERAGE_GTEST_FILTER}")
  endif()

  add_custom_target(
    code_cov_tests
    DEPENDS rocblas-test
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${coverage_dir}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${coverage_dir}/profraw"
    COMMAND
      ${CMAKE_COMMAND} -E env
      LLVM_PROFILE_FILE="${coverage_dir}/profraw/rocblas-coverage_%p.profraw"
      $<TARGET_FILE:rocblas-test> ${coverage_test_arguments}
    COMMENT "Running coverage with test command: $<TARGET_FILE:rocblas-test> ${coverage_test_arguments}")

  find_program(LLVM_PROFDATA llvm-profdata HINTS llvm/bin REQUIRED)
  find_program(LLVM_COV llvm-cov REQUIRED HINTS llvm/bin)

  add_custom_target(
    coverage
    DEPENDS code_cov_tests
    COMMAND
      ${LLVM_PROFDATA} merge -sparse
      "${coverage_dir}/profraw/rocblas-coverage_*.profraw" -o
      "${coverage_dir}/rocblas.profdata"
    COMMAND ${LLVM_COV} report -object $<TARGET_FILE:rocblas>
            -instr-profile="${coverage_dir}/rocblas.profdata"
    COMMAND
      ${LLVM_COV} show -object $<TARGET_FILE:rocblas>
      -instr-profile="${coverage_dir}/rocblas.profdata" -format=html
      -output-dir="${coverage_dir}"
    COMMENT "Generating coverage... report can be found in ${coverage_dir}/index.html")

endif()
