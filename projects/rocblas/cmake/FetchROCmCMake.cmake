# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Attempt to find ROCmCMakeBuildTools in the system, if not found, fetch from source
# To install ROCmCMakeBuildTools from source, see: https://github.com/ROCm/rocm-cmake

find_package(ROCmCMakeBuildTools QUIET CONFIG)

if(NOT ROCmCMakeBuildTools_FOUND)
  message(STATUS "ROCmCMakeBuildTools not found. Fetching from source...")
  include(FetchContent)
  FetchContent_Declare(
    rocm-cmake
    GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
    GIT_TAG develop
    GIT_SHALLOW TRUE)

  FetchContent_GetProperties(rocm-cmake)
  if(NOT rocm-cmake_POPULATED)
    FetchContent_Populate(rocm-cmake)
    list(APPEND CMAKE_MODULE_PATH ${rocm-cmake_SOURCE_DIR}/share/rocm/cmake)
    add_subdirectory(${rocm-cmake_SOURCE_DIR} ${rocm-cmake_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
else()
  # ROCmCMakeBuildTools found - ensure its module path is available
  if(ROCmCMakeBuildTools_DIR)
    get_filename_component(ROCM_CMAKE_MODULE_DIR "${ROCmCMakeBuildTools_DIR}/../../../share/rocm/cmake" ABSOLUTE)
    if(EXISTS "${ROCM_CMAKE_MODULE_DIR}")
      list(APPEND CMAKE_MODULE_PATH "${ROCM_CMAKE_MODULE_DIR}")
    endif()
  endif()
endif()

include(ROCMSetupVersion)
include(ROCMClients)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
