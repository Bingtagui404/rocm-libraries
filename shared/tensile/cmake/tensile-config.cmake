# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# tensile-config.cmake - Conventional CMake package configuration file
# This file allows external projects to find and use the Tensile library

@PACKAGE_INIT@

# Dependencies required by Tensile
include(CMakeFindDependencyMacro)
find_dependency(hip REQUIRED)

# Optional: msgpack dependency (if TENSILE_MSGPACK was enabled)
# find_dependency(msgpack)

# Include the targets file
if(NOT TARGET roc::tensile-host)
  include("${CMAKE_CURRENT_LIST_DIR}/tensile-targets.cmake")
endif()

# Provide legacy TensileHost target alias for backward compatibility
if(NOT TARGET TensileHost AND TARGET roc::tensile-host)
  add_library(TensileHost ALIAS roc::tensile-host)
endif()

check_required_components(tensile)
