# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# HIP backend implementation files
target_sources(tensile-host
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/HipSolutionAdapter.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/HipHardware.cpp"
)
