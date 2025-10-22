# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# LLVM YAML backend implementation files
target_sources(tensile-host
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/YAML.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Loading.cpp"
)
