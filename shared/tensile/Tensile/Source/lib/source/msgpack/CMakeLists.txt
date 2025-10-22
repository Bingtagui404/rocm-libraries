# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# MessagePack backend implementation files
target_sources(tensile-host
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/MessagePack.cpp"
)
