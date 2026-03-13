# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(test_offset_compute test_offset_compute.cpp)
target_link_libraries(test_offset_compute PRIVATE utility hip::host hip::device)
target_compile_options(test_offset_compute PRIVATE -Wno-global-constructors -Wno-shadow)

add_executable(test_offset_load test_offset_load.cpp)
target_link_libraries(test_offset_load PRIVATE utility hip::host hip::device)
target_compile_options(test_offset_load PRIVATE -Wno-global-constructors -Wno-shadow -Wno-float-equal)
