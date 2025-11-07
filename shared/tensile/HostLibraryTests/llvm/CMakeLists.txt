# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(tensile-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticUnitPredicate_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CUEfficiencyPredicate_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DeterministicModePredicate_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelLanguagePredicate_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LLVMYAMLContraction_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LibraryPerformance_test.cpp"
)
