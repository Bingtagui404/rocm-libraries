# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Core Tensile library source files (only files that exist in this version)
target_sources(tensile-host
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/AMDGPU.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticUnitTypes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionProblem.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionSolution.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Debug.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/EmbeddedLibrary.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelArguments.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelLanguageTypes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MLFeatures.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PerformanceMetricTypes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ScalarValueTypes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TensorDescriptor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TensorOps.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/UserDrivenTuningParser.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp"
)

# Add backend-specific subdirectories
add_subdirectory(hip)

if(TENSILE_ENABLE_LLVM)
    add_subdirectory(llvm)
endif()

if(TENSILE_ENABLE_MSGPACK)
    add_subdirectory(msgpack)
endif()
