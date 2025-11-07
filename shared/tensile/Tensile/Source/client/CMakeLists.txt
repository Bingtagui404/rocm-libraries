# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(tensile-client-lib
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/source/BenchmarkTimer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/ClientProblemFactory.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/ConvolutionProblem.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/CSVStackFile.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/DataInitialization.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/HardwareMonitor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/HardwareMonitorListener.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/LibraryUpdateReporter.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/MetaRunListener.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/PerformanceReporter.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/ProgressListener.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/Reference.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/ReferenceValidator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/ResultFileReporter.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/ResultReporter.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/SolutionIterator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/source/TimingEvents.cpp"
)

# Tensile client executable main file
target_sources(tensile-client
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)
