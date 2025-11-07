# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # We are being run standalone
    cmake_minimum_required(VERSION 3.25)
    project(tensile-tests)

    # Pretend this is a project, but without a literal project() call.
    # (This will prevent dev warnings about missing project())
    set(_STANDALONE TRUE)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
    
    # Optionally set policies that would otherwise warn
    if(COMMAND cmake_policy)
        cmake_policy(SET CMP0003 NEW)
        cmake_policy(SET CMP0012 NEW)
    endif()

    # Bring in the parent (or one level up)
    set(TENSILE_ENABLE_DEVICE OFF)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_BINARY_DIR}/parent_build")
endif()

enable_testing()
find_package(hip REQUIRED)
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_subdirectory(configs)

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

foreach(FILE ${SOLUTION_LIBRARY_FILES})
    if(EXISTS "${FILE}.gz")
        execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
    endif()
    if(EXISTS ${FILE})
        file(COPY ${FILE} DESTINATION "${TEST_DATA_DIR}")
    endif()
endforeach(FILE)

add_subdirectory(testlib)

add_executable(tensile-tests)

target_sources(tensile-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CachingLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionProblem_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionSelectionLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionFitness_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MultipleSolutionsPerSize_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/EmbeddedData_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelArguments_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Predicates_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ProjectedPerformance_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DecisionTree_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TensorDescriptor_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestData_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Utils_test.cpp"
)

if(TENSILE_ENABLE_LLVM)
    target_sources(tensile-tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/ContractionLibraryLoading_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/ContractionFitness_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/MultipleSolutionsPerSize_test.cpp"
    )
    add_subdirectory(llvm)
endif()

if(TENSILE_ENABLE_CLIENT)
    target_sources(tensile-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/client/DataInitialization_test.cpp")
endif()

if(TENSILE_ENABLE_HIP)
    add_subdirectory(hip)
endif()

target_link_libraries(tensile-tests
    PUBLIC
        roc::tensile-host
        Boost::filesystem
    PRIVATE
        tensile::tensile-test-lib
        GTest::gtest
        hip::device
)

if(TENSILE_ENABLE_LLVM)
    find_library(LLVMObjectYAML_LIBRARY NAMES LLVMObjectYAML PATHS ${LLVM_LIBRARY_DIR})
    if(LLVMObjectYAML_LIBRARY)
        target_link_libraries(tensile-tests PUBLIC LLVMObjectYAML)
    else()
        target_link_libraries(tensile-tests PUBLIC LLVM)
    endif()
    target_include_directories(tensile-tests PUBLIC ${LLVM_INCLUDE_DIRS})
endif()

if(TENSILE_ENABLE_HIP)
    target_link_libraries(tensile-tests PRIVATE "-Xlinker --whole-archive")
    target_link_libraries(tensile-tests PUBLIC ${HIP_TEST_LIBRARIES})
    target_link_libraries(tensile-tests PRIVATE "-Xlinker --no-whole-archive")
endif()

if(TENSILE_ENABLE_OPENMP)
    find_package(OpenMP CONFIG)
    if(OpenMP_FOUND)
        add_library(rocm-omp-shim INTERFACE)
        add_library(OpenMP::OpenMP_CXX ALIAS rocm-omp-shim)
        set_target_properties(rocm-omp-shim PROPERTIES INTERFACE_COMPILE_OPTIONS -fopenmp INTERFACE_LINK_LIBRARIES OpenMP::omp)
    else()
        message(STATUS "Did not find rocm-omp - falling back to gomp")
        find_package(OpenMP REQUIRED)
    endif()
    target_link_libraries(tensile-tests PRIVATE OpenMP::OpenMP_CXX)
endif()

gtest_discover_tests(tensile-tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} TIMEOUT 60)

if(TENSILE_ENABLE_COVERAGE OR BUILD_CODE_COVERAGE)
    message(STATUS "Code coverage for tensile-host is ENABLED.")

    target_compile_options(tensile-host PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(tensile-host PUBLIC -fprofile-instr-generate)

    set(coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")

    add_custom_target(
        code_cov_tests
        DEPENDS tensile-tests
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${coverage_dir}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${coverage_dir}/profraw"
        COMMAND
            ${CMAKE_COMMAND} -E env
            "LLVM_PROFILE_FILE=${coverage_dir}/profraw/tensile-host-coverage_%m.profraw"
            GTEST_LISTENER=NO_PASS_LINE_IN_LOG $<TARGET_FILE:tensile-tests>
            "--gtest_filter=-*Extended*:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/504:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/505:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/506:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/507:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/508:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/509"
    )

    find_program(LLVM_PROFDATA llvm-profdata PATH_SUFFIXES llvm/bin REQUIRED)
    find_program(LLVM_COV llvm-cov REQUIRED PATH_SUFFIXES llvm/bin)

    add_custom_target(
        coverage
        DEPENDS code_cov_tests
        COMMAND
            ${LLVM_PROFDATA} merge -sparse "${coverage_dir}/profraw/tensile-host-coverage_*.profraw"
            -o "${coverage_dir}/tensile-host.profdata"
        COMMAND
            ${LLVM_COV} report -object $<TARGET_FILE:tensile-host>
            -instr-profile="${coverage_dir}/tensile-host.profdata"
        COMMAND
            ${LLVM_COV} show -object $<TARGET_FILE:tensile-host>
            -instr-profile="${coverage_dir}/tensile-host.profdata" -format=html
            -output-dir="${coverage_dir}"
        COMMENT
            "Generating tensile-host coverage... HTML report in ${coverage_dir}/index.html"
    )
endif()
