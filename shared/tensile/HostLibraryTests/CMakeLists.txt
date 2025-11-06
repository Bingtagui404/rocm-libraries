# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # We are being run standalone
    cmake_minimum_required(VERSION 3.25)

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
    if(EXISTS ${FILE})
        message(WARNING "Overwriting ${FILE} from ${FILE}.gz")
    endif()
    execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
    file(COPY ${FILE} DESTINATION "${TEST_DATA_DIR}")
endforeach(FILE)

add_subdirectory(testlib)

add_executable(TensileTests)

target_sources(TensileTests
    PRIVATE
        test.cpp
        CachingLibrary_test.cpp
        ContractionProblem_test.cpp
        ContractionSelectionLibrary_test.cpp
        ContractionFitness_test.cpp
        MultipleSolutionsPerSize_test.cpp
        DataTypes_test.cpp
        EmbeddedData_test.cpp
        KernelArguments_test.cpp
        Predicates_test.cpp
        ProjectedPerformance_test.cpp
        DecisionTree_test.cpp
        TensorDescriptor_test.cpp
        TestData_test.cpp
        Utils_test.cpp
)

if(TENSILE_ENABLE_LLVM)
    target_sources(TensileTests
        PRIVATE
            ContractionLibraryLoading_test.cpp
            ContractionFitness_test.cpp
            MultipleSolutionsPerSize_test.cpp
            llvm/ArithmeticUnitPredicate_test.cpp
            llvm/CUEfficiencyPredicate_test.cpp
            llvm/DeterministicModePredicate_test.cpp
            llvm/KernelLanguagePredicate_test.cpp
            llvm/LLVMYAMLContraction_test.cpp
            llvm/LibraryPerformance_test.cpp
    )
endif()

if(TENSILE_ENABLE_CLIENT)
    target_sources(TensileTests PRIVATE client/DataInitialization_test.cpp)
endif()

if(TENSILE_ENABLE_HIP)
    add_subdirectory(hip)
endif()

target_link_libraries(TensileTests
    PUBLIC
        roc::tensile-host
        Boost::filesystem
    PRIVATE
        TensileTestLib
        GTest::gtest
        hip::device
)

if(TENSILE_ENABLE_LLVM)
    find_library(LLVMObjectYAML_LIBRARY NAMES LLVMObjectYAML PATHS ${LLVM_LIBRARY_DIR})
    if(LLVMObjectYAML_LIBRARY)
        target_link_libraries(TensileTests PUBLIC LLVMObjectYAML)
    else()
        target_link_libraries(TensileTests PUBLIC LLVM)
    endif()
    target_include_directories(TensileTests PUBLIC ${LLVM_INCLUDE_DIRS})
endif()

if(TENSILE_ENABLE_HIP)
    target_link_libraries(TensileTests PRIVATE "-Xlinker --whole-archive")
    target_link_libraries(TensileTests PUBLIC ${HIP_TEST_LIBRARIES})
    target_link_libraries(TensileTests PRIVATE "-Xlinker --no-whole-archive")

    foreach(arch IN LISTS GPU_TARGETS)
        target_link_libraries(TensileTests PRIVATE "--offload-arch=${arch}")
    endforeach(arch)
endif()

if(TENSILE_ENABLE_OPENMP)
    target_link_libraries(TensileTests PRIVATE OpenMP::OpenMP_CXX)
endif()

gtest_discover_tests(TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} TIMEOUT 60)

if(TENSILE_ENABLE_COVERAGE OR BUILD_CODE_COVERAGE)
    message(STATUS "Code coverage for tensile-host is ENABLED.")

    target_compile_options(tensile-host PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(tensile-host PUBLIC -fprofile-instr-generate)

    set(coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")

    add_custom_target(
        code_cov_tests
        DEPENDS TensileTests
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${coverage_dir}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${coverage_dir}/profraw"
        COMMAND
            ${CMAKE_COMMAND} -E env
            "LLVM_PROFILE_FILE=${coverage_dir}/profraw/tensile-host-coverage_%m.profraw"
            GTEST_LISTENER=NO_PASS_LINE_IN_LOG $<TARGET_FILE:TensileTests>
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
