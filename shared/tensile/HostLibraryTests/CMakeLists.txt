# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.25)
  project(tensile-tests LANGUAGES CXX)

  set(_STANDALONE TRUE)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")

  if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
    cmake_policy(SET CMP0012 NEW)
  endif()

  find_package(hip REQUIRED)

  set(_TENSILE_HOSTLIBRARYTESTS_INCLUDED TRUE)
  option(TENSILE_BUILD_TESTING "Build host library tests" ON)
  option(TENSILE_ENABLE_GEMM "Build high-performance gemm libraries" OFF)
  option(TENSILE_ENABLE_CLIENT "Build client app" ON)
  option(TENSILE_ENABLE_LLVM "Use llvm yaml library" ON)

  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_BINARY_DIR}/parent_build")
endif()

if(TESNILE_ENABLE_CTEST)
  enable_testing()
endif()

find_package(hip REQUIRED)
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_executable(tensile-tests)
target_link_libraries(tensile-tests
  PRIVATE
    Boost::filesystem
    GTest::gtest
    hip::device
)
if(TENSILE_ENABLE_OPENMP)
  target_link_libraries(tensile-tests PRIVATE tensile::omp-shim)
endif()
if(TENSILE_ENABLE_LLVM)
  target_link_libraries(tensile-tests PUBLIC LLVMObjectYAML)
  target_include_directories(tensile-tests PUBLIC ${LLVM_INCLUDE_DIRS})
endif()
target_include_directories(tensile-tests
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../Tensile/Source/client/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/testlib/include"
)
target_compile_definitions(tensile-tests
  PRIVATE
    BOOST_ALL_NO_LIB
    BOOST_ERROR_CODE_HEADER_ONLY
    BOOST_FILESYSTEM_DYN_LINK
    __HIP_HCC_COMPAT_MODE__=1
)

set(TENSILE_TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TENSILE_TEST_DATA_DIR}")

add_subdirectory(configs)

foreach(FILE ${SOLUTION_LIBRARY_FILES})
  if(EXISTS "${FILE}.gz")
    execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
  endif()
  if(EXISTS ${FILE})
    file(COPY ${FILE} DESTINATION "${TENSILE_TEST_DATA_DIR}")
  endif()
endforeach(FILE)

add_subdirectory(testlib)

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
  find_package(LLVM REQUIRED)
  find_library(LLVMObjectYAML_LIBRARY NAMES LLVMObjectYAML PATHS ${LLVM_LIBRARY_DIR})

  if(LLVMObjectYAML_LIBRARY)
    target_link_libraries(tensile-tests PUBLIC LLVMObjectYAML)
  else()
    target_link_libraries(tensile-tests PUBLIC LLVM)
  endif()
  target_include_directories(tensile-tests PUBLIC ${LLVM_INCLUDE_DIRS})

  target_sources(tensile-tests
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/ContractionLibraryLoading_test.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/ContractionFitness_test.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/MultipleSolutionsPerSize_test.cpp"
  )
  add_subdirectory(llvm)
endif()

if(TENSILE_ENABLE_CLIENT)
  target_link_libraries(tensile-tests PRIVATE tensile::client-lib)
  target_sources(tensile-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/client/DataInitialization_test.cpp")
endif()

if(TENSILE_ENABLE_HIP)
  add_subdirectory(hip)
  target_link_libraries(tensile-tests
    PRIVATE
      -Xlinker --whole-archive
  )
  target_link_libraries(tensile-tests
    PRIVATE
      roc::tensile-host
      test_kernels_lite
      test_kernels_lite2
      test_kernels_lite_mixed
      test_tile_aware_selection
    )
  target_link_libraries(tensile-tests
    PRIVATE
      -Xlinker --no-whole-archive
      hip::host
  )
  foreach(arch IN LISTS GPU_TARGETS)
    target_link_libraries(tensile-tests PRIVATE "--offload-arch=${arch}")
  endforeach()
endif()

target_compile_definitions(tensile-tests PRIVATE TENSILE_STATIC_ONLY)

gtest_discover_tests(tensile-tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} TIMEOUT 60)
