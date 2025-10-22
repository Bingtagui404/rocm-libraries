# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Helper interface target to hold common build/usage requirements for all client targets
# This avoids repeating the same properties across multiple targets
add_library(rocblas-clients-helper INTERFACE)

target_include_directories(rocblas-clients-helper
  INTERFACE
    # Client-local header directories - REQUIRED
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas1>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas3>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/blas_ex>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/blas_ex>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/common>
    # TODO EVALUATE: May be redundant with roc::rocblas link propagation
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    # TODO EVALUATE: Internal rocBLAS headers - needed for some tests accessing internals
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
    # TODO EVALUATE: May not be needed
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src>
)

target_compile_definitions(rocblas-clients-helper
  INTERFACE
    ROCM_USE_FLOAT16
    ROCBLAS_INTERNAL_API
    ROCBLAS_NO_DEPRECATED_WARNINGS
    GOOGLE_TEST
)

if(ROCBLAS_ENABLE_TENSILE)
  target_compile_definitions(rocblas-clients-helper INTERFACE BUILD_WITH_TENSILE)
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
  target_compile_definitions(rocblas-clients-helper INTERFACE BUILD_WITH_HIPBLASLT)
endif()

if(NOT ROCBLAS_ENABLE_FORTRAN)
  target_compile_definitions(rocblas-clients-helper INTERFACE CLIENTS_NO_FORTRAN)
endif()

if(ROCBLAS_ENABLE_ASAN)
  target_compile_definitions(rocblas-clients-helper INTERFACE ASAN_BUILD)
endif()

# FLAG: -Wno-c++17-extensions suppresses warnings from AMD Clang about extension usage
target_compile_options(rocblas-clients-helper INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wno-c++17-extensions>)

target_link_libraries(rocblas-clients-helper
  INTERFACE
    roc::rocblas
    hip::host
    hip::device
    $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
    $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main>
)

# Common library for all rocBLAS clients
add_library(rocblas-clients-common OBJECT)

target_link_libraries(rocblas-clients-common PUBLIC rocblas-clients-helper)

if(OpenMP_CXX_FOUND)
  target_link_libraries(rocblas-clients-common PUBLIC OpenMP::OpenMP_CXX)
endif()

if(Threads_FOUND)
  target_link_libraries(rocblas-clients-common PUBLIC Threads::Threads)
endif()

target_sources(rocblas-clients-common
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/singletons.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/client_omp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/client_utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/frequency_monitor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_arguments.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/argument_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_random.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_parse_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_alloc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gtest_helpers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_matrix.cpp"
)

# Library target for sample common sources - compiled once, linked by all samples
add_library(rocblas-samples-common OBJECT)

target_link_libraries(rocblas-samples-common PUBLIC rocblas-clients-helper)

if(Threads_FOUND)
  target_link_libraries(rocblas-samples-common PUBLIC Threads::Threads)
endif()

target_sources(rocblas-samples-common
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/singletons.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/client_utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_random.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_alloc.cpp"
)

# Testing common library with BLAS reference implementations
add_library(rocblas-clients-testing-common OBJECT)

target_link_libraries(rocblas-clients-testing-common PUBLIC rocblas-clients-helper)

# Link to BLAS library (either BLIS or generic BLAS)
if(ROCBLAS_ENABLE_BLIS)
  target_link_libraries(rocblas-clients-testing-common PUBLIC BLIS::BLIS)
else()
  target_link_libraries(rocblas-clients-testing-common PUBLIC ${BLAS_LIBRARIES})
  # BLAS doesn't provide a target with usage requirements, manually add include if available
  if(BLAS_INCLUDE_DIRS)
    target_include_directories(rocblas-clients-testing-common SYSTEM PRIVATE ${BLAS_INCLUDE_DIRS})
  endif()
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(rocblas-clients-testing-common PUBLIC OpenMP::OpenMP_CXX)
  
  # Set RPATH for OpenMP library on Linux with Clang
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT WIN32)
    if(TARGET OpenMP::OpenMP_CXX)
      set_target_properties(rocblas-clients-testing-common PROPERTIES
        BUILD_RPATH "${HIP_CLANG_ROOT}/lib"
        INSTALL_RPATH "$ORIGIN/../llvm/lib"
      )
    elseif(TARGET OpenMP::omp AND DEFINED openmp_LIB_DIR)
      set_target_properties(rocblas-clients-testing-common PROPERTIES
        BUILD_RPATH "${HIP_CLANG_ROOT}/${openmp_LIB_DIR}"
        INSTALL_RPATH "$ORIGIN/../llvm/${openmp_LIB_DIR}"
      )
    endif()
  endif()
endif()

if(Threads_FOUND)
  target_link_libraries(rocblas-clients-testing-common PUBLIC Threads::Threads)
endif()

target_sources(rocblas-clients-testing-common
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cblas_interface.cpp"
)

# Add BLIS interface if using BLIS
if(ROCBLAS_ENABLE_BLIS)
  target_sources(rocblas-clients-testing-common
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/blis_interface.cpp"
  )
endif()

# Add subdirectories to populate testing common target with sources
add_subdirectory(blas1)
add_subdirectory(blas2)
add_subdirectory(blas3)
add_subdirectory(blas_ex)
