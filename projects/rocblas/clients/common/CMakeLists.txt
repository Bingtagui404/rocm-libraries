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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    # TODO EVALUATE: May be redundant with roc::rocblas link propagation
    #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    # TODO EVALUATE: Internal rocBLAS headers - needed for some tests accessing internals
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
    # Required for internal BLAS implementation headers
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/blas1>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/blas2>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/blas3>
)

target_compile_definitions(rocblas-clients-helper
  INTERFACE
    ROCM_USE_FLOAT16
    ROCBLAS_INTERNAL_API
    ROCBLAS_NO_DEPRECATED_WARNINGS
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
)

# Common library for all rocBLAS clients
add_library(rocblas-clients-common OBJECT)

target_link_libraries(rocblas-clients-common PUBLIC rocblas-clients-helper)

# GTest is needed for clients-common (used by benchmarks and tests)
# Use PRIVATE to prevent propagation to targets that don't need it
target_compile_definitions(rocblas-clients-common PRIVATE GOOGLE_TEST)
target_link_libraries(rocblas-clients-common 
  PUBLIC
    $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
    $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main>
)

# Link ROCmSMI for frequency monitoring (Linux only)
if(NOT WIN32 AND ROCBLAS_ENABLE_ROCM_SMI)
    target_link_libraries(rocblas-clients-common PUBLIC rocm_smi64)
endif()

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

# GTest is only needed for testing clients, not samples or benchmarks
target_compile_definitions(rocblas-clients-testing-common PUBLIC GOOGLE_TEST)
target_link_libraries(rocblas-clients-testing-common 
  PUBLIC
    $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
    $<IF:$<TARGET_EXISTS:GTest::gtest_main>,GTest::gtest_main,GTest::Main>
)

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

# OpenMP configuration - used for parallel reference implementations
if(OpenMP_CXX_FOUND)
    target_link_libraries(rocblas-clients-testing-common PUBLIC OpenMP::OpenMP_CXX)
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

# Fortran client interface library
if(ROCBLAS_ENABLE_FORTRAN)
  # Compile Fortran wrapper functions as a static library
  add_library(rocblas-fortran-wrappers STATIC)
  target_sources(rocblas-fortran-wrappers
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran.f90"
  )
  if(ROCBLAS_ENABLE_TENSILE)
    target_sources(rocblas-fortran-wrappers
      PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran_tensile.f90"
    )
  endif()
  add_dependencies(rocblas-fortran-wrappers rocblas-fortran-module)
  target_include_directories(rocblas-fortran-wrappers
    PRIVATE
      $<TARGET_PROPERTY:rocblas-fortran-module,Fortran_MODULE_DIRECTORY>
  )

  # Interface library for Fortran client linking
  add_library(rocblas-fortran-client INTERFACE)
  target_link_libraries(rocblas-fortran-client INTERFACE 
    rocblas::rocblas-fortran-module
    rocblas-fortran-wrappers
  )
  target_include_directories(rocblas-fortran-client
    INTERFACE
      $<TARGET_PROPERTY:rocblas-fortran-module,Fortran_MODULE_DIRECTORY>
  )
  
  # Link Fortran wrappers to testing-common so tests/benchmarks can use them
  target_link_libraries(rocblas-clients-testing-common PUBLIC rocblas-fortran-wrappers)
endif()
