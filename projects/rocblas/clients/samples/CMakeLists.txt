# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Set output directory for all samples at directory level
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

# Helper function to configure a C++ sample with common settings
# NOTE: All include paths, compile definitions, and most compile options are
# provided by rocblas-clients-helper via rocblas-samples-common
function(configure_rocblas_sample target)
  target_compile_features(${target} PRIVATE cxx_std_14)
  
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(${target} PRIVATE rocblas-clients-helper rocblas-samples-common)
endfunction()

# === C Example ===

add_executable(rocblas-example-c-dgeam)
target_sources(rocblas-example-c-dgeam PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_c_dgeam.c")
target_compile_definitions(rocblas-example-c-dgeam PRIVATE __HIP_PLATFORM_HCC__)
# FLAG: -std=c11 - C example requires C11 standard
target_compile_options(rocblas-example-c-dgeam PRIVATE -std=c11)
# TODO EVALUATE: library/include may be redundant with roc::rocblas link propagation
target_include_directories(rocblas-example-c-dgeam
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../library/include"
)
target_link_libraries(rocblas-example-c-dgeam PRIVATE roc::rocblas Threads::Threads hip::host)
set_target_properties(rocblas-example-c-dgeam PROPERTIES LINKER_LANGUAGE C)

# === C++ BLAS1 Examples ===

add_executable(rocblas-example-sscal)
target_sources(rocblas-example-sscal PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sscal.cpp")
configure_rocblas_sample(rocblas-example-sscal)

add_executable(rocblas-example-scal-template)
target_sources(rocblas-example-scal-template PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_scal_template.cpp")
configure_rocblas_sample(rocblas-example-scal-template)

add_executable(rocblas-example-scal-multiple-strided-batch)
target_sources(rocblas-example-scal-multiple-strided-batch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_scal_multiple_strided_batch.cpp")
configure_rocblas_sample(rocblas-example-scal-multiple-strided-batch)

# === Other C++ Examples ===

add_executable(rocblas-example-solver)
target_sources(rocblas-example-solver PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_solver_rocblas.cpp")
configure_rocblas_sample(rocblas-example-solver)

add_executable(rocblas-example-hip-complex-her2)
target_sources(rocblas-example-hip-complex-her2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hip_complex_her2.cpp")
configure_rocblas_sample(rocblas-example-hip-complex-her2)

add_executable(rocblas-example-gemv-graph-capture)
target_sources(rocblas-example-gemv-graph-capture PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_gemv_graph_capture.cpp")
configure_rocblas_sample(rocblas-example-gemv-graph-capture)

# === Header Check Example (compile-time test, not installed) ===

add_executable(rocblas-example-header-check)
target_sources(rocblas-example-header-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_header_check.cpp")
# PRIVATE includes don't need BUILD_INTERFACE wrapper
target_include_directories(rocblas-example-header-check
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../library/include"
)
target_compile_features(rocblas-example-header-check PRIVATE cxx_std_11)
set_target_properties(rocblas-example-header-check PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(rocblas-example-header-check PRIVATE roc::rocblas hip::host hip::device Threads::Threads)

# === BLAS3 Examples (work with or without Tensile) ===

add_executable(rocblas-example-sgemm)
target_sources(rocblas-example-sgemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sgemm.cpp")
configure_rocblas_sample(rocblas-example-sgemm)

add_executable(rocblas-example-sgemm-strided-batched)
target_sources(rocblas-example-sgemm-strided-batched PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sgemm_strided_batched.cpp")
configure_rocblas_sample(rocblas-example-sgemm-strided-batched)

add_executable(rocblas-example-sgemm-multiple-strided-batch)
target_sources(rocblas-example-sgemm-multiple-strided-batch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sgemm_multiple_strided_batch.cpp")
configure_rocblas_sample(rocblas-example-sgemm-multiple-strided-batch)

# === Tensile-specific Example ===

if(ROCBLAS_ENABLE_TENSILE)
  add_executable(rocblas-example-user-driven-tuning)
  target_sources(rocblas-example-user-driven-tuning PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_user_driven_tuning.cpp")
  configure_rocblas_sample(rocblas-example-user-driven-tuning)
endif()

# === Fortran Examples ===

if(ROCBLAS_ENABLE_FORTRAN)
  add_executable(rocblas-example-fortran-axpy)
  target_sources(rocblas-example-fortran-axpy PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_axpy.f90")
  target_link_libraries(rocblas-example-fortran-axpy PRIVATE rocblas-fortran-client roc::rocblas hip::amdhip64 Threads::Threads)
  add_dependencies(rocblas-example-fortran-axpy rocblas-fortran-module)

  add_executable(rocblas-example-fortran-scal)
  target_sources(rocblas-example-fortran-scal PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_scal.f90")
  target_link_libraries(rocblas-example-fortran-scal PRIVATE rocblas-fortran-client roc::rocblas hip::amdhip64 Threads::Threads)
  add_dependencies(rocblas-example-fortran-scal rocblas-fortran-module)

  add_executable(rocblas-example-fortran-gemv)
  target_sources(rocblas-example-fortran-gemv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_gemv.f90")
  target_link_libraries(rocblas-example-fortran-gemv PRIVATE rocblas-fortran-client roc::rocblas hip::amdhip64 Threads::Threads)
  add_dependencies(rocblas-example-fortran-gemv rocblas-fortran-module)
endif()
