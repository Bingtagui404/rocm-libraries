# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

add_library(rocblas-samples-common INTERFACE)
add_library(rocblas::samples-common ALIAS rocblas-samples-common)

target_link_libraries(rocblas-samples-common INTERFACE rocblas::clients-common)

target_compile_features(rocblas-samples-common INTERFACE cxx_std_14)
set_target_properties(rocblas-samples-common PROPERTIES CXX_EXTENSIONS OFF)

#if(Threads_FOUND)
#    target_link_libraries(rocblas-samples-common INTERFACE Threads::Threads) # need to figure out where in the previous build system we were linking in threads
#endif()

add_library(rocblas-device-samples-common INTERFACE)
add_library(rocblas::device-samples-common ALIAS rocblas-device-samples-common)
target_link_libraries(rocblas-device-samples-common INTERFACE rocblas::samples-common hip::device)

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

add_executable(rocblas-example-sscal)
target_sources(rocblas-example-sscal PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sscal.cpp")
target_link_libraries(rocblas-example-sscal PRIVATE rocblas::device-samples-common)

add_executable(rocblas-example-scal-template)
target_sources(rocblas-example-scal-template PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_scal_template.cpp")
target_link_libraries(rocblas-example-scal-template PRIVATE rocblas::device-samples-common)

add_executable(rocblas-example-scal-multiple-strided-batch)
target_sources(rocblas-example-scal-multiple-strided-batch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_scal_multiple_strided_batch.cpp")
target_link_libraries(rocblas-example-scal-multiple-strided-batch PRIVATE rocblas::device-samples-common)

add_executable(rocblas-example-solver)
target_sources(rocblas-example-solver PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_solver_rocblas.cpp")
target_link_libraries(rocblas-example-solver PRIVATE rocblas::device-samples-common)

add_executable(rocblas-example-hip-complex-her2)
target_sources(rocblas-example-hip-complex-her2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hip_complex_her2.cpp")
target_link_libraries(rocblas-example-hip-complex-her2 PRIVATE rocblas::device-samples-common)

add_executable(rocblas-example-gemv-graph-capture)
target_sources(rocblas-example-gemv-graph-capture PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_gemv_graph_capture.cpp")
target_link_libraries(rocblas-example-gemv-graph-capture PRIVATE rocblas::device-samples-common)

add_executable(rocblas-example-header-check)
target_sources(rocblas-example-header-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_header_check.cpp")
target_compile_features(rocblas-example-header-check PRIVATE cxx_std_11)
set_target_properties(rocblas-example-header-check PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(rocblas-example-header-check PRIVATE roc::rocblas hip::host hip::device Threads::Threads)

add_executable(rocblas-example-sgemm)
target_sources(rocblas-example-sgemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sgemm.cpp")
target_link_libraries(rocblas-example-sgemm PRIVATE rocblas::device-samples-common)

add_executable(rocblas-example-sgemm-strided-batched)
target_sources(rocblas-example-sgemm-strided-batched PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sgemm_strided_batched.cpp")
target_link_libraries(rocblas-example-sgemm-strided-batched PRIVATE rocblas::device-samples-common)

add_executable(rocblas-example-sgemm-multiple-strided-batch)
target_sources(rocblas-example-sgemm-multiple-strided-batch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sgemm_multiple_strided_batch.cpp")
target_link_libraries(rocblas-example-sgemm-multiple-strided-batch PRIVATE rocblas::device-samples-common)

if(ROCBLAS_ENABLE_TENSILE)
    add_executable(rocblas-example-user-driven-tuning)
    target_sources(rocblas-example-user-driven-tuning PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_user_driven_tuning.cpp")
    target_link_libraries(rocblas-example-user-driven-tuning PRIVATE rocblas::device-samples-common)
endif()

if(ROCBLAS_ENABLE_OPENMP) # might be a good idea to collect all openmp related code to the same directory 
    add_executable(rocblas-example-openmp)
    target_sources(rocblas-example-openmp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_openmp.cpp")
    target_link_libraries(rocblas-example-openmp PRIVATE rocblas::samples-common OpenMP::OpenMP_CXX) 
endif()

if(ROCBLAS_ENABLE_FORTRAN)
  add_executable(rocblas-example-fortran-axpy)
  target_sources(rocblas-example-fortran-axpy PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_axpy.f90")
  target_link_libraries(rocblas-example-fortran-axpy PRIVATE rocblas-fortran-client roc::rocblas hip::amdhip64 Threads::Threads) # is hip::amdhip64 necessary?
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
