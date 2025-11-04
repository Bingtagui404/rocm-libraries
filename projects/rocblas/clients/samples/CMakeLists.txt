# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

set(CMAKE_CXX_STANDARD 14)

add_library(rocblas-samples-common INTERFACE)
add_library(rocblas::samples-common ALIAS rocblas-samples-common)

target_link_libraries(rocblas-samples-common INTERFACE hip::device roc::rocblas rocblas::testing-common rocblas::warnings)
target_include_directories(rocblas-samples-common INTERFACE "${PROJECT_SOURCE_DIR}/clients/include")
target_compile_definitions(rocblas-samples-common INTERFACE ROCBLAS_INTERNAL_API ROCM_USE_FLOAT16 __HIP_HCC_COMPAT_MODE__=1)

add_executable(rocblas-example-c-dgeam)

target_link_libraries(rocblas-example-c-dgeam PRIVATE rocblas::samples-common)

# FLAG: -std=c11 - C example requires C11 standard
target_compile_options(rocblas-example-c-dgeam PRIVATE -std=c11)

set_target_properties(rocblas-example-c-dgeam PROPERTIES LINKER_LANGUAGE C)

target_sources(rocblas-example-c-dgeam PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_c_dgeam.c")

add_executable(rocblas-example-sscal)
target_link_libraries(rocblas-example-sscal PRIVATE rocblas::samples-common)
target_sources(rocblas-example-sscal PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sscal.cpp")

add_executable(rocblas-example-scal-template)
target_link_libraries(rocblas-example-scal-template PRIVATE rocblas::samples-common)
target_sources(rocblas-example-scal-template PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_scal_template.cpp")

add_executable(rocblas-example-scal-multiple-strided-batch)
target_link_libraries(rocblas-example-scal-multiple-strided-batch PRIVATE rocblas::samples-common)
target_sources(rocblas-example-scal-multiple-strided-batch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_scal_multiple_strided_batch.cpp")

add_executable(rocblas-example-solver)
target_link_libraries(rocblas-example-solver PRIVATE rocblas::samples-common)
target_sources(rocblas-example-solver PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_solver_rocblas.cpp")

add_executable(rocblas-example-hip-complex-her2)
target_link_libraries(rocblas-example-hip-complex-her2 PRIVATE rocblas::samples-common)
target_sources(rocblas-example-hip-complex-her2 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hip_complex_her2.cpp")

add_executable(rocblas-example-gemv-graph-capture)
target_link_libraries(rocblas-example-gemv-graph-capture PRIVATE rocblas::samples-common)
target_sources(rocblas-example-gemv-graph-capture PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_gemv_graph_capture.cpp")

add_executable(rocblas-example-header-check)

target_link_libraries(rocblas-example-header-check
    PRIVATE
        hip::device
        hip::host
        roc::rocblas
        Threads::Threads
)

# Uses C++11 to verify header compatibility with older C++ standard - override directory default
set_target_properties(rocblas-example-header-check PROPERTIES CXX_STANDARD 11)

target_compile_definitions(rocblas-example-header-check PRIVATE ROCBLAS_INTERNAL_API ROCM_USE_FLOAT16)
target_compile_options(rocblas-example-header-check PRIVATE -Wno-unused-result)

target_sources(rocblas-example-header-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_header_check.cpp")

add_executable(rocblas-example-sgemm)
target_link_libraries(rocblas-example-sgemm PRIVATE rocblas::samples-common)
target_sources(rocblas-example-sgemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sgemm.cpp")

add_executable(rocblas-example-sgemm-strided-batched)
target_link_libraries(rocblas-example-sgemm-strided-batched PRIVATE rocblas::samples-common)
target_sources(rocblas-example-sgemm-strided-batched PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sgemm_strided_batched.cpp")

add_executable(rocblas-example-sgemm-multiple-strided-batch)
target_link_libraries(rocblas-example-sgemm-multiple-strided-batch PRIVATE rocblas::samples-common)
target_sources(rocblas-example-sgemm-multiple-strided-batch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_sgemm_multiple_strided_batch.cpp")

if(ROCBLAS_ENABLE_TENSILE)
    add_executable(rocblas-example-user-driven-tuning)
    target_link_libraries(rocblas-example-user-driven-tuning PRIVATE rocblas::samples-common)
    target_sources(rocblas-example-user-driven-tuning PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_user_driven_tuning.cpp")
endif()

if(ROCBLAS_ENABLE_FORTRAN)
    add_executable(rocblas-example-fortran-axpy)

    target_link_libraries(rocblas-example-fortran-axpy
        PRIVATE
            hip::amdhip64
            roc::rocblas
            rocblas-fortran-client
            Threads::Threads
    )

    add_dependencies(rocblas-example-fortran-axpy rocblas-fortran-module)

    target_sources(rocblas-example-fortran-axpy PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_axpy.f90")

    add_executable(rocblas-example-fortran-scal)

    target_link_libraries(rocblas-example-fortran-scal
        PRIVATE
            hip::amdhip64
            roc::rocblas
            rocblas-fortran-client
            Threads::Threads
    )

    add_dependencies(rocblas-example-fortran-scal rocblas-fortran-module)

    target_sources(rocblas-example-fortran-scal PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_scal.f90")

    add_executable(rocblas-example-fortran-gemv)

    target_link_libraries(rocblas-example-fortran-gemv
        PRIVATE
            hip::amdhip64
            roc::rocblas
            rocblas-fortran-client
            Threads::Threads
    )

    add_dependencies(rocblas-example-fortran-gemv rocblas-fortran-module)

    target_sources(rocblas-example-fortran-gemv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_fortran_gemv.f90")
endif()
