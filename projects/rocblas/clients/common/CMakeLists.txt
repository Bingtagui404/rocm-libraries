# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(rocblas-clients-common STATIC)
add_library(rocblas::clients-common ALIAS rocblas-clients-common)
#set_target_properties(rocblas-clients-common PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(rocblas-clients-common
    PUBLIC 
        rocblas::clients-helper
    PRIVATE
        hip::device
)
target_compile_definitions(rocblas-clients-common PRIVATE GOOGLE_TEST) # Is this the right place for GOOGLE_TEST

if(ROCBLAS_ENABLE_ROCM_SMI)
    target_link_libraries(rocblas-clients-common PUBLIC rocm_smi64) # linking to un-namespaced target
endif()

#if(OpenMP_CXX_FOUND)
#  target_link_libraries(rocblas-clients-common PUBLIC OpenMP::OpenMP_CXX)
#endif()
#
#if(Threads_FOUND)
#    target_link_libraries(rocblas-clients-common PUBLIC Threads::Threads)
#endif()

if(NOT WIN32)
    # Need to investigate stdc++fs:
    # Not clear if we need this flag. When/why did the compiler add the flag?
    # Can we use with c++14? Do we have users still compiling with 14?
    target_link_libraries(rocblas-clients-common PUBLIC m stdc++fs)
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

if(ROCBLAS_ENABLE_FORTRAN)
    add_library(rocblas-fortran-wrappers STATIC)
    add_library(rocblas::fortran-wrappers ALIAS rocblas-fortran-wrappers) # Need to go back to original target name for what is now wrappers
    target_sources(rocblas-fortran-wrappers
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran.f90" # use project source and remove ../ so we can move logic without changing path the fact that we are doing this suggests that there are organizational issues
    )
    if(ROCBLAS_ENABLE_TENSILE)
        target_sources(rocblas-fortran-wrappers
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_fortran_tensile.f90" # use project source and remove ../ so we can move logic without changing path
        )
    endif()
    add_dependencies(rocblas-fortran-wrappers rocblas-fortran-module)
    target_include_directories(rocblas-fortran-wrappers
        PRIVATE
            $<TARGET_PROPERTY:rocblas-fortran-module,Fortran_MODULE_DIRECTORY>
    )

    add_library(rocblas-fortran-client INTERFACE)
    target_link_libraries(rocblas-fortran-client INTERFACE
        rocblas::rocblas-fortran-module
        rocblas::fortran-wrappers
    )
    target_include_directories(rocblas-fortran-client
        INTERFACE
            $<TARGET_PROPERTY:rocblas-fortran-module,Fortran_MODULE_DIRECTORY>
    )
endif()

target_sources(rocblas-testing-common 
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/cblas_interface.cpp"
)
if(ROCBLAS_ENABLE_BLIS)
    target_sources(rocblas-testing-common
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/clients/common/blis_interface.cpp"
    )
endif()

add_subdirectory(blas1)
add_subdirectory(blas2)
add_subdirectory(blas3)
add_subdirectory(blas_ex)
