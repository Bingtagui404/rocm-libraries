# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(rocblas-clients-common STATIC)
add_library(rocblas::clients-common ALIAS rocblas-clients-common)

target_include_directories(rocblas-clients-common
  PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clients/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
)

target_compile_definitions(rocblas-clients-common
    PRIVATE
        __HIP_HCC_COMPAT_MODE__=1
        #ROCBLAS_INTERNAL_API
        ROCM_USE_FLOAT16
        ROCBLAS_NO_DEPRECATED_WARNINGS
)

if(ROCBLAS_ENABLE_TENSILE)
  target_compile_definitions(rocblas-clients-common PRIVATE BUILD_WITH_TENSILE)
endif()

if(ROCBLAS_ENABLE_HIPBLASLT)
  target_compile_definitions(rocblas-clients-common PRIVATE BUILD_WITH_HIPBLASLT)
endif()

if(NOT ROCBLAS_ENABLE_FORTRAN)
  target_compile_definitions(rocblas-clients-common PRIVATE CLIENTS_NO_FORTRAN)
endif()

if(ROCBLAS_ENABLE_ASAN)
    target_compile_definitions(rocblas-clients-common PRIVATE ASAN_BUILD)
endif()

if(ROCBLAS_ENABLE_BLIS)
    target_compile_definitions(rocblas-clients-common PRIVATE
        BLIS_ENABLE_NO_UNDERSCORE_API
        BLIS_ENABLE_CBLAS)
endif()

# Compile options migrated from rocblas-clients-helper
# FLAG: -Wno-c++17-extensions suppresses warnings from AMD Clang about extension usage
target_compile_options(rocblas-clients-common PRIVATE -Wno-c++17-extensions)

# Link libraries migrated from rocblas-clients-helper
target_link_libraries(rocblas-clients-common
    PRIVATE
        roc::rocblas
        hip::host
        rocblas::warnings
        hip::device
)

if(ROCBLAS_ENABLE_ROCM_SMI)
    target_link_libraries(rocblas-clients-common PRIVATE rocm_smi64) # linking to un-namespaced target
endif()

if(NOT WIN32)
    # Need to investigate stdc++fs:
    # Not clear if we need this flag. When/why did the compiler add the flag?
    # Can we use with c++14? Do we have users still compiling with 14?
    target_link_libraries(rocblas-clients-common PRIVATE m stdc++fs)
endif()

target_sources(rocblas-clients-common
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/singletons.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/frequency_monitor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_arguments.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/argument_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_random.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_parse_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/host_alloc.cpp"
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
