# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21.2)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(tensile_shim)

project(tensile VERSION 4.44.0 LANGUAGES CXX)

include(CMakeDependentOption)
include(tensile_supported_architectures)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(tensile_python)

# Set C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TENSILE_BUILD_SHARED_LIBS "Build shared library" OFF)
option(TENSILE_BUILD_TESTING "Build host library tests" OFF)
option(TENSILE_ENABLE_HOST "Build host library" ON)
option(TENSILE_ENABLE_DEVICE "Build device libraries" ON)
option(TENSILE_ENABLE_CLIENT "Build client app" OFF)
option(TENSILE_ENABLE_MSGPACK "Enable MessagePack support" ON)
option(TENSILE_ENABLE_LLVM "Use llvm yaml library" OFF)
cmake_dependent_option(TENSILE_ENABLE_ROCM_SMI "Use rocm_smi." ON "NOT WIN32" OFF)

set(TENSILE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Source")

set(TENSILE_CXX_STANDARD "17" CACHE STRING "C++ standard version to use (14, 17, 20)")
set_property(CACHE TENSILE_CXX_STANDARD PROPERTY STRINGS "14" "17" "20")

set(GPU_TARGETS "" CACHE STRING "AMD GFX targets to cross-compile")

if(NOT GPU_TARGETS OR GPU_TARGETS STREQUAL "all")
    tensile_get_base_architectures(GPU_TARGETS)
else()
    tensile_validate_gpu_targets("${GPU_TARGETS}")
endif()

message(STATUS "Building for GPU targets: ${GPU_TARGETS}")

find_package(OpenMP REQUIRED)
find_package(hip REQUIRED)

tensile_find_python(Development)
tensile_configure_bundled_python_command("${CMAKE_BINARY_DIR}/lib" "")

if(TENSILE_ENABLE_HOST)
    if(TENSILE_BUILD_SHARED_LIBS)
        add_library(tensile-host SHARED)
    else()
        add_library(tensile-host STATIC)
    endif()

    add_library(roc::tensile-host ALIAS tensile-host)

    target_link_libraries(tensile-host PRIVATE hip::host)

    target_include_directories(tensile-host
        PUBLIC
            $<BUILD_INTERFACE:${TENSILE_SOURCE_DIR}/lib/include>
        PRIVATE
            "${TENSILE_SOURCE_DIR}/lib/include/Tensile"
    )

    target_compile_definitions(tensile-host PUBLIC TENSILE_USE_HIP __HIP_HCC_COMPAT_MODE__=1)

    # NOTE: C++ standard inherited from global CMAKE_CXX_STANDARD=17 setting
    # NOTE: CXX_EXTENSIONS inherited from global CMAKE_CXX_EXTENSIONS=OFF setting

    if(NOT TENSILE_BUILD_SHARED_LIBS)
        set_target_properties(tensile-host PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
        target_compile_definitions(tensile-host PRIVATE TENSILE_STATIC_ONLY)
    endif()

    if(TENSILE_ENABLE_MSGPACK)
        # Prefer msgpack-cxx 6.x, fallback to 5.x
        find_package(msgpack-cxx CONFIG)
        if(msgpack-cxx_FOUND)
            message(STATUS "Found msgpack-cxx (>=6.x)")
        else()
            find_package(msgpack::msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
            message(STATUS "Found msgpack (<=5.x)")
            add_library(msgpack::msgpack-cxx ALIAS msgpackc-cxx)
        endif()

        target_link_libraries(tensile-host PRIVATE msgpack::msgpack-cxx)
        target_compile_definitions(tensile-host PRIVATE TENSILE_MSGPACK)
    endif()

    if(TENSILE_ENABLE_LLVM)
        find_package(LLVM REQUIRED)
        find_library(LLVMObjectYAML_LIBRARY NAMES LLVMObjectYAML PATHS ${LLVM_LIBRARY_DIR})

        if(LLVMObjectYAML_LIBRARY)
            target_link_libraries(tensile-host PRIVATE LLVMObjectYAML)
        else()
            target_link_libraries(tensile-host PRIVATE LLVM)
        endif()

        target_include_directories(tensile-host PRIVATE ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(tensile-host PRIVATE TENSILE_YAML)
    endif()

    if(TENSILE_ENABLE_LLVM OR TENSILE_ENABLE_MSGPACK)
        target_compile_definitions(tensile-host PUBLIC TENSILE_DEFAULT_SERIALIZATION)
    endif()

    add_subdirectory("${TENSILE_SOURCE_DIR}/lib")
endif()

if(TENSILE_ENABLE_DEVICE)
    add_subdirectory(device-library)
endif()

if(TENSILE_ENABLE_ROCM_SMI)
    find_package(rocm_smi REQUIRED)
endif()

if(TENSILE_ENABLE_CLIENT)
    find_package(Boost REQUIRED COMPONENTS program_options)

    add_library(tensile-client-lib OBJECT)
    add_library(tensile::tensile-client-lib ALIAS tensile-client-lib)

    target_link_libraries(tensile-client-lib
        PRIVATE
            Boost::boost
            Boost::program_options
            OpenMP::OpenMP_CXX
            roc::tensile-host
    )

    target_include_directories(tensile-client-lib PUBLIC $<BUILD_INTERFACE:${TENSILE_SOURCE_DIR}/client/include>)

    # NOTE: C++ standard inherited from global CMAKE_CXX_STANDARD=17 setting

    if(rocm_smi_FOUND)
        target_link_libraries(tensile-client-lib PRIVATE rocm_smi64)
    endif()

    add_executable(tensile-client)

    target_link_libraries(tensile-client PRIVATE roc::tensile-host tensile::tensile-client-lib)

    target_compile_definitions(tensile-client
        PRIVATE
            BOOST_ALL_NO_LIB
            BOOST_PROGRAM_OPTIONS_DYN_LINK
        PUBLIC
            __HIP_HCC_COMPAT_MODE__=1
    )

    add_subdirectory("${TENSILE_SOURCE_DIR}/client")
endif()

if(TENSILE_BUILD_TESTING)
    find_package(Boost REQUIRED COMPONENTS filesystem)
    find_package(GTest REQUIRED)

    add_library(tensile-test-lib OBJECT)
    add_library(tensile::tensile-test-lib ALIAS tensile-test-lib)

    target_link_libraries(tensile-test-lib PRIVATE roc::tensile-host)

    target_include_directories(tensile-test-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/HostLibraryTests/testlib/include>)

    target_compile_definitions(tensile-test-lib
        PRIVATE
            BOOST_ALL_NO_LIB
            BOOST_ERROR_CODE_HEADER_ONLY
            BOOST_FILESYSTEM_DYN_LINK
    )

    # NOTE: C++ standard inherited from global CMAKE_CXX_STANDARD=17 setting

    add_executable(tensile-tests)

    target_link_libraries(tensile-tests
        PRIVATE
            Boost::filesystem
            GTest::gtest
            OpenMP::OpenMP_CXX
            hip::host
            roc::tensile-host
            tensile-client-lib
            tensile::tensile-test-lib
    )

    if(TENSILE_ENABLE_LLVM)
        target_link_libraries(tensile-tests PRIVATE LLVMObjectYAML)
    endif()

    target_include_directories(tensile-tests PRIVATE $<BUILD_INTERFACE:${TENSILE_SOURCE_DIR}/client/include>)

    # Data directory needs to be parallel to tensile-tests executable
    set(TENSILE_TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/../data")
    file(MAKE_DIRECTORY "${TENSILE_TEST_DATA_DIR}")
endif()

install(
    TARGETS tensile-host
    EXPORT tensile-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime
)

install(
    EXPORT tensile-targets
    FILE tensile-targets.cmake
    NAMESPACE roc::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensile
)

install(
    DIRECTORY "${TENSILE_SOURCE_DIR}/lib/include/Tensile"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT devel
)
