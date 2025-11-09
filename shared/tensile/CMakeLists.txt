# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21.2)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(tensile_shim)

project(tensile VERSION 4.45.0 LANGUAGES CXX)

include(CMakeDependentOption)
include(tensile_supported_architectures)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(tensile_python)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TENSILE_BUILD_SHARED_LIBS "Build shared library" OFF)
option(TENSILE_BUILD_TESTING "Build host library tests" OFF)
option(TENSILE_ENABLE_HOST "Build host library" ON)
option(TENSILE_ENABLE_DEVICE "Build device libraries" ON)
option(TENSILE_ENABLE_CLIENT "Build client app" OFF)
option(TENSILE_ENABLE_MSGPACK "Enable MessagePack support" ON)
option(TENSILE_ENABLE_LLVM "Use llvm yaml library" OFF)
option(TENSILE_ENABLE_OPENMP "Enable openmp support" ON)
cmake_dependent_option(TENSILE_ENABLE_ROCM_SMI "Use rocm_smi." ON "NOT WIN32" OFF)

set(GPU_TARGETS "" CACHE STRING "AMD GFX targets to cross-compile")

if(NOT GPU_TARGETS OR GPU_TARGETS STREQUAL "all")
  tensile_get_base_architectures(GPU_TARGETS)
else()
  tensile_validate_gpu_targets("${GPU_TARGETS}")
endif()

message(STATUS "Building for GPU targets: ${GPU_TARGETS}")

set(CMAKE_SKIP_BUILD_RPATH FALSE CACHE BOOL "Skip build RPATH")
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE CACHE BOOL "Build with install RPATH")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../llvm/lib" CACHE STRING "Install RPATH")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE CACHE BOOL "Use link path for RPATH")

mark_as_advanced(
  CMAKE_SKIP_BUILD_RPATH
  CMAKE_BUILD_WITH_INSTALL_RPATH
  CMAKE_INSTALL_RPATH
  CMAKE_INSTALL_RPATH_USE_LINK_PATH
)

find_package(hip REQUIRED)

if(TENSILE_ENABLE_OPENMP)
  find_package(OpenMP CONFIG)
  add_library(tensile-omp-shim INTERFACE)
  add_library(tensile::omp-shim ALIAS tensile-omp-shim)
  if(OpenMP_FOUND)
    set_target_properties(tensile-omp-shim
      PROPERTIES
        INTERFACE_COMPILE_OPTIONS -fopenmp
        INTERFACE_LINK_LIBRARIES OpenMP::omp
    )
  else()
    message(STATUS "Did not find rocm-omp - falling back to gomp")
    find_package(OpenMP REQUIRED)
    target_link_libraries(tensile-omp-shim INTERFACE OpenMP::OpenMP_CXX)
  endif()
endif()

tensile_find_python(Development)
tensile_configure_bundled_python_command("${CMAKE_BINARY_DIR}/lib" "")

if(TENSILE_ENABLE_HOST)
  if(TENSILE_BUILD_SHARED_LIBS)
    add_library(tensile-host SHARED)
  else()
    add_library(tensile-host STATIC)
  endif()

  add_library(roc::tensile-host ALIAS tensile-host)

  target_link_libraries(tensile-host PRIVATE hip::device)

  target_include_directories(tensile-host
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Source/lib/include>
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Source/lib/include/Tensile"
  )

  target_compile_definitions(tensile-host PUBLIC TENSILE_USE_HIP __HIP_HCC_COMPAT_MODE__=1)

  if(NOT TENSILE_BUILD_SHARED_LIBS)
    set_target_properties(tensile-host PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    target_compile_definitions(tensile-host PRIVATE TENSILE_STATIC_ONLY)
  endif()

  if(TENSILE_ENABLE_MSGPACK)
    find_package(msgpack-cxx CONFIG)
    if(msgpack-cxx_FOUND)
      message(STATUS "Found msgpack-cxx (>=6.x)")
    else()
      find_package(msgpack::msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
      message(STATUS "Found msgpack (<=5.x)")
      add_library(msgpack::msgpack-cxx ALIAS msgpackc-cxx)
    endif()

    target_link_libraries(tensile-host PUBLIC msgpack::msgpack-cxx)
    target_compile_definitions(tensile-host PUBLIC TENSILE_MSGPACK)
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
    target_compile_definitions(tensile-host PUBLIC TENSILE_YAML)
  endif()

  if(TENSILE_ENABLE_LLVM OR TENSILE_ENABLE_MSGPACK)
    target_compile_definitions(tensile-host PUBLIC TENSILE_DEFAULT_SERIALIZATION)
  endif()

  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Source/lib")
endif()

if(TENSILE_ENABLE_DEVICE)
  add_subdirectory(device-library)
endif()

if(TENSILE_ENABLE_ROCM_SMI)
  find_package(rocm_smi REQUIRED)
endif()

if(TENSILE_ENABLE_CLIENT)

  add_library(tensile-client-lib STATIC)
  add_library(tensile::client-lib ALIAS tensile-client-lib)

  find_package(Boost REQUIRED COMPONENTS program_options)

  target_link_libraries(tensile-client-lib
    PRIVATE
      Boost::boost
      Boost::program_options
      tensile::omp-shim
      roc::tensile-host
      hip::device
  )

  target_include_directories(tensile-client-lib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Source/client/include")

  if(TENSILE_ENABLE_ROCM_SMI)
    target_link_libraries(tensile-client-lib PRIVATE rocm_smi64)
  endif()

  target_compile_definitions(tensile-client-lib
    PRIVATE
      BOOST_ALL_NO_LIB
      BOOST_PROGRAM_OPTIONS_DYN_LINK
      __HIP_HCC_COMPAT_MODE__=1
  )

  add_executable(tensile-client)
  target_link_libraries(tensile-client PRIVATE tensile::client-lib roc::tensile-host hip::device)
  target_include_directories(tensile-client PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Source/client/include")

  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Source/client")
endif()

if(TENSILE_BUILD_TESTING AND NOT _TENSILE_HOSTLIBRARYTESTS_INCLUDED)
  add_subdirectory(HostLibraryTests)
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
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Source/lib/include/Tensile"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT devel
)
