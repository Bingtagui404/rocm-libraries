# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(tensile-test-lib OBJECT)
add_library(tensile::tensile-test-lib ALIAS tensile-test-lib)

target_sources(tensile-test-lib
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/source/TestData.cpp"
)

target_include_directories(tensile-test-lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(tensile-test-lib
    PUBLIC
        roc::tensile-host
    PRIVATE
        Boost::filesystem
)

target_compile_definitions(tensile-test-lib PRIVATE BOOST_ERROR_CODE_HEADER_ONLY)

# NOTE: C++ standard inherited from global CMAKE_CXX_STANDARD=17 setting

if(TENSILE_USE_OPENMP)
    target_link_libraries(tensile-test-lib PRIVATE OpenMP::OpenMP_CXX)
endif()
