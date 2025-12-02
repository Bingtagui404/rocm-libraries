# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

message(STATUS "Code coverage for tensile-host is ENABLED.")

target_compile_options(tensile-host PRIVATE -fprofile-instr-generate -fcoverage-mapping)
target_link_options(tensile-host PUBLIC -fprofile-instr-generate)
target_compile_options(tensile-client-lib PRIVATE -fprofile-instr-generate -fcoverage-mapping)
target_link_options(tensile-client-lib PUBLIC -fprofile-instr-generate)

if(TENSILE_BUILD_TESTING OR BUILD_TESTING)

  set(coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")

  add_custom_target(
    code_cov_tests
    DEPENDS tensile-tests
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${coverage_dir}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${coverage_dir}/profraw"
    COMMAND
      ${CMAKE_COMMAND} -E env
      "LLVM_PROFILE_FILE=${coverage_dir}/profraw/tensile-host-coverage_%m.profraw"
      GTEST_LISTENER=NO_PASS_LINE_IN_LOG $<TARGET_FILE:tensile-tests>
      "--gtest_filter=-*Extended*:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/504:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/505:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/506:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/507:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/508:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/509"
  )
  find_program(LLVM_PROFDATA llvm-profdata
    PATHS ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES llvm/bin
    NO_DEFAULT_PATH
    REQUIRED
  )
  find_program(LLVM_COV llvm-cov
    PATHS ENV ROCM_PATH /opt/rocm
    PATH_SUFFIXES llvm/bin
    NO_DEFAULT_PATH
    REQUIRED
  )
  message(STATUS "COV: ${LLVM_COV}")
  message(STATUS "PROF: ${LLVM_PROFDATA}")
  add_custom_target(
    coverage
    DEPENDS code_cov_tests
    COMMAND
      ${LLVM_PROFDATA} merge -sparse "${coverage_dir}/profraw/tensile-host-coverage_*.profraw"
      -o "${coverage_dir}/tensile-host.profdata"
    COMMAND
      ls -al "${coverage_dir}/*"
    COMMAND
      ${LLVM_COV} report -object $<TARGET_FILE:tensile-host>
      -instr-profile="${coverage_dir}/tensile-host.profdata"
    COMMAND
      ls -al "${coverage_dir}/*"
    COMMAND
      ${LLVM_COV} show -object $<TARGET_FILE:tensile-host>
      -instr-profile="${coverage_dir}/tensile-host.profdata" -format=html
      -output-dir="${coverage_dir}"
    COMMAND
      ls -al "${coverage_dir}/*"
    COMMAND
      tree "${coverage_dir}"
    COMMENT
      "Generating tensile-host coverage... HTML report in ${coverage_dir}/index.html"
  )
endif()
