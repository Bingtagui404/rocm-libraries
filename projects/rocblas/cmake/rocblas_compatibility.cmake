# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# Validate and handle incompatible build option combinations
# =============================================================================

# Address sanitizer configuration
if(ROCBLAS_ENABLE_ASAN)
  # ASAN requires shared libraries (matches reference build requirement)
  if(NOT ROCBLAS_BUILD_SHARED_LIBS)
    message(FATAL_ERROR 
      "Address sanitizer requires shared libraries.\n"
      "Please set: -DROCBLAS_BUILD_SHARED_LIBS=ON\n"
      "Or disable ASAN: -DROCBLAS_ENABLE_ASAN=OFF")
  endif()
  
  # ASAN incompatible with Fortran linking (matches reference build)
  if(ROCBLAS_ENABLE_FORTRAN)
    message(FATAL_ERROR 
      "Cannot enable Fortran clients and address sanitizer.\n"
      "Please set: -DROCBLAS_ENABLE_FORTRAN=OFF\n"
      "Or disable ASAN: -DROCBLAS_ENABLE_ASAN=OFF")
  endif()
endif()

# Code coverage configuration
if(ROCBLAS_ENABLE_COVERAGE OR BUILD_CODE_COVERAGE)
  # Coverage requires shared libraries for instrumentation
  if(NOT ROCBLAS_BUILD_SHARED_LIBS)
    message(FATAL_ERROR 
      "Code coverage requires shared libraries.\n"
      "Please set: -DROCBLAS_BUILD_SHARED_LIBS=ON\n"
      "Or disable coverage: -DROCBLAS_ENABLE_COVERAGE=OFF")
  endif()
  
  # Coverage incompatible with Fortran linking (matches reference build)
  if(ROCBLAS_ENABLE_FORTRAN)
    message(FATAL_ERROR 
      "Cannot enable Fortran clients and code coverage.\n"
      "Please set: -DROCBLAS_ENABLE_FORTRAN=OFF\n"
      "Or disable coverage: -DROCBLAS_ENABLE_COVERAGE=OFF")
  endif()
endif()
