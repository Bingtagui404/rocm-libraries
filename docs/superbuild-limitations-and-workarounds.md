# Superbuild Limitations and Workarounds

## rocblas

### Fortran language support

When building rocblas with Fortran enabled (default), language violations relating to the use of `c_int` before `use iso_c_binding` is processed causes compilation errors when using `flang`:
```bash
/mnt/host/projects/rocblas/clients/samples/example_fortran_scal.f90:115:24: error: No explicit type declared for 'c_int'
                  result(c_int) &
                         ^^^^^
/mnt/host/projects/rocblas/clients/samples/example_fortran_scal.f90:117:17: error: Cannot use-associate 'c_int'; it is already declared in this scope
              use iso_c_binding
```

Although `flang` is included in the ROCm distributed binaries it strictly enforces this violation, whereas `gfortran` allows it.

**Option 1:** Force compilation with `gfortran`. This requires that `gfortran` is installed on the system.
```bash
cmake --preset rocblas -D CMAKE_Fortran_COMPILER=gfortran
```

**Option 2:** Disable fortran.
```bash
cmake --preset rocblas -D ROCBLAS_ENABLE_FORTRAN=OFF
```