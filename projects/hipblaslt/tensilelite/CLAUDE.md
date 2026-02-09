# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensileLite is an auto-tuning framework for GPU tensor contractions (GEMMs and higher-dimensional operations) that's part of the hipBLASLt library. It uses YAML configurations to describe problems and generates optimized GPU kernel code with assembly-level optimizations for AMD GPUs.

## Build Commands

**Build client with invoke (recommended for development):**
```bash
pip3 install invoke
invoke build-client                                    # defaults: build_tmp, Release, gfx90a
invoke build-client --build-type Debug --gpu-targets gfx942
invoke build-client --clean                            # clean rebuild
```

**Build with CMake preset:**
```bash
cmake --preset tensilelite -S .. -B my-build -DGPU_TARGETS=gfx90a
cmake --build my-build --parallel
```

**Rebuild assembly to object code (for tuning):**
```bash
make co TENSILE_OUT=tensile-out ARCH=gfx942 WAVE=64
```

## Testing

**Full test suite:**
```bash
tox -e py3 -- Tensile/Tests -m common
```

**Unit tests only (fast, no client build):**
```bash
tox -e unit -- Tensile/Tests/unit
pytest Tensile/Tests/unit/
```

**Single test:**
```bash
pytest -v Tensile/Tests/unit/test_CustomSchedule.py::TestCustomSchedule::test_schedule_256x256x64_16bit
```

**Run individual YAML test:**
```bash
Tensile/bin/Tensile Tensile/Tests/common/gemm/fp16_use_e.yaml tensile-out
# Or with custom client location:
Tensile/bin/Tensile <test>.yaml tensile-out --prebuilt-client=/path/to/tensilelite-client
```

**Test markers:** common, unit, validate, gemm, groupedgemm, sparse, streamk, gsu, amaxd, gfx11, gfx12

## Linting and Formatting

```bash
tox -e lint        # flake8 (pyflakes only, ignores style)
tox -e format      # black (line-length=100)
tox -e isort       # isort (black profile)
tox -e pre_commit  # flake8 + unit tests
```

## Architecture

### Core Components

- **Tensile/Tensile.py** - Main orchestrator; workflow: BenchmarkProblems → LibraryLogic → ClientWriter
- **Tensile/KernelWriter*.py** - Code generation (KernelWriterAssembly.py has 14k+ lines of assembly optimization)
- **Tensile/Components/** - Optimization components:
  - `CustomSchedule.py` - Loop scheduling (205KB)
  - `CMSValidator.py` - Custom Memory Scheduling validation
  - `StreamK.py` - Stream-K kernel support
  - `GSU.py` - Generic Stream Unit
  - `LocalRead.py` - Local memory operations
- **Tensile/SolutionStructs/** - Problem and solution definitions with validators
- **Tensile/Common/** - Shared utilities (GlobalParameters, Architectures, DataType)

### rocisa Module

C++ AMDGPU ISA module with Python bindings (nanobind). Located in `rocisa/`. Handles assembly/parsing for GPU instructions.

### Client Application

C++ executable in `client/` for running generated kernels. Built via invoke or CMake.

### Test Structure

- `Tensile/Tests/unit/` - Python unit tests (pytest)
- `Tensile/Tests/common/` - YAML-based integration tests organized by feature (gemm/, groupedgemm/, sparse/, streamk/, etc.)

## Supported Architectures

CDNA: gfx940, gfx941, gfx942, gfx950
RDNA: gfx1010-1012, gfx1030, gfx1100-1102, gfx1200, gfx1201
Older: gfx900, gfx906, gfx908, gfx90a

## Kernel Arguments

Kernel argument signatures are versioned (v0, v1, v2). See `Tensile/Components/README.md` for detailed byte-level layouts including internalArgs bitfields (input type, StaggerU, GSU, WGM).
