################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""
CMS Schedule IO - CLI for parsing and serializing custom schedules.

This module provides functions to:
- Find GUI schedule markers in CustomSchedule.py
- Parse schedule data between markers into JSON-serializable format
- Write modified schedule data back to the file
- Load YAML files through Tensile pipeline to get ScheduleInfo objects

CLI Usage:
    python -m Tensile.Components.CMSScheduleIO find-markers <file>
    python -m Tensile.Components.CMSScheduleIO parse <file>
    python -m Tensile.Components.CMSScheduleIO write <file> --schedule <json_file>
    python -m Tensile.Components.CMSScheduleIO get-schedule-info <yaml_file> [--arch gfx950]

Python API:
    from Tensile.Components.CMSScheduleIO import get_schedule_info_from_yaml
    
    has_sched, sched_info, solution = get_schedule_info_from_yaml("path/to/config.yaml")
    if has_sched:
        sched_info.pretty_print()
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from Tensile.Components.CustomSchedule import hasCustomSchedule, ScheduleInfo


def _setup_tensile_infrastructure(arch: str, cxx_compiler: Optional[str] = None):
    """
    Setup Tensile infrastructure (compiler detection, isaInfoMap, assembler).
    
    Returns:
        Tuple of (isa_info_map, assembler, cxx_compiler)
    """
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.GlobalParameters import assignGlobalParameters
    from Tensile.Common.Utilities import setVerbosity, getVerbosity
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    
    # Suppress verbose output (capability tables, etc.)
    old_verbosity = getVerbosity()
    setVerbosity(0)
    
    try:
        # Auto-detect compiler if not provided
        if cxx_compiler is None:
            cxx_compiler = shutil.which('amdclang++')
            if cxx_compiler is None:
                for rocm_path in ['/opt/rocm/bin/amdclang++', '/opt/rocm/llvm/bin/clang++']:
                    if Path(rocm_path).exists():
                        cxx_compiler = rocm_path
                        break
            if cxx_compiler is None:
                raise FileNotFoundError(
                    "Could not find amdclang++. Please provide cxx_compiler path or ensure ROCm is installed."
                )
        
        # Find offload bundler
        bundler = shutil.which('clang-offload-bundler')
        if bundler is None:
            import glob
            bundler_search_paths = [
                '/opt/rocm/bin/clang-offload-bundler',
                '/opt/rocm/llvm/bin/clang-offload-bundler',
                '/opt/rocm/lib/llvm/bin/clang-offload-bundler',
            ]
            # Also search versioned ROCm paths
            bundler_search_paths.extend(glob.glob('/opt/rocm-*/lib/llvm/bin/clang-offload-bundler'))
            bundler_search_paths.extend(glob.glob('/opt/rocm-*/bin/clang-offload-bundler'))
            for bundler_path in bundler_search_paths:
                if Path(bundler_path).exists():
                    bundler = bundler_path
                    break
        if bundler is None:
            raise FileNotFoundError(
                "Could not find clang-offload-bundler. Please ensure ROCm is installed."
            )
        
        # Setup Tensile infrastructure
        target_isas = [gfxToIsa(arch)]
        isa_info_map = makeIsaInfoMap(target_isas, cxx_compiler)
        
        # Create assembler toolchain
        asm_toolchain = makeAssemblyToolchain(
            cxx_compiler,
            bundler,
            co_version='default',
            build_id_kind='sha1',
            debug=False
        )
        assembler = asm_toolchain.assembler
        
        # Assign global parameters (required for Solution parsing)
        assignGlobalParameters({}, isa_info_map)
    finally:
        # Restore verbosity
        setVerbosity(old_verbosity)
    
    return isa_info_map, assembler, cxx_compiler


def _generate_solutions_from_benchmark_config(
    config: dict,
    isa_info_map: dict,
    assembler,
    print_rejection_reason: bool = False,
) -> List:
    """
    Generate Solution objects from a benchmark config dictionary.
    
    This processes the BenchmarkProblems section of a Tensile YAML config
    and creates Solution objects without running the actual benchmark.
    
    Args:
        config: Parsed YAML config dictionary containing BenchmarkProblems
        isa_info_map: ISA info map from makeIsaInfoMap
        assembler: Assembler object
        print_rejection_reason: Whether to print solution rejection reasons
        
    Returns:
        List of Solution objects
    """
    from copy import deepcopy
    from Tensile.BenchmarkStructs import BenchmarkProcess, constructForkPermutations
    from Tensile.SolutionStructs import Solution
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters, validateMIParameters
    )
    from Tensile.Common import DebugConfig
    
    solutions = []
    debug_config = DebugConfig(
        printSolutionRejectionReason=print_rejection_reason,
        printIndexAssignmentInfo=False,
    )
    
    benchmark_problems = config.get("BenchmarkProblems", [])
    
    for problem_config in benchmark_problems:
        if len(problem_config) < 1:
            continue
            
        problem_type_config = problem_config[0]
        problem_size_group_configs = problem_config[1:] if len(problem_config) > 1 else [{}]
        
        for size_group_config in problem_size_group_configs:
            # Create BenchmarkProcess to parse the config
            benchmark_process = BenchmarkProcess(
                problem_type_config,
                size_group_config,
                debug_config.printIndexAssignmentInfo
            )
            
            # Get the benchmark steps
            for step in benchmark_process.benchmarkSteps:
                # Construct fork permutations
                if step.forkParams:
                    fork_permutations = constructForkPermutations(
                        step.forkParams,
                        step.paramGroups
                    )
                else:
                    fork_permutations = [{}]
                
                # Generate solutions from permutations (following _generateForkedSolutions logic)
                for perm in fork_permutations:
                    # Build solution config like _generateForkedSolutions does
                    solution_config = {
                        "ProblemType": deepcopy(benchmark_process.problemType.state),
                        "ISA": next(iter(isa_info_map.keys()))
                    }
                    solution_config.update(step.constantParams)
                    solution_config.update(perm)
                    
                    # Handle MatrixInstruction conversion (9-element to 4-element + params)
                    mi = solution_config.get("MatrixInstruction", [])
                    wavefrontSize = solution_config.get("WavefrontSize", 64)
                    workgroup = solution_config.get("WorkGroup", [16, 16, 1])
                    ptype = solution_config["ProblemType"]
                    isa = solution_config["ISA"]
                    
                    if len(mi) == 9:
                        miParams = matrixInstructionToMIParameters(
                            mi, isa, wavefrontSize, ptype, workgroup, isa_info_map
                        )
                        solution_config.update(miParams)
                    elif len(mi) == 0:
                        solution_config["EnableMatrixInstruction"] = False
                    
                    # Validate MI parameters before creating Solution
                    if not validateMIParameters(solution_config, isa_info_map, print_rejection_reason):
                        if print_rejection_reason:
                            print(f"Solution rejected by MI validation")
                        continue
                    
                    try:
                        solution = Solution(
                            solution_config,
                            splitGSU=debug_config.splitGSU,
                            printSolutionRejectionReason=print_rejection_reason,
                            printIndexAssignmentInfo=debug_config.printIndexAssignmentInfo,
                            assembler=assembler,
                            isaInfoMap=isa_info_map
                        )
                        if solution["Valid"]:
                            solutions.append(solution)
                    except Exception as e:
                        if print_rejection_reason:
                            print(f"Solution rejected: {e}")
    
    return solutions


def get_schedule_info_from_yaml(
    yaml_path: Union[str, Path],
    cxx_compiler: Optional[str] = None,
) -> Tuple[bool, Optional[ScheduleInfo], Optional[dict]]:
    """
    Load a Tensile benchmark config YAML file and return the ScheduleInfo for the solution.
    
    This function sets up the necessary Tensile infrastructure (assembler, isaInfoMap),
    parses the benchmark config YAML, generates the Solution object, and extracts
    the custom schedule information.
    
    Note: The YAML config must generate exactly one solution.
    
    Args:
        yaml_path: Path to the benchmark config YAML file (e.g., 128x256x32NT.yaml)
        cxx_compiler: Path to C++ compiler. If None, auto-detects amdclang++
        
    Returns:
        Tuple of:
        - has_schedule: bool - Whether a custom schedule was found
        - schedule_info: Optional[ScheduleInfo] - The schedule info if found
        - solution_state: Optional[dict] - The solution state dict (kernel config)
        
    Raises:
        FileNotFoundError: If yaml_path or compiler not found
        RuntimeError: If Tensile infrastructure setup fails
        ValueError: If ArchitectureName not found in YAML
        AssertionError: If YAML generates zero or more than one solution
        
    Example:
        >>> has_sched, sched_info, solution = get_schedule_info_from_yaml(
        ...     "128x256x32NT/128x256x32NT.yaml"
        ... )
        >>> if has_sched:
        ...     sched_info.pretty_print()
    """
    import yaml
    import io
    import contextlib
    from Tensile.Common.Utilities import setVerbosity, getVerbosity
    
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    # Load the YAML file first to extract architecture
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load YAML file: {e}") from e
    
    # Extract architecture from LibraryLogic.ArchitectureName
    arch = config.get("LibraryLogic", {}).get("ArchitectureName")
    if arch is None:
        raise ValueError(
            f"ArchitectureName not found in LibraryLogic section of {yaml_path}. "
            f"Expected a field like 'ArchitectureName: \"gfx950\"'"
        )
    
    # Suppress all verbose output during Tensile operations
    old_verbosity = getVerbosity()
    setVerbosity(0)
    
    # Also redirect stdout to suppress any remaining prints
    stdout_capture = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(stdout_capture):
            # Setup Tensile infrastructure
            try:
                isa_info_map, assembler, cxx_compiler = _setup_tensile_infrastructure(arch, cxx_compiler)
            except Exception as e:
                raise RuntimeError(f"Failed to setup Tensile infrastructure: {e}") from e
            
            # Check if this is a benchmark config (has BenchmarkProblems)
            if "BenchmarkProblems" not in config:
                raise ValueError(
                    f"YAML file does not contain 'BenchmarkProblems' section. "
                    f"Expected a benchmark config YAML file like 128x256x32NT.yaml"
                )
            
            # Assign global parameters from config if present
            from Tensile.Common.GlobalParameters import assignGlobalParameters
            if "GlobalParameters" in config:
                assignGlobalParameters(config["GlobalParameters"], isa_info_map)
            
            # Generate solutions from the benchmark config
            try:
                solutions = _generate_solutions_from_benchmark_config(
                    config, isa_info_map, assembler, print_rejection_reason=False
                )
            except Exception as e:
                raise RuntimeError(f"Failed to generate solutions from config: {e}") from e
            
            assert len(solutions) == 1, (
                f"Expected exactly 1 solution from YAML config, but got {len(solutions)}. "
                f"Ensure the YAML defines a single kernel configuration."
            )
            solution = solutions[0]
            
            has_schedule, schedule_info = hasCustomSchedule(solution)
            solution_state = dict(solution._state) if hasattr(solution, '_state') else dict(solution)
    finally:
        setVerbosity(old_verbosity)
    
    return has_schedule, schedule_info, solution_state


# Marker patterns
GUI_SCHEDULE_BEGIN = "# GUI_SCHEDULE_BEGIN"
GUI_SCHEDULE_END = "# GUI_SCHEDULE_END"


def find_gui_schedule_markers(filepath: str) -> Dict[str, Any]:
    """
    Find and validate exactly one GUI_SCHEDULE_BEGIN/END pair in the file.
    
    Args:
        filepath: Path to CustomSchedule.py
        
    Returns:
        Dictionary with:
        - function_name: Name of the schedule function (or None if not specified)
        - start_line: Line number of BEGIN marker (1-indexed)
        - end_line: Line number of END marker (1-indexed)
        - decorator_start: Line number of @RegisterSchedule (1-indexed, or None)
        
    Raises:
        ValueError: If markers are not found or multiple pairs exist
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    begin_markers = []
    end_markers = []
    
    for i, line in enumerate(lines, 1):
        if GUI_SCHEDULE_BEGIN in line and GUI_SCHEDULE_END not in line:
            # Extract function name from marker if present (format: # GUI_SCHEDULE_BEGIN: func_name)
            match = re.search(r'GUI_SCHEDULE_BEGIN:\s*(\w+)', line)
            func_name = match.group(1) if match else None
            begin_markers.append((i, func_name))
        elif GUI_SCHEDULE_END in line:
            end_markers.append(i)
    
    if len(begin_markers) == 0:
        raise ValueError(f"No {GUI_SCHEDULE_BEGIN} marker found in {filepath}")
    if len(begin_markers) > 1:
        raise ValueError(f"Multiple {GUI_SCHEDULE_BEGIN} markers found in {filepath}: lines {[m[0] for m in begin_markers]}")
    if len(end_markers) == 0:
        raise ValueError(f"No {GUI_SCHEDULE_END} marker found in {filepath}")
    if len(end_markers) > 1:
        raise ValueError(f"Multiple {GUI_SCHEDULE_END} markers found in {filepath}: lines {end_markers}")
    
    start_line, func_name = begin_markers[0]
    end_line = end_markers[0]
    
    if end_line <= start_line:
        raise ValueError(f"END marker (line {end_line}) must come after BEGIN marker (line {start_line})")
    
    # Find the @RegisterSchedule decorator (should be right after BEGIN marker)
    decorator_start = None
    for i in range(start_line, min(start_line + 5, len(lines) + 1)):
        if '@RegisterSchedule' in lines[i - 1]:
            decorator_start = i
            break
    
    return {
        "function_name": func_name,
        "start_line": start_line,
        "end_line": end_line,
        "decorator_start": decorator_start
    }


def get_transpose_from_yaml(yaml_path: str) -> str:
    """
    Read TransposeA and TransposeB from YAML file and return transpose combo.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        Transpose combination string: "TN", "NT", "NN", or "TT"
    """
    import yaml
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Navigate to BenchmarkProblems -> first item -> ProblemType
    transpose_a = 0
    transpose_b = 0
    
    if 'BenchmarkProblems' in config:
        for problem_group in config['BenchmarkProblems']:
            if isinstance(problem_group, list):
                for item in problem_group:
                    if isinstance(item, dict):
                        if 'TransposeA' in item:
                            transpose_a = item['TransposeA']
                        if 'TransposeB' in item:
                            transpose_b = item['TransposeB']
                        if 'TransposeA' in item or 'TransposeB' in item:
                            break
    
    # Convert to letter codes: 0 = N (not transposed), 1 = T (transposed)
    a_code = 'T' if transpose_a else 'N'
    b_code = 'T' if transpose_b else 'N'
    
    return a_code + b_code


def _find_transpose_branch(code: str, lines: List[str], start_line: int, end_line: int, transpose: str) -> Tuple[int, int]:
    """
    Find the line range for the specific transpose branch (e.g., "NT", "TN").
    
    Args:
        code: Full code between markers
        lines: List of all lines in the file
        start_line: Start line of markers (1-indexed)
        end_line: End line of markers (1-indexed)
        transpose: Transpose combo like "TN", "NT", "NN", "TT"
        
    Returns:
        Tuple of (branch_start_line, branch_end_line) as 1-indexed line numbers
    """
    # Pattern to match transpose check: if isTN(kernel) or elif isNT(kernel), etc.
    transpose_pattern = f"is{transpose}\\(kernel\\)"
    
    branch_start = None
    branch_end = None
    indent_level = None
    
    for i in range(start_line - 1, end_line):
        line = lines[i]
        
        # Check if this line contains our transpose check
        if re.search(transpose_pattern, line) and ('if ' in line or 'elif ' in line):
            branch_start = i + 1  # Convert to 1-indexed
            # Get the indentation level of the if/elif
            indent_level = len(line) - len(line.lstrip())
            continue
        
        # If we're inside the branch, look for the end
        if branch_start is not None and indent_level is not None:
            # Check for elif or else at the same indent level (end of our branch)
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
            
            if current_indent <= indent_level and stripped and not stripped.startswith('#'):
                if stripped.startswith('elif ') or stripped.startswith('else:') or stripped.startswith('return '):
                    branch_end = i  # Line before this (0-indexed, so this is correct for 1-indexed end)
                    break
    
    # If no end found, use the marker end
    if branch_start is not None and branch_end is None:
        branch_end = end_line
    
    if branch_start is None:
        # Fallback: return full range if transpose branch not found
        return start_line, end_line
    
    return branch_start, branch_end


def parse_schedule(filepath: str, yaml_path: str = None, transpose: str = None) -> Dict[str, Any]:
    """
    Parse the schedule source code for kernel settings and check for dynamic values.
    
    NOTE: This function is simplified - optSchedule, syncCode, mfmaReorder now come
    from get_schedule_info_from_yaml (computed via Tensile pipeline). This function
    only extracts kernel settings as raw strings and checks for unsupported dynamic values.
    
    Args:
        filepath: Path to CustomSchedule.py
        yaml_path: Optional path to YAML file to determine transpose combo
        transpose: Optional transpose combo override ("TN", "NT", "NN", "TT")
        
    Returns:
        Dictionary with:
        - function_name: Name of the schedule function
        - kernel_settings_raw: List of raw kernel setting strings (written back as-is)
        - transpose: The transpose combination being used
        - branch_lines: Tuple of (start, end) line numbers for the active branch
        - markers: Marker info for writing back to file
        
    Raises:
        NotImplementedError: If dynamic values (# @DYNAMIC markers) are found in syncCode
    """
    markers = find_gui_schedule_markers(filepath)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Determine transpose combination
    if transpose is None and yaml_path:
        transpose = get_transpose_from_yaml(yaml_path)
    
    # Extract the code between markers
    start_idx = markers['start_line'] - 1
    end_idx = markers['end_line'] - 1
    full_raw_code = ''.join(lines[start_idx:end_idx + 1])
    
    # Find the specific transpose branch if transpose is specified
    branch_start = markers['start_line']
    branch_end = markers['end_line']
    
    if transpose:
        branch_start, branch_end = _find_transpose_branch(
            full_raw_code, lines, markers['start_line'], markers['end_line'], transpose
        )
    
    # Extract code for just the active branch
    branch_code = ''.join(lines[branch_start - 1:branch_end])
    
    # Parse dynamic expressions from source (instead of erroring out)
    dynamic_expressions = _parse_dynamic_sync_expressions(branch_code)
    
    # Extract kernel settings as RAW strings (not parsed)
    kernel_settings_raw = _extract_kernel_settings_raw(branch_code)
    
    return {
        "function_name": markers['function_name'],
        "kernel_settings_raw": kernel_settings_raw,
        "transpose": transpose,
        "branch_lines": (branch_start, branch_end),
        "markers": markers,
        "dynamic_sync_expressions": dynamic_expressions
    }


def _parse_dynamic_sync_expressions(code: str) -> Dict[int, Dict[str, str]]:
    """
    Parse syncTable/syncCode from source to extract dynamic expressions.
    
    Looks for lines with # @DYNAMIC markers and extracts the dscnt/vlcnt/vscnt
    expressions (not the computed values).
    
    Args:
        code: The Python source code containing syncTable or syncCode
        
    Returns:
        Dict mapping instruction index to dict of field expressions:
        { 0: { 'dscnt': 'dscnt_after_finish(...)', 'vlcnt': 'vlcnt_after_finish(...)' }, ... }
    """
    result = {}
    
    # Find all SWaitCnt lines with @DYNAMIC marker
    # Match: index, SWaitCnt(...) with @DYNAMIC marker
    # Use a pattern that captures everything up to the # @DYNAMIC marker
    pattern = r'(\d+),\s*SWaitCnt\((.+?)\)\s*[,]?\s*#\s*@DYNAMIC'
    
    for match in re.finditer(pattern, code):
        index = int(match.group(1))
        params_str = match.group(2)
        
        expressions = {}
        
        # Extract each parameter, handling nested parentheses
        for param in ['dscnt', 'vlcnt', 'vscnt']:
            # Find the start of this parameter
            param_start = params_str.find(f'{param}=')
            if param_start == -1:
                continue
            
            # Extract the value, handling nested parentheses
            value_start = param_start + len(param) + 1  # +1 for '='
            value = _extract_balanced_expression(params_str, value_start)
            
            if value and re.match(r'^[a-zA-Z_]', value) and '(' in value:
                expressions[param] = value
        
        if expressions:
            result[index] = expressions
    
    return result


def _extract_balanced_expression(s: str, start: int) -> str:
    """
    Extract an expression from string starting at position 'start',
    handling nested parentheses properly.
    
    Stops at a comma that's not inside parentheses, or at end of string.
    """
    paren_depth = 0
    i = start
    while i < len(s):
        char = s[i]
        if char == '(':
            paren_depth += 1
        elif char == ')':
            if paren_depth == 0:
                # End of expression (closing paren of outer SWaitCnt)
                break
            paren_depth -= 1
        elif char == ',' and paren_depth == 0:
            # Comma outside of any parentheses - end of this parameter
            break
        i += 1
    
    return s[start:i].strip()


def _extract_kernel_settings_raw(code: str) -> List[str]:
    """
    Extract kernel settings as raw strings (to be written back as-is).
    
    Extracts lines like:
        kernel["UseMFMAF32XEmulation"] = True
        kernel["MfmaInitCVgprs"] = True
    
    Args:
        code: The Python source code to extract from
        
    Returns:
        List of raw kernel setting strings
    """
    pattern = r'(kernel\s*\[\s*["\'][^"\']+["\']\s*\]\s*=\s*[^\n]+)'
    return [line.strip() for line in re.findall(pattern, code)]


def write_schedule(filepath: str, schedule_data: Dict[str, Any], yaml_path: str = None, transpose: str = None) -> bool:
    """
    Write the modified schedule back to the file between markers.
    
    Generates complete Python code including:
    - numMfma and numCodePaths
    - Kernel settings
    - optSchedule dictionary (hardcoded arrays)
    - syncCode array (with @DYNAMIC markers preserved)
    - nglshift/nllshift
    - mfmaReorder (if present)
    - return statement
    
    Args:
        filepath: Path to CustomSchedule.py
        schedule_data: Dictionary containing the schedule data to write
        yaml_path: Optional path to YAML file to determine transpose combo
        transpose: Optional transpose combo override ("TN", "NT", "NN", "TT")
        
    Returns:
        True if successful
    """
    markers = find_gui_schedule_markers(filepath)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Determine the search range based on transpose branch
    if 'branch_lines' in schedule_data and schedule_data['branch_lines']:
        start_idx = schedule_data['branch_lines'][0] - 1
        end_idx = schedule_data['branch_lines'][1] - 1
    else:
        if transpose is None and yaml_path:
            transpose = get_transpose_from_yaml(yaml_path)
        
        if transpose:
            branch_start, branch_end = _find_transpose_branch(
                ''.join(lines[markers['start_line']-1:markers['end_line']]),
                lines, markers['start_line'], markers['end_line'], transpose
            )
            start_idx = branch_start - 1
            end_idx = branch_end - 1
        else:
            start_idx = markers['start_line'] - 1
            end_idx = markers['end_line'] - 1
    
    # Find the line with GUI_SCHEDULE_BEGIN comment
    gui_begin_line = None
    for i in range(start_idx, end_idx + 1):
        if 'GUI_SCHEDULE_BEGIN' in lines[i]:
            gui_begin_line = i
            break
    
    # Find the return statement line (end of the schedule block)
    # Look for "return True, ScheduleInfo" or "return ScheduleInfo"
    return_line = None
    for i in range(end_idx, start_idx, -1):
        if 'return' in lines[i] and 'ScheduleInfo' in lines[i]:
            return_line = i
            break
    
    if gui_begin_line is None:
        raise ValueError("Could not find GUI_SCHEDULE_BEGIN marker")
    
    if return_line is None:
        # Fall back to finding GUI_SCHEDULE_END
        for i in range(end_idx, start_idx, -1):
            if 'GUI_SCHEDULE_END' in lines[i]:
                return_line = i - 1
                break
    
    # Generate the new schedule code block
    new_code = _generate_full_schedule_code(schedule_data)
    
    # Replace the content between GUI_SCHEDULE_BEGIN and return statement
    # Keep the line with GUI_SCHEDULE_BEGIN comment and the closing
    # Use return_line + 1 to skip the original return line (new_code includes its own return)
    new_lines = lines[:gui_begin_line + 1] + [new_code] + lines[return_line + 1:]
    
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    
    return True


def _generate_full_schedule_code(schedule_data: Dict[str, Any]) -> str:
    """
    Generate complete Python code for the schedule block.
    
    Uses the syncTable pattern where indices and instructions are paired:
    - syncTable = [idx1, instr1, idx2, instr2, ...]
    - syncCode = syncTable[1::2]
    - optSchedule['SYNC'] = [syncTable[::2]]
    
    Includes: numMfma, numCodePaths, kernel settings, syncTable, optSchedule, syncCode, shifts, mfmaReorder, return
    """
    indent = "        "  # 8 spaces
    lines = []
    
    # 1. numMfma and numCodePaths
    num_mfma = schedule_data.get('num_mfma', 96)
    num_codepaths = schedule_data.get('num_codepaths', 1)
    lines.append(f"{indent}numMfma = {num_mfma}")
    lines.append(f"{indent}numCodePaths = {num_codepaths}")
    
    # 2. Kernel settings (use raw strings if available, otherwise fall back to parsed)
    kernel_settings_raw = schedule_data.get('kernel_settings_raw', [])
    if kernel_settings_raw:
        for raw_line in kernel_settings_raw:
            lines.append(f'{indent}{raw_line}')
    else:
        # Fallback to parsed kernel_settings (deprecated)
        kernel_settings = schedule_data.get('kernel_settings', {})
        for key, value in kernel_settings.items():
            if isinstance(value, bool):
                lines.append(f'{indent}kernel["{key}"] = {value}')
            elif isinstance(value, str):
                lines.append(f'{indent}kernel["{key}"] = "{value}"')
            else:
                lines.append(f'{indent}kernel["{key}"] = {value}')
    
    lines.append("")  # Empty line
    
    # 3. optSchedule dictionary (with empty SYNC list initially if sync_code exists)
    sync_code = schedule_data.get('sync_code', [])
    lines.append(_generate_opt_schedule_code(
        schedule_data.get('opt_schedule', {}),
        schedule_data.get('instruction_types', []),
        use_sync_table=bool(sync_code)
    ))
    lines.append("")
    
    # 4. Generate syncTable (paired indices and instructions) after optSchedule
    if sync_code:
        lines.append(_generate_sync_table(sync_code))
        lines.append("")
        lines.append(f"{indent}syncCode = syncTable[1::2]")
        lines.append(f"{indent}optSchedule['SYNC'] = [syncTable[::2]]")
    
    lines.append("")  # Empty line
    
    # 5. nglshift and nllshift
    nglshift = schedule_data.get('nglshift', 0)
    nllshift = schedule_data.get('nllshift', 0)
    if nglshift == nllshift:
        lines.append(f"{indent}nglshift = nllshift = {nglshift}")
    else:
        lines.append(f"{indent}nglshift = {nglshift}")
        lines.append(f"{indent}nllshift = {nllshift}")
    
    # 6. mfmaReorder (if present)
    mfma_reorder = schedule_data.get('mfma_reorder')
    if mfma_reorder:
        lines.append(f"{indent}mfmaReorder = {mfma_reorder}")
    
    lines.append("")  # Empty line
    
    # 7. Return statement - always returns (True, ScheduleInfo)
    # GUI-generated schedules are always valid, so we always return True
    if mfma_reorder:
        lines.append(f"{indent}return True, ScheduleInfo(numCodePaths, numMfma, optSchedule, syncCode, nglshift, nllshift, mfmaReorder=mfmaReorder)")
    else:
        lines.append(f"{indent}return True, ScheduleInfo(numCodePaths, numMfma, optSchedule, syncCode, nglshift, nllshift)")
    
    return '\n'.join(lines) + '\n'


def _generate_opt_schedule_code(opt_schedule: Dict, instruction_types: List[str], use_sync_table: bool = False) -> str:
    """
    Generate Python code for the optSchedule dictionary with hardcoded arrays.
    
    If use_sync_table=True, uses an empty list for the SYNC key (will be updated after syncTable is defined).
    """
    indent = "        "
    lines = [f"{indent}optSchedule = {{"]
    
    # Use instruction_types order if provided, otherwise use dict order
    keys = instruction_types if instruction_types else list(opt_schedule.keys())
    
    for key in keys:
        if key in opt_schedule:
            if key == 'SYNC' and use_sync_table:
                # Empty list initially - will be updated after syncTable is defined
                lines.append(f"{indent}    'SYNC': [[]],")
            else:
                value = opt_schedule[key]
                # Always output as hardcoded arrays (not variable references)
                lines.append(f"{indent}    '{key}': {value},")
    
    lines.append(f"{indent}}}")
    return '\n'.join(lines)


def _generate_sync_table(sync_code: List[Dict]) -> str:
    """
    Generate Python code for the syncTable array.
    
    syncTable contains paired (index, instruction) entries:
    syncTable = [idx1, instr1, idx2, instr2, ...]
    
    Then syncCode = syncTable[1::2] and optSchedule['SYNC'] = [syncTable[::2]]
    """
    indent = "        "
    lines = [f"{indent}syncTable = ["]
    
    for item in sync_code:
        inst = item.get('instruction', item)  # Handle both indexed and non-indexed format
        index = item.get('index', -1)
        inst_type = inst.get('type', '')
        comment = inst.get('comment', '')
        
        if inst_type == 'SBarrier':
            lines.append(f'{indent}    {index}, SBarrier(comment="{comment}"),')
        
        elif inst_type == 'SWaitCnt':
            dscnt = inst.get('dscnt', -1)
            vlcnt = inst.get('vlcnt', -1)
            vscnt = inst.get('vscnt', -1)
            
            # Build the instruction string
            parts = []
            
            # Handle dscnt
            dscnt_str, dscnt_dynamic = _format_wait_count(dscnt, 'dscnt')
            if dscnt_str:
                parts.append(dscnt_str)
            
            # Handle vlcnt
            vlcnt_str, vlcnt_dynamic = _format_wait_count(vlcnt, 'vlcnt')
            if vlcnt_str:
                parts.append(vlcnt_str)
            
            # Handle vscnt
            vscnt_str, vscnt_dynamic = _format_wait_count(vscnt, 'vscnt')
            if vscnt_str:
                parts.append(vscnt_str)
            
            # Add comment
            parts.append(f'comment="{comment}"')
            
            # Determine if any field is dynamic
            is_dynamic = dscnt_dynamic or vlcnt_dynamic or vscnt_dynamic
            
            inst_str = f'{indent}    {index}, SWaitCnt({", ".join(parts)}),'
            if is_dynamic:
                inst_str += '  # @DYNAMIC'
            lines.append(inst_str)
        
        elif inst_type == 'SNop':
            count = inst.get('count', 1)
            lines.append(f'{indent}    {index}, SNop(count={count}, comment="{comment}"),')
    
    lines.append(f"{indent}]")
    return '\n'.join(lines)


def _generate_sync_code(sync_code: List[Dict]) -> str:
    """
    Generate Python code for the syncCode array (direct format, not using syncTable).
    
    This is kept for backwards compatibility but _generate_sync_table is preferred.
    Preserves @DYNAMIC markers for dynamic expressions.
    """
    indent = "        "
    lines = [f"{indent}syncCode = ["]
    
    for item in sync_code:
        inst = item.get('instruction', item)  # Handle both indexed and non-indexed format
        inst_type = inst.get('type', '')
        comment = inst.get('comment', '')
        
        if inst_type == 'SBarrier':
            lines.append(f'{indent}    SBarrier(comment="{comment}"),')
        
        elif inst_type == 'SWaitCnt':
            dscnt = inst.get('dscnt', -1)
            vlcnt = inst.get('vlcnt', -1)
            vscnt = inst.get('vscnt', -1)
            
            # Build the instruction string
            parts = []
            
            # Handle dscnt
            dscnt_str, dscnt_dynamic = _format_wait_count(dscnt, 'dscnt')
            if dscnt_str:
                parts.append(dscnt_str)
            
            # Handle vlcnt
            vlcnt_str, vlcnt_dynamic = _format_wait_count(vlcnt, 'vlcnt')
            if vlcnt_str:
                parts.append(vlcnt_str)
            
            # Handle vscnt
            vscnt_str, vscnt_dynamic = _format_wait_count(vscnt, 'vscnt')
            if vscnt_str:
                parts.append(vscnt_str)
            
            # Add comment
            parts.append(f'comment="{comment}"')
            
            # Determine if any field is dynamic
            is_dynamic = dscnt_dynamic or vlcnt_dynamic or vscnt_dynamic
            
            inst_str = f'{indent}    SWaitCnt({", ".join(parts)}),'
            if is_dynamic:
                inst_str += '  # @DYNAMIC'
            lines.append(inst_str)
        
        elif inst_type == 'SNop':
            count = inst.get('count', 1)
            lines.append(f'{indent}    SNop(count={count}, comment="{comment}"),')
    
    lines.append(f"{indent}]")
    return '\n'.join(lines)


def _format_wait_count(value: Union[int, Dict], field_name: str) -> Tuple[str, bool]:
    """
    Format a wait count value for code generation.
    
    Returns: (formatted_string, is_dynamic)
    """
    if isinstance(value, dict):
        # Dynamic value - use the expression
        expr = value.get('expr', '-1')
        return f'{field_name}={expr}', True
    elif isinstance(value, int):
        if value == -1:
            return f'{field_name}=-1', False
        return f'{field_name}={value}', False
    else:
        return f'{field_name}=-1', False


def _serialize_sync_code(opt_schedule: dict, sync_code: list, dynamic_expressions: Dict[int, Dict[str, str]] = None) -> list:
    """
    Serialize syncCode list to JSON-serializable format with MFMA indices.
    
    Args:
        opt_schedule: The optSchedule dictionary containing SYNC indices
        sync_code: List of sync instruction objects (SWaitCnt, SBarrier, SNop)
        dynamic_expressions: Optional dict of dynamic expressions parsed from source
        
    Returns:
        List of dicts with 'index' and 'instruction' keys
    """
    if dynamic_expressions is None:
        dynamic_expressions = {}
    
    sync_indices = opt_schedule.get('SYNC', [[]])[0] if 'SYNC' in opt_schedule else []
    result = []
    for i, instr in enumerate(sync_code):
        index = sync_indices[i] if i < len(sync_indices) else -1
        # Get dynamic expressions for this instruction (by index)
        dyn_exprs = dynamic_expressions.get(index, {})
        result.append({
            "index": index,
            "instruction": _serialize_instruction(instr, dyn_exprs)
        })
    return result


def _serialize_instruction(instr, dynamic_expressions: Dict[str, str] = None) -> dict:
    """
    Serialize SWaitCnt/SBarrier/SNop instruction object to JSON-serializable dict.
    
    Args:
        instr: A sync instruction object (SWaitCnt, SBarrier, or SNop)
        dynamic_expressions: Optional dict of dynamic expressions for this instruction
        
    Returns:
        Dict with type and instruction-specific fields
    """
    if dynamic_expressions is None:
        dynamic_expressions = {}
    
    class_name = instr.__class__.__name__
    if class_name == 'SWaitCnt':
        # For each field, check if there's a dynamic expression
        # If dynamic, store as { expr: "...", values: [computed_value] }
        dscnt_val = instr.dscnt
        vlcnt_val = instr.vlcnt
        vscnt_val = instr.vscnt
        
        if 'dscnt' in dynamic_expressions:
            dscnt_val = { "expr": dynamic_expressions['dscnt'], "values": [instr.dscnt] }
        if 'vlcnt' in dynamic_expressions:
            vlcnt_val = { "expr": dynamic_expressions['vlcnt'], "values": [instr.vlcnt] }
        if 'vscnt' in dynamic_expressions:
            vscnt_val = { "expr": dynamic_expressions['vscnt'], "values": [instr.vscnt] }
        
        return {
            "type": "SWaitCnt",
            "dscnt": dscnt_val,
            "vlcnt": vlcnt_val,
            "vscnt": vscnt_val,
            "comment": str(instr.comment) if instr.comment else ""
        }
    elif class_name == 'SBarrier':
        return {
            "type": "SBarrier",
            "comment": str(instr.comment) if instr.comment else ""
        }
    elif class_name == 'SNop':
        return {
            "type": "SNop",
            "count": instr.waitStates,
            "comment": str(instr.comment) if instr.comment else ""
        }
    return {"type": "Unknown"}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CMS Schedule IO - Parse and serialize custom schedules"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # find-markers command
    find_parser = subparsers.add_parser(
        'find-markers',
        help='Find GUI schedule markers in file'
    )
    find_parser.add_argument('file', help='Path to CustomSchedule.py')
    find_parser.add_argument('--yaml', help='Path to YAML file for transpose detection')
    find_parser.add_argument('--transpose', choices=['TN', 'NT', 'NN', 'TT'], 
                            help='Transpose combination override')
    
    # parse command
    parse_parser = subparsers.add_parser(
        'parse',
        help='Parse schedule between markers'
    )
    parse_parser.add_argument('file', help='Path to CustomSchedule.py')
    parse_parser.add_argument('--yaml', help='Path to YAML file for transpose detection')
    parse_parser.add_argument('--transpose', choices=['TN', 'NT', 'NN', 'TT'], 
                             help='Transpose combination override')
    
    # write command
    write_parser = subparsers.add_parser(
        'write',
        help='Write schedule to file'
    )
    write_parser.add_argument('file', help='Path to CustomSchedule.py')
    write_parser.add_argument(
        '--schedule',
        required=True,
        help='Path to JSON file containing schedule data'
    )
    write_parser.add_argument('--yaml', help='Path to YAML file for transpose detection')
    write_parser.add_argument('--transpose', choices=['TN', 'NT', 'NN', 'TT'], 
                             help='Transpose combination override')
    
    # get-schedule-info command
    schedule_info_parser = subparsers.add_parser(
        'get-schedule-info',
        help='Load YAML and get ScheduleInfo from Tensile pipeline'
    )
    schedule_info_parser.add_argument('yaml', help='Path to benchmark config YAML file')
    schedule_info_parser.add_argument('--arch', default='gfx950', help='Target architecture (default: gfx950)')
    schedule_info_parser.add_argument('--compiler', help='Path to C++ compiler (auto-detects if not provided)')
    
    # validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate a schedule using CMSValidator'
    )
    validate_parser.add_argument('yaml', help='Path to benchmark config YAML file')
    validate_parser.add_argument('--compiler', help='Path to C++ compiler (auto-detects if not provided)')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'find-markers':
            result = find_gui_schedule_markers(args.file)
            # Add transpose info if yaml provided
            if hasattr(args, 'yaml') and args.yaml:
                result['transpose'] = get_transpose_from_yaml(args.yaml)
            elif hasattr(args, 'transpose') and args.transpose:
                result['transpose'] = args.transpose
            print(json.dumps(result, indent=2))
            
        elif args.command == 'parse':
            yaml_path = getattr(args, 'yaml', None)
            transpose = getattr(args, 'transpose', None)
            result = parse_schedule(args.file, yaml_path=yaml_path, transpose=transpose)
            # Don't include raw_code in output (too verbose)
            output = {k: v for k, v in result.items() if k != 'raw_code'}
            print(json.dumps(output, indent=2))
            
        elif args.command == 'write':
            with open(args.schedule, 'r') as f:
                schedule_data = json.load(f)
            yaml_path = getattr(args, 'yaml', None)
            transpose = getattr(args, 'transpose', None)
            success = write_schedule(args.file, schedule_data, yaml_path=yaml_path, transpose=transpose)
            print(json.dumps({"success": success}))
            
        elif args.command == 'get-schedule-info':
            has_sched, sched_info, solution_state = get_schedule_info_from_yaml(
                args.yaml,
                cxx_compiler=args.compiler
            )
            output = {
                "has_schedule": has_sched,
                "schedule_info": None,
                "kernel_config": {
                    "MacroTile0": solution_state.get("MacroTile0") if solution_state else None,
                    "MacroTile1": solution_state.get("MacroTile1") if solution_state else None,
                    "DepthU": solution_state.get("DepthU") if solution_state else None,
                    "MatrixInstruction": solution_state.get("MatrixInstruction") if solution_state else None,
                }
            }
            if has_sched and sched_info:
                output["schedule_info"] = {
                    "numCodePaths": sched_info.numCodePaths,
                    "numMfma": sched_info.numMfma,
                    "optSchedule": sched_info.optSchedule,
                    "nglshift": sched_info.nglshift,
                    "nllshift": sched_info.nllshift,
                    "syncCode": _serialize_sync_code(sched_info.optSchedule, sched_info.syncCode),
                    "mfmaReorder": list(sched_info.mfmaReorder) if sched_info.mfmaReorder else [],
                }
            print(json.dumps(output, indent=2))
            
        elif args.command == 'validate':
            import io
            import contextlib
            import Tensile.Components.CMSValidator as cmsv
            
            has_sched, sched_info, solution_state = get_schedule_info_from_yaml(
                args.yaml,
                cxx_compiler=args.compiler
            )
            
            if not has_sched or not sched_info:
                output = {
                    "valid": False,
                    "message": "No custom schedule found for this YAML configuration"
                }
            else:
                # Build context with kernel info (idMap not available without full code generation)
                context = {
                    'kernel': solution_state
                }
                
                # Run validation, capturing any warning output
                warning_capture = io.StringIO()
                with contextlib.redirect_stdout(warning_capture):
                    is_valid, message = cmsv.isValid(sched_info, context)
                
                # Include captured warnings in the message if any
                warnings = warning_capture.getvalue().strip()
                if warnings and not message:
                    message = warnings
                elif warnings and message:
                    message = f"{message}\nWarnings: {warnings}"
                
                output = {
                    "valid": is_valid,
                    "message": message if message else ("Schedule is valid" if is_valid else "Validation failed")
                }
            
            print(json.dumps(output, indent=2))
            
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
