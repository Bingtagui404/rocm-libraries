################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import argparse
import csv
import os
import re
import sys
from itertools import chain
from dataclasses import dataclass, field
from enum import Enum, auto
from glob import glob
from typing import List, Tuple, Dict, Optional, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from Tensile.Components.CustomSchedule import ScheduleInfo
from CMSScheduleIO import get_schedule_info_from_yaml


class InstructionClass(Enum):
    """Classification of GPU assembly instructions by type."""
    BARRIER = auto()    # Synchronization barrier (s_barrier)
    BRANCH = auto()     # Control flow (s_branch, s_cbranch_*)
    LDS = auto()        # Local (LDS) reads (ds_read_*)
    MFMA = auto()       # Matrix FMA operations (v_mfma_*)
    NOP = auto()        # No operation (s_nop*)
    OTHER = auto()      # Unclassified instructions
    SALU = auto()       # Other scalar ALU (s_*)
    VALU = auto()       # Other vector ALU (v_*)
    VMEM = auto()       # Global memory loads/stores (buffer_load_*, global_load_*, buffer_store_*, global_store_*)
    WAITCNT = auto()    # Wait for counters (s_waitcnt*)

@dataclass
class LoopRanges:
    """
    Ranges of line numbers (inclusive on both ends) into the CSV files for which lines map to the different loops.
    It is assumed, given that they all refer to the same code, that the ranges are the same for all CSV files.
    """
    # If there is SIMD specialization, there will be one mainloop for each SIMD path.
    mainloops: list[tuple[int, int]]
    # The NGL loop is the second-last MAINLOOP invocation.
    ngl: tuple[int, int]
    # The NLL loop is the last MAINLOOP invocation.
    nll: tuple[int, int]

@dataclass
class LoopData:
    """Unified loop data structure for both SIMD and non-SIMD kernels."""
    # Map between loop name and list of (id, instruction, class) tuples
    instructions_by_loop: Dict[str, List[Tuple[int, str, InstructionClass]]]
    # List of mainloop names (e.g., ['mainloop'] or ['mainloop_simd_0', ...])
    mainloop_names: List[str]
    # Map between loop name and line number (for SIMD CSV matching), None for non-SIMD
    loop_line_mapping: Optional[Dict[str, int]]


CYCLES_FOR_MFMA = {
    "v_mfma_f32_32x32x4_2b_bf16": 64,
    "v_mfma_f32_16x16x4_4b_bf16": 32,
    "v_mfma_f32_4x4x4_16b_bf16": 8,
    "v_mfma_f32_32x32x8_bf16": 32,
    "v_mfma_f32_16x16x16_bf16": 16,
    "v_mfma_f32_16x16x32_bf16": 16,
    "v_mfma_f32_32x32x16_bf16": 32,
}


def classify_instruction(instruction: str) -> InstructionClass:
    """
    Classify an instruction by its mnemonic prefix.
    
    Args:
        instruction: Full instruction text (e.g., "ds_read_b32 v[vgprValuA...] // comment")
        
    Returns:
        The InstructionClass for this instruction
    """
    # Get the mnemonic (first word, lowercase)
    inst = instruction.split()[0].lower()
    
    # Order matters: check more specific patterns first
    if inst.startswith("v_mfma_"):
        return InstructionClass.MFMA
    if inst.startswith("ds_"):
        return InstructionClass.LDS
    if inst.startswith("buffer_load") or inst.startswith("global_load") or inst.startswith("buffer_store") or inst.startswith("global_store"):
        return InstructionClass.VMEM
    if inst.startswith("s_waitcnt"):
        return InstructionClass.WAITCNT
    if inst == "s_barrier":
        return InstructionClass.BARRIER
    if inst.startswith("s_nop"):
        return InstructionClass.NOP
    if inst.startswith("s_branch") or inst.startswith("s_cbranch"):
        return InstructionClass.BRANCH
    if inst.startswith("s_"):
        return InstructionClass.SALU
    if inst.startswith("v_"):
        return InstructionClass.VALU
    return InstructionClass.OTHER


INSTRUCTION_CLASS_COLORS = {
    InstructionClass.MFMA: "#0A5B0C",       # Dark Green
    InstructionClass.LDS: "#FF7F00",        # Orange
    InstructionClass.VMEM: "#E6D700",       # Yellow
    InstructionClass.WAITCNT: "#000000",    # Black
    InstructionClass.BARRIER: "#999999",    # Grey
    InstructionClass.NOP: "#999999",        # Grey
    InstructionClass.SALU: "#A6FFD9",       # Light Mint
    InstructionClass.VALU: "#118C13",       # Green
    InstructionClass.BRANCH: "#E78AC3",     # Magenta
    InstructionClass.OTHER: "#A6D854",      # Lime
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze mainloop cycle counts from ATT profiling CSV files."
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help="Partial name to match CSV files (e.g., 'stats_ui_output_agent_38698')",
    )
    parser.add_argument(
        "--yaml",
        required=True,
        help="Path to the Tensile YAML config file",
    )
    parser.add_argument(
        "--tensile-dir",
        required=True,
        help="Directory with 1_BenchmarkProblems/, 2_BenchmarkData/, etc.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--output-latency-json",
        action="store_true",
        help="Output latency_dict.json with semantic labels mapped to median latencies for the main loop.",
    )
    parser.add_argument(
        "--output-excess-latency-table",
        action="store_true",
        help="Output excess_latency_table.txt showing instructions with latency above baseline.",
    )
    return parser.parse_args()


def find_assembly_file(tensile_dir: str) -> str:
    benchmark_dir = os.path.join(tensile_dir, "1_BenchmarkProblems")
    if not os.path.isdir(benchmark_dir):
        raise FileNotFoundError(f"1_BenchmarkProblems directory not found in {tensile_dir}")
    
    # Search recursively for .s files
    s_files = glob(os.path.join(benchmark_dir, "**", "*.s"), recursive=True)
    
    if not s_files:
        raise FileNotFoundError(f"No .s assembly file found in {benchmark_dir}")
    
    if len(s_files) > 1:
        raise ValueError(f"Multiple .s files found in {benchmark_dir}: {s_files}")
    
    return s_files[0]


def is_comment_or_empty(line: str) -> bool:
    """
    Check if a line is a comment, empty, or a label.
    
    Args:
        line: Line of assembly code
        
    Returns:
        True if the line should be skipped (comment/empty/label)
    """
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("/*"):
        return True
    if stripped.startswith("//"):
        return True
    if stripped.endswith(":") and not " " in stripped:
        # Labels like "label_name:" but not instructions with comments ending in ":"
        return True
    return False

def find_loop_ranges(asm_file: str) -> Dict[str, Tuple[int, int, List[Tuple[int, str, InstructionClass]]]]:
    """
    Find the loop ranges for main loop, NGL (NoGlobalLoad) loop, and NLL (NoLoad) loop.
    
    The kernel has three stages:
    - Main loop: runs for K // depthU - 2 iterations (has global loads, local writes, local reads, MFMA)
    - NGL (NoGlobalLoad) loop: runs for 1 iteration (no global loads, has local writes, local reads, MFMA)
    - NLL (NoLoad) loop: runs for 1 iteration (no loads at all, just local reads and MFMA)
    
    Args:
        asm_file: Path to the .s assembly file
        
    Returns:
        Dictionary with keys 'mainloop', 'ngl', 'nll', each containing:
        (start_line, end_line, instructions_list) where instructions_list contains
        (line_number, instruction_text, instruction_class) tuples
    """
    # Markers for the different loop stages
    mainloop_start_marker = "mfmaIndex:0"
    mainloop_end_marker = "label_LoopEndL:"
    ngl_start_marker = "/* Ord. NoGlobalLoadLoop - Begin"
    ngl_end_marker = "label_toPGR1:"
    nll_start_marker = "/* Ord. NoLoadLoop - Begin"
    nll_end_marker = "label_toPGR1end_OrdNLL:"
    
    with open(asm_file, "r") as f:
        lines = f.readlines()
    
    def find_marker(marker: str, start_from: int = 0) -> Optional[int]:
        """Find line index containing marker."""
        normalized_marker = ' '.join(marker.split())
        for i, line in enumerate(lines[start_from:], start=start_from):
            normalized_line = ' '.join(line.split())
            if normalized_marker in normalized_line:
                return i
        return None
    
    def find_first_instruction_after(start_idx: int, end_idx: int) -> Optional[int]:
        """Find first non-comment line after start_idx."""
        for i in range(start_idx + 1, end_idx):
            if not is_comment_or_empty(lines[i]):
                return i + 1  # 1-indexed
        return None
    
    def find_last_instruction_before(end_idx: int, start_idx: int) -> Optional[int]:
        """Find last non-comment line before end_idx."""
        for i in range(end_idx - 1, start_idx, -1):
            if not is_comment_or_empty(lines[i]):
                return i + 1  # 1-indexed
        return None
    
    def collect_instructions(start_line: int, end_line: int) -> List[Tuple[int, str, InstructionClass]]:
        """Collect all non-comment instructions with their class in range (1-indexed)."""
        instructions = []
        for i in range(start_line - 1, end_line):  # Convert to 0-indexed
            line = lines[i]
            if not is_comment_or_empty(line):
                inst_text = line.strip()
                inst_class = classify_instruction(inst_text)
                instructions.append((i + 1, inst_text, inst_class))  # Store 1-indexed
        return instructions
    
    results = {}
    
    # Find main loop range
    mainloop_start_idx = find_marker(mainloop_start_marker)
    if mainloop_start_idx is None:
        raise ValueError(f"Main loop start marker '{mainloop_start_marker}' not found")
    
    mainloop_end_idx = find_marker(mainloop_end_marker, mainloop_start_idx)
    if mainloop_end_idx is None:
        raise ValueError(f"Main loop end marker '{mainloop_end_marker}' not found")
    
    mainloop_start = find_first_instruction_after(mainloop_start_idx, mainloop_end_idx)
    mainloop_end = find_last_instruction_before(mainloop_end_idx, mainloop_start_idx)
    
    if mainloop_start and mainloop_end:
        results['mainloop'] = (mainloop_start, mainloop_end, 
                               collect_instructions(mainloop_start, mainloop_end))
    else:
        raise ValueError("Could not find main loop instruction boundaries")
    
    # Find NGL (NoGlobalLoad) loop range
    ngl_start_idx = find_marker(ngl_start_marker, mainloop_end_idx)
    if ngl_start_idx is None:
        raise ValueError(f"NGL loop start marker '{ngl_start_marker}' not found")
    
    ngl_end_idx = find_marker(ngl_end_marker, ngl_start_idx)
    if ngl_end_idx is None:
        raise ValueError(f"NGL loop end marker '{ngl_end_marker}' not found")
    
    ngl_start = find_first_instruction_after(ngl_start_idx, ngl_end_idx)
    ngl_end = find_last_instruction_before(ngl_end_idx, ngl_start_idx)
    
    if ngl_start and ngl_end:
        results['ngl'] = (ngl_start, ngl_end, collect_instructions(ngl_start, ngl_end))
    else:
        raise ValueError("Could not find NGL loop instruction boundaries")
    
    # Find NLL (NoLoad) loop range
    nll_start_idx = find_marker(nll_start_marker, ngl_end_idx)
    if nll_start_idx is None:
        raise ValueError(f"NLL loop start marker '{nll_start_marker}' not found")
    
    nll_end_idx = find_marker(nll_end_marker, nll_start_idx)
    if nll_end_idx is None:
        raise ValueError(f"NLL loop end marker '{nll_end_marker}' not found")
    
    nll_start = find_first_instruction_after(nll_start_idx, nll_end_idx)
    nll_end = find_last_instruction_before(nll_end_idx, nll_start_idx)
    
    if nll_start and nll_end:
        results['nll'] = (nll_start, nll_end, collect_instructions(nll_start, nll_end))
    else:
        raise ValueError("Could not find NLL loop instruction boundaries")
    
    return results


def find_loop_ranges_simd_specialized(asm_file: str, num_simd_paths: int) -> Dict[str, int]:
    """
    Find the MAINLOOP invocation line numbers for SIMD specialized kernels.
    
    In SIMD specialized kernels:
    - Each SIMD path has its own mainloop (invoked via MAINLOOP macro)
    - NGL and NLL are the second-last and last MAINLOOP invocations
    
    Args:
        asm_file: Path to the .s assembly file
        num_simd_paths: Number of SIMD paths (2 or 4)
        
    Returns:
        Dictionary mapping loop names to their invocation line numbers:
        - 'mainloop_simd_0', 'mainloop_simd_1', etc. for each SIMD mainloop
        - 'ngl' for the NoGlobalLoad loop
        - 'nll' for the NoLoad loop
    """
    with open(asm_file, "r") as f:
        lines = f.readlines()
    
    # Find all MAINLOOP invocation lines (line number is 1-indexed)
    mainloop_lines = []
    for i, line in enumerate(lines):
        # Match lines that invoke the MAINLOOP macro (not the macro definition)
        stripped = line.strip()
        if stripped.startswith("MAINLOOP "):
            mainloop_lines.append(i + 1)  # 1-indexed
    
    if len(mainloop_lines) < num_simd_paths + 2:
        raise ValueError(
            f"Expected at least {num_simd_paths + 2} MAINLOOP invocations "
            f"(for {num_simd_paths} SIMDs + NGL + NLL), but found {len(mainloop_lines)}"
        )
    
    result = {}
    
    # Find SIMD code path comments and map to their mainloop lines
    for simd_idx in range(num_simd_paths):
        simd_marker = f"SIMD {simd_idx} code-path"
        marker_line = None
        
        for i, line in enumerate(lines):
            if simd_marker in line:
                marker_line = i
                break
        
        if marker_line is None:
            raise ValueError(f"Could not find '{simd_marker}' in assembly")
        
        # Find the first MAINLOOP after this marker
        for mainloop_line in mainloop_lines:
            if mainloop_line > marker_line:
                result[f'mainloop_simd_{simd_idx}'] = mainloop_line
                break
        else:
            raise ValueError(f"Could not find MAINLOOP after '{simd_marker}'")
    
    # NGL is the second-last MAINLOOP, NLL is the last MAINLOOP
    result['ngl'] = mainloop_lines[-2]
    result['nll'] = mainloop_lines[-1]
    
    return result


def find_all_loop_ranges(
    asm_file: str,
    csv_file: str, 
    num_code_paths: int
) -> LoopRanges:
    """
    Find loop ranges as CSV row indices for all loop types.
    
    Scans a CSV file to determine which rows belong to each loop type (mainloop(s),
    NGL, NLL) based on the assembly line numbers in the Source column.
    
    Args:
        asm_file: Path to the .s assembly file
        csv_file: Path to a CSV file with profiling data (must have Source column)
        num_code_paths: Number of code paths (0 for non-SIMD, typically 2 or 4 for SIMD)
        
    Returns:
        LoopRanges dataclass with CSV row indices (0-indexed, inclusive):
        - mainloops: List of (start_row, end_row) tuples for each mainloop variant
        - ngl: (start_row, end_row) tuple for the NoGlobalLoad loop
        - nll: (start_row, end_row) tuple for the NoLoad loop
        
    Note:
        Row indices are 0-indexed (row 0 is the first data row after the header).
        All CSV files for the same kernel should have identical row ranges since
        they trace the same code.
    """
    if num_code_paths > 0:
        line_mapping = find_loop_ranges_simd_specialized(asm_file, num_code_paths)
        # line_mapping: {'mainloop_simd_0': line, ..., 'ngl': line, 'nll': line}
        
        # Reverse mapping: assembly line -> loop name
        asm_line_to_loop = {line: name for name, line in line_mapping.items()}
        
        # Define loop names in order
        mainloop_names = [f'mainloop_simd_{i}' for i in range(num_code_paths)]
        
        def get_loop_for_asm_line(asm_line: int) -> Optional[str]:
            return asm_line_to_loop.get(asm_line)
    else:
        loop_ranges = find_loop_ranges(asm_file)
        # loop_ranges: {'mainloop': (start, end, instrs), 'ngl': ..., 'nll': ...}
        
        mainloop_names = ['mainloop']
        
        def get_loop_for_asm_line(asm_line: int) -> Optional[str]:
            for loop_name in ['mainloop', 'ngl', 'nll']:
                start, end, _ = loop_ranges[loop_name]
                if start <= asm_line <= end:
                    return loop_name
            return None
    
    # Track first and last row index for each loop
    # Initialize with None to detect first occurrence
    loop_first_row: Dict[str, Optional[int]] = {name: None for name in mainloop_names + ['ngl', 'nll']}
    loop_last_row: Dict[str, Optional[int]] = {name: None for name in mainloop_names + ['ngl', 'nll']}
    
    # Scan CSV file to find row ranges
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            source = row.get("Source", "")
            asm_line = extract_line_number_from_source(source)
            
            if asm_line is None:
                continue
                
            loop_name = get_loop_for_asm_line(asm_line)
            if loop_name is None:
                continue
            
            # Update first/last row tracking
            if loop_first_row[loop_name] is None:
                loop_first_row[loop_name] = row_idx
            loop_last_row[loop_name] = row_idx
    
    # Build mainloops list
    mainloops = []
    for name in mainloop_names:
        first = loop_first_row[name]
        last = loop_last_row[name]
        if first is None or last is None:
            raise ValueError(f"Could not find any CSV rows for loop '{name}'")
        mainloops.append((first, last))
    
    # Get NGL and NLL ranges
    ngl_first, ngl_last = loop_first_row['ngl'], loop_last_row['ngl']
    nll_first, nll_last = loop_first_row['nll'], loop_last_row['nll']
    
    if ngl_first is None or ngl_last is None:
        raise ValueError("Could not find any CSV rows for NGL loop")
    if nll_first is None or nll_last is None:
        raise ValueError("Could not find any CSV rows for NLL loop")
    
    return LoopRanges(
        mainloops=mainloops,
        ngl=(ngl_first, ngl_last),
        nll=(nll_first, nll_last),
    )


def parse_yaml_for_mfma(yaml_path: str) -> Tuple[int, int]:
    """
    Parse the YAML config file to extract MFMA instruction type, cycles, loop info.
    
    Args:
        yaml_path: Path to the Tensile YAML config file
        
    Returns:
        Tuple of (cycles_per_mfma, total_iterations)
        - cycles_per_mfma: Cycles per MFMA instruction
        - total_iterations: Total number of loop iterations K // depthU
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Find MatrixInstruction and DepthU in ForkParameters
    matrix_inst = None
    depth_u = None
    problem_k = None
    
    for problem_group in config.get("BenchmarkProblems", []):
        for item in problem_group:
            if isinstance(item, dict):
                if "ForkParameters" in item:
                    for param in item["ForkParameters"]:
                        if isinstance(param, dict):
                            if "MatrixInstruction" in param:
                                # Get the first MatrixInstruction configuration
                                matrix_inst = param["MatrixInstruction"][0]
                            if "DepthU" in param:
                                # Get the first DepthU value
                                depth_u = param["DepthU"][0]
                
                if "BenchmarkFinalParameters" in item:
                    for param in item["BenchmarkFinalParameters"]:
                        if isinstance(param, dict) and "ProblemSizes" in param:
                            for size_spec in param["ProblemSizes"]:
                                if isinstance(size_spec, dict) and "Exact" in size_spec:
                                    # Exact format: [M, N, Batch, K]
                                    exact = size_spec["Exact"]
                                    if len(exact) >= 4:
                                        problem_k = exact[3]
                                    break
        
    if matrix_inst is None:
        raise ValueError(f"MatrixInstruction not found in {yaml_path}")
    if depth_u is None:
        raise ValueError(f"DepthU not found in {yaml_path}")
    if problem_k is None:
        raise ValueError(f"ProblemSizes with Exact K dimension not found in {yaml_path}")
    
    # Extract M, N, K dimensions (first 3 values)
    m, n, k = matrix_inst[0], matrix_inst[1], matrix_inst[2]
    dim_pattern = f"_{m}x{n}x{k}_"
    
    # Search for matching MFMA instruction by dimensions
    mfma_name = None
    for name in CYCLES_FOR_MFMA:
        if dim_pattern in name:
            mfma_name = name
            break
    if mfma_name is None:
        raise ValueError(f"Unknown MatrixInstruction dimensions: ({m}, {n}, {k})")
    cycles_per_mfma = CYCLES_FOR_MFMA[mfma_name]
    
    assert problem_k % depth_u == 0, f"K ({problem_k}) is not divisible by depthU ({depth_u}), case not supported."
    total_iterations = problem_k // depth_u

    return cycles_per_mfma, total_iterations


def gather_csv_files(csv_dir: str, pattern: str) -> List[str]:
    """
    Find CSV files matching the pattern.
    
    Args:
        csv_dir: Directory containing CSV files
        pattern: Partial name to match
        
    Returns:
        List of paths to matching CSV files
    """
    csv_files = []
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv") and pattern in filename:
            csv_files.append(os.path.join(csv_dir, filename))
    csv_files.sort()

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found matching the pattern {pattern} in {csv_dir}")
    
    return csv_files


def extract_line_number_from_source(source: str) -> Optional[int]:
    """
    Extract the line number from the Source column.
    
    Args:
        source: Source column value (e.g., ".../file.s:1898")
        
    Returns:
        Line number or None if not parseable
    """
    # Source format: "/path/to/file.s:1234"
    match = re.search(r":(\d+)$", source)
    if match:
        return int(match.group(1))
    return None


def prepare_histogram_data(
    cycle_data: np.ndarray,
    instructions: List[Tuple[int, str, InstructionClass]],
    semantic_labels: Dict[int, str],
    total_iterations: int,
) -> tuple[np.ndarray, list[tuple[int, str, InstructionClass]], dict[int, str], np.ndarray]:
    """
    Preprocess histogram data by filtering and computing y-values.
    
    Args:
        cycle_data: Array of shape [num_instructions, num_csv_files]
        instructions: List of (line_number, instruction_text, instruction_class) tuples
        semantic_labels: Dictionary mapping instruction id to semantic label
        total_iterations: Total number of loop iterations K // depthU
        filter_zero_excess: If True, filter out instructions with zero excess latency
        baseline_cycles: Baseline cycle count for excess latency calculation
        
    Returns:
        Tuple of (filtered_data, filtered_instructions, filtered_labels, y_values)
    """
    filtered_data = cycle_data
    filtered_instructions = instructions
    filtered_labels = semantic_labels.copy() if semantic_labels else {}
    
    # Compute y-values (mean cycles per iteration)
    mean_cycles_per_iter = np.mean(filtered_data, axis=1) / total_iterations
    return filtered_data, filtered_instructions, filtered_labels, mean_cycles_per_iter


@dataclass
class Instruction:
    """
    In optSchedule, instructions are scheduled as name: [[indices_for_codepath_0], [indices_for_codepath_1], ...]
    E.g. LRA0: [[10, 20, 30], [15, 25, 35]], LRA0[0][1] == 20 "The 2nd LRA0 for codepath 0 is at MFMA index 20."
    """
    # PackA3, SYNC, etc.
    name: str
    # Which codepath this is for: 0, 1, 2, or 4
    codepath: int
    # Index into the list of MFMA indices for this instruction and codepath
    idx: int
    # MFMA index this instruction was scheduled at.
    mfma_index: int
    # Issue latencies for this instruction (including stall + actual issue time) in cycles.
    # Values are averaged across the SIMDs. they are issued on.
    latencies: list[float] = field(default_factory=list)
    # Classification of the instruction (MFMA, LDS, VMEM, etc.)
    instruction_class: InstructionClass = InstructionClass.OTHER

    def __str__(self) -> str:
        return f"{self.name}[{self.idx}]"

@dataclass
class Schedule:
    """
    """
    codepaths: List[List[Instruction]]
    num_mfma: int
    # Latencies are the average across SIMDs.
    nll_latencies: list[float] = field(default_factory=list)
    ngl_latencies: list[float] = field(default_factory=list)

    @property
    def num_code_paths(self) -> int:
        return len(self.codepaths)


def schedule_from_yaml(yaml_path: str) -> Schedule:
    _, schedule_info, _ = get_schedule_info_from_yaml(yaml_path)

    codepaths = []
    for i_c in range(schedule_info.numCodePaths):
        codepath = [[]]  # Start with empty list for the -1 location.
        for i_mfma in range(schedule_info.numMfma):
            codepath.append([Instruction(name="MFMA", codepath=i_c, idx=i_mfma, mfma_index=i_mfma)])
        codepaths.append(codepath)

    for name, codepath_lists in schedule_info.optSchedule.items():
        # If an instruction type has only 1 codepath list but there are multiple
        # codepaths, the same schedule applies to all codepaths.
        if len(codepath_lists) == 1 and schedule_info.numCodePaths > 1:
            codepath_lists = codepath_lists * schedule_info.numCodePaths
        
        for codepath, mfma_indices in enumerate(codepath_lists):
            for i, mfma_index in enumerate(mfma_indices):
                # NOTE: +1 to handle the -1 location.
                codepaths[codepath][mfma_index+1].append(Instruction(name=name, codepath=codepath, idx=i, mfma_index=mfma_index))

    codepaths = [list(chain.from_iterable(codepath)) for codepath in codepaths]
    return Schedule(codepaths=codepaths, num_mfma=schedule_info.numMfma)


def fill_schedule_with_latencies(
    schedule: Schedule,
    loop_ranges: LoopRanges,
    csv_files: List[str]
) -> None:
    """
    Fill the schedule with latency data from profiled CSV files.
    
    Maps instructions to CSV rows by position (1:1 match in execution order):
    - schedule.codepaths[i][j] maps to CSV row loop_ranges.mainloops[i].start + j
    - NGL/NLL latencies are summed from their respective row ranges
    
    Also classifies instructions using the assembly text from the first CSV file.
    
    Args:
        schedule: The Schedule object with codepaths containing Instruction objects
        loop_ranges: LoopRanges object with CSV row indices for each loop section
        csv_files: List of CSV file paths containing profiled latency data
    """
    for csv_idx, csv_path in enumerate(csv_files):
        # Read latencies and instruction text from the CSV file
        latencies_by_row: Dict[int, int] = {}
        instructions_by_row: Dict[int, str] = {}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                latency = int(row.get("Latency", "0").strip())
                latencies_by_row[row_idx] = latency
                # Only need instruction text for classification on first file
                if csv_idx == 0:
                    instructions_by_row[row_idx] = row.get("Instruction", "").strip()
        
        # Fill latencies for each codepath's mainloop instructions
        for codepath_idx, (start_row, end_row) in enumerate(loop_ranges.mainloops):
            num_rows = end_row - start_row + 1
            num_instructions = len(schedule.codepaths[codepath_idx])
            
            if num_rows != num_instructions:
                raise ValueError(
                    f"Mismatch for codepath {codepath_idx}: "
                    f"{num_rows} CSV rows (rows {start_row}-{end_row}) vs "
                    f"{num_instructions} schedule instructions"
                )
            
            simds_per_codepath = 4 // schedule.num_code_paths
            for inst_idx, instruction in enumerate(schedule.codepaths[codepath_idx]):
                row_idx = start_row + inst_idx
                instruction.latencies.append(float(latencies_by_row[row_idx]) / simds_per_codepath)
                # Classify instruction on first CSV file only
                if csv_idx == 0:
                    instruction_text = instructions_by_row[row_idx]
                    instruction.instruction_class = classify_instruction(instruction_text)
        
        # Sum NGL latencies
        ngl_start, ngl_end = loop_ranges.ngl
        ngl_total = sum(
            latencies_by_row[row_idx]
            for row_idx in range(ngl_start, ngl_end + 1)
        )
        # Divide by 4 to get the average latency per SIMD.
        schedule.ngl_latencies.append(float(ngl_total) / 4)
        
        # Sum NLL latencies
        nll_start, nll_end = loop_ranges.nll
        nll_total = sum(
            latencies_by_row[row_idx]
            for row_idx in range(nll_start, nll_end + 1)
        )
        # Divide by 4 to get the average latency per SIMD.
        schedule.nll_latencies.append(float(nll_total) / 4)
    
    for codepath in schedule.codepaths:
        for instruction in codepath:
            assert len(instruction.latencies) > 0, f"No latencies for instruction {instruction}"


def generate_boxplot(
    cycle_data: np.ndarray,
    instructions: List[Tuple[int, str, InstructionClass]],
    efficiency: np.ndarray,
    output_path: str,
    title: str,
    y_label: str,
    num_runs: int,
    total_iterations: int,
    y_max: Optional[float] = None,
    semantic_labels: Optional[Dict[int, str]] = None,
    baseline_cycles: Optional[float] = None
) -> str:
    """
    Generate and save a histogram with box plots overlaid showing distribution across runs.
    
    Args:
        cycle_data: Array of shape [num_instructions, num_csv_files] with raw cycle counts
        instructions: List of (line_number, instruction_text, instruction_class) tuples
        efficiency: Array of efficiency percentages
        output_path: Full path to save the plot
        title: Main title text for the plot
        y_label: Label for the y-axis
        num_runs: Number of CSV files (runs) used to calculate the average
        total_iterations: Total number of loop iterations (K // depthU)
        y_max: Optional maximum value for y-axis (for consistent scaling across plots)
        semantic_labels: Optional dictionary mapping instruction id to semantic label
        baseline_cycles: If provided, subtract this baseline and clip to 0 (for excess latency)
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    total_instructions = len(instructions)
    
    if total_instructions == 0:
        # Create empty plot for consistency
        fig, ax = plt.subplots(figsize=(30, 10))
        fig.patch.set_facecolor('#505050')
        ax.set_facecolor('#505050')
        ax.set_title(f"{title}\nNo data to display", color='white', fontweight='bold')
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
        plt.close()
        return output_path
    
    fig, ax = plt.subplots(figsize=(30, 10))
    
    # Set middle grey background
    fig.patch.set_facecolor('#505050')
    ax.set_facecolor('#505050')
    
    # Normalize cycle data to per-iteration values
    # cycle_data shape: [num_instructions, num_csv_files]
    data_per_iter = cycle_data / total_iterations
    
    # Apply baseline subtraction if requested (for excess latency)
    if baseline_cycles is not None:
        data_per_iter = np.maximum(0, data_per_iter - baseline_cycles)
    
    # Calculate median values for the histogram bars (matches box plot median)
    y_values = np.median(data_per_iter, axis=1)
    
    # Create bar positions (0-indexed)
    x_positions = np.arange(total_instructions)
    
    # Get colors for each instruction based on its class
    colors = [INSTRUCTION_CLASS_COLORS[inst_class] for _, _, inst_class in instructions]
    
    # Create histogram with one bar per instruction, colored by class
    ax.bar(x_positions, y_values, width=0.95, color=colors, alpha=0.7)
    
    # Prepare data for box plot - list of arrays, one per instruction
    box_data = [data_per_iter[i, :] for i in range(total_instructions)]
    
    # Overlay box plot on top of histogram
    bp = ax.boxplot(box_data, positions=x_positions, widths=0.5, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='o', markersize=2, alpha=0.6,
                                                      markerfacecolor='white', markeredgecolor='white'))
    
    # Make box plot boxes transparent with white edges
    for patch in bp['boxes']:
        patch.set_facecolor('none')
        patch.set_edgecolor('white')
        patch.set_linewidth(1.5)
    
    # Style the box plot elements for visibility on grey background
    for element in ['whiskers', 'caps']:
        for item in bp[element]:
            item.set_color('white')
            item.set_linewidth(1.2)
    
    # Style medians to match their corresponding bar color
    for median, color in zip(bp['medians'], colors):
        median.set_color(color)
        median.set_linewidth(2)
    
    # Set consistent y-axis limit if provided
    if y_max is not None:
        ax.set_ylim(0, y_max)
    
    # Set white, bold text for labels and title
    ax.set_xlabel("Instruction", color='white', fontweight='bold')
    ax.set_ylabel(y_label, color='white', fontweight='bold')
    ax.set_title(
        f"{title}\n"
        f"Mean Efficiency: {np.mean(efficiency):.1f}% | "
        f"Min: {np.min(efficiency):.1f}% | Max: {np.max(efficiency):.1f}% | "
        f"Runs: {num_runs}\n",
        color='white', fontweight='bold'
    )
    
    # Set white tick labels
    ax.tick_params(axis='both', colors='white')
    
    # Set white spines
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # Add legend showing instruction classes present in the data
    present_classes = sorted(set(inst_class for _, _, inst_class in instructions), key=lambda x: x.name)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=INSTRUCTION_CLASS_COLORS[cls], alpha=0.7)
        for cls in present_classes
    ]
    # Add box plot legend entry
    legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc='none', ec='white', linewidth=1.5))
    legend_labels = [cls.name for cls in present_classes] + ['Distribution (box plot)']
    
    legend = ax.legend(legend_handles, legend_labels,
                       loc='upper right', fontsize=8, title="Instruction Class",
                       facecolor='#606060', edgecolor='white')
    legend.get_title().set_color('white')
    legend.get_title().set_fontweight('bold')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Add box plot explanation text box below the legend
    boxplot_info = (
        "Box Plot Components:\n"
        "─ White box: IQR (25th-75th percentile)\n"
        "─ Colored line: Median\n"
        "─ Whiskers: 1.5× IQR from box edges\n"
        "─ White dots: Outliers"
    )
    # Position in upper right, below the legend
    ax.text(0.005, 0.98, boxplot_info, transform=ax.transAxes,
            fontsize=7, verticalalignment='top', horizontalalignment='left',
            color='white', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#606060', 
                      edgecolor='white', alpha=0.9))
    
    # Set x-ticks at every instruction position with semantic labels if available
    ax.set_xticks(x_positions)
    if semantic_labels:
        x_labels = [semantic_labels.get(inst_id, str(inst_id)) for inst_id, _, _ in instructions]
    else:
        x_labels = [str(inst_id) for inst_id, _, _ in instructions]
    ax.set_xticklabels(x_labels, rotation=90, ha='center', fontsize=6)
    
    # Make MFMA labels bold to highlight them
    for tick_label in ax.get_xticklabels():
        label_text = tick_label.get_text()
        if label_text.startswith('MFMA'):
            tick_label.set_fontweight('bold')
        else:
            tick_label.set_fontweight('normal')
    
    # Add padding on x-axis to create more spacing between labels
    if total_instructions > 0:
        ax.set_xlim(-0.5, total_instructions - 0.5)
    
    # Adjust layout to prevent label cutoff with extra bottom padding for rotated labels
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    
    return output_path



def generate_excess_latency_table(
    latency_dict: Dict[str, Dict[str, Dict[str, float]]],
    baseline_cycles: float = 4.0,
    output_path: Optional[str] = None,
    top_n: int = 10,
    exclude_mfma: bool = True
) -> str:
    """
    Generate a text table of instructions with excess latency.
    
    Args:
        latency_dict: Dictionary with structure from export_latency_dict():
            {"codepath_0": {"LRA0[0]": {"whisker_low": ..., "q1": ..., "median": ..., 
                                         "q3": ..., "whisker_high": ...}, ...}, ...}
        baseline_cycles: Baseline cycle count threshold (default 4.0)
        output_path: Optional path to save the table (if None, returns string only)
        top_n: Number of top excess latency instructions to show per codepath (default 10)
        exclude_mfma: If True, exclude MFMA instructions from the table (default True)
        
    Returns:
        Formatted string table of excess latency instructions
    """
    lines = []
    lines.append(f"Excess Latency Report (threshold: {baseline_cycles} cycles)")
    lines.append("=" * 80)
    
    for codepath_key in sorted(latency_dict.keys()):
        codepath_data = latency_dict[codepath_key]
        
        # Collect instructions with excess latency
        excess_data = []
        for label, stats in codepath_data.items():
            # Skip MFMA instructions if requested
            if exclude_mfma and label.startswith("MFMA"):
                continue
            
            median = stats["median"]
            excess = max(0, median - baseline_cycles)
            
            if excess > 0:
                iqr = stats["q3"] - stats["q1"]
                excess_data.append({
                    'label': label,
                    'median': median,
                    'excess': excess,
                    'iqr': iqr
                })
        
        # Sort by excess latency (descending)
        excess_data.sort(key=lambda x: x['excess'], reverse=True)
        
        # Limit to top N entries
        displayed_data = excess_data[:top_n] if top_n else excess_data
        
        # Build section for this codepath
        lines.append("")
        lines.append(f"--- {codepath_key} (showing top {len(displayed_data)} of {len(excess_data)}) ---")
        lines.append("")
        
        if not excess_data:
            lines.append("No instructions with excess latency found.")
        else:
            # Header
            header = f"{'Label':<20} | {'Median':>10} | {'Excess':>10} | {'IQR':>10}"
            lines.append(header)
            lines.append("-" * len(header))
            
            # Data rows (limited to top N)
            for row in displayed_data:
                line = f"{row['label']:<20} | {row['median']:>10.2f} | {row['excess']:>10.2f} | {row['iqr']:>10.2f}"
                lines.append(line)
            
            lines.append("")
            lines.append(f"Total instructions with excess latency: {len(excess_data)}")
            lines.append(f"Total excess cycles per iteration: {sum(r['excess'] for r in excess_data):.2f}")
    
    table_str = "\n".join(lines)
    
    # Save to file if path provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(table_str)
    
    return table_str


def create_latency_dict(
    schedule: Schedule,
    total_iterations: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Create a latency dictionary with box plot statistics for each instruction.
    
    Args:
        schedule: Schedule object containing codepaths with Instruction objects
        total_iterations: Total number of loop iterations (for per-iteration values)
        
    Returns:
        Dictionary with structure:
        {
            "codepath_0": {
                "LRA0[0]": {
                    "whisker_low": 3.8,
                    "q1": 4.0,
                    "median": 4.2,
                    "q3": 4.5,
                    "whisker_high": 4.9
                },
                ...
            },
            ...
        }
    """
    result = {}
    
    for codepath_idx, instructions in enumerate(schedule.codepaths):
        codepath_key = f"codepath_{codepath_idx}"
        result[codepath_key] = {}
        
        for instruction in instructions:
            label = str(instruction)  # Returns "LRA0[0]", "MFMA[0]", etc.
            
            # Convert to per-iteration values
            latencies = np.array(instruction.latencies) / total_iterations
            
            # Calculate quartiles
            q1 = float(np.percentile(latencies, 25))
            median = float(np.median(latencies))
            q3 = float(np.percentile(latencies, 75))
            
            # Calculate IQR and whisker bounds (1.5 * IQR rule)
            iqr = q3 - q1
            whisker_low_bound = q1 - 1.5 * iqr
            whisker_high_bound = q3 + 1.5 * iqr
            
            # Whiskers are the min/max values within the bounds (excluding outliers)
            whisker_low = float(np.min(latencies[latencies >= whisker_low_bound]))
            whisker_high = float(np.max(latencies[latencies <= whisker_high_bound]))
            
            result[codepath_key][label] = {
                "whisker_low": round(whisker_low, 3),
                "q1": round(q1, 3),
                "median": round(median, 3),
                "q3": round(q3, 3),
                "whisker_high": round(whisker_high, 3),
            }
    
    return result


class EfficiencyStats(NamedTuple):
    """Statistics from efficiency calculation."""
    efficiency: np.ndarray
    total_iterations: int
    mean_cycles: float
    theoretical_cycles: int
    mean_efficiency: float
    std_efficiency: float


def compute_and_report_efficiency(
    schedule: Schedule,
    yaml_path: str
) -> EfficiencyStats:
    """
    Compute efficiency statistics and print a summary report.
    
    Args:
        schedule: Schedule object with filled latencies from CSV files
        yaml_path: Path to the Tensile YAML config file
        
    Returns:
        EfficiencyStats containing efficiency array and related metrics
    """
    cycles_per_mfma, total_iterations = parse_yaml_for_mfma(yaml_path)
    
    # Calculate total cycles from schedule
    # Sum latencies per codepath, take MAX across codepaths (parallel execution)
    num_csv_files = len(schedule.ngl_latencies)
    codepath_totals = []
    for codepath in schedule.codepaths:
        codepath_sum = np.zeros(num_csv_files)
        for instruction in codepath:
            codepath_sum += np.array(instruction.latencies)
        codepath_totals.append(codepath_sum)
    mainloop_cycles = np.maximum.reduce(codepath_totals)
    # Add NGL and NLL (sequential)
    total_cycles = mainloop_cycles + np.array(schedule.ngl_latencies) + np.array(schedule.nll_latencies)
    
    # Calculate efficiency
    mainloop_theoretical = schedule.num_mfma * cycles_per_mfma * total_iterations
    ngl_theoretical = schedule.num_mfma * cycles_per_mfma * 1
    nll_theoretical = schedule.num_mfma * cycles_per_mfma * 1
    theoretical_cycles = mainloop_theoretical + ngl_theoretical + nll_theoretical
    efficiency = np.where(
        total_cycles > 0,
        (theoretical_cycles / total_cycles) * 100,
        0
    )
    
    mean_cycles = float(np.mean(total_cycles))
    mean_efficiency = float(np.mean(efficiency))
    std_efficiency = float(np.std(efficiency))
    
    print(f"Actual cycles: {mean_cycles:.0f}")
    print(f"Theoretical cycles: {theoretical_cycles}")
    print(f"Mean efficiency: {mean_efficiency:.1f}% +/- {std_efficiency:.1f}%")
    
    return EfficiencyStats(
        efficiency=efficiency,
        total_iterations=total_iterations,
        mean_cycles=mean_cycles,
        theoretical_cycles=theoretical_cycles,
        mean_efficiency=mean_efficiency,
        std_efficiency=std_efficiency
    )


def generate_all_histograms(
    schedule: Schedule,
    efficiency: np.ndarray,
    output_dir: str,
    total_iterations: int,
) -> None:
    """
    Generate histograms for all mainloop codepaths (including MFMAs).
    
    Args:
        schedule: Schedule object with codepaths containing Instruction objects with latencies
        efficiency: Array of efficiency percentages
        output_dir: Directory to save plots
        total_iterations: Total number of loop iterations
    """
    filename_prefix = "cycles_histogram"
    title_template = "Main Loop Instruction Median Cycle Counts per Iteration"
    title_template_simd = "Codepath {codepath_idx} Main Loop Instruction Median Cycle Counts per Iteration"
    y_label = "Median Latency per Iteration (cycles)"
    
    num_codepaths = schedule.num_code_paths
    num_runs = len(schedule.codepaths[0][0].latencies) if schedule.codepaths and schedule.codepaths[0] else 0
    
    def extract_data_from_codepath(codepath: List[Instruction]) -> Tuple[np.ndarray, List[Tuple[int, str, InstructionClass]], Dict[int, str]]:
        """Extract cycle_data, instructions list, and semantic_labels from a codepath."""
        num_instructions = len(codepath)
        num_csv_files = len(codepath[0].latencies) if codepath else 0
        
        # Build cycle_data array: shape [num_instructions, num_csv_files]
        cycle_data = np.zeros((num_instructions, num_csv_files))
        for i, inst in enumerate(codepath):
            cycle_data[i, :] = inst.latencies
        
        # Build instructions list: (position, name, instruction_class)
        instructions = [
            (i, inst.name, inst.instruction_class)
            for i, inst in enumerate(codepath)
        ]
        
        # Build semantic_labels dict: {position: "name[codepath][idx]"}
        semantic_labels = {
            i: str(inst)
            for i, inst in enumerate(codepath)
        }
        
        return cycle_data, instructions, semantic_labels
    
    # For single codepath, don't add codepath index to filename
    if num_codepaths == 1:
        codepath = schedule.codepaths[0]
        cycle_data, instructions, semantic_labels = extract_data_from_codepath(codepath)
        
        output_path = os.path.join(output_dir, f"{filename_prefix}.png")
        generate_boxplot(
            cycle_data, instructions, efficiency,
            output_path, title_template, y_label, num_runs, total_iterations,
            semantic_labels=semantic_labels
        )
    else:
        # Multiple codepaths: calculate consistent y-axis across all
        global_y_max = 0
        histogram_data = []
        
        for codepath_idx, codepath in enumerate(schedule.codepaths):
            cycle_data, instructions, semantic_labels = extract_data_from_codepath(codepath)
            
            # Track global max for consistent y-axis (use max of data for box plot upper whiskers)
            filtered_data, filtered_instructions, filtered_labels, y_values = prepare_histogram_data(
                cycle_data, instructions, semantic_labels, total_iterations
            )
            if len(y_values) > 0:
                data_per_iter = filtered_data / total_iterations
                local_max = np.max(data_per_iter)
                if local_max > global_y_max:
                    global_y_max = local_max
            
            histogram_data.append((codepath_idx, filtered_data, filtered_instructions, filtered_labels))
        
        # Add 5% padding to y_max for visual clarity
        global_y_max = global_y_max * 1.05
        
        for codepath_idx, filtered_data, filtered_instructions, filtered_labels in histogram_data:
            output_path = os.path.join(output_dir, f"{filename_prefix}_codepath_{codepath_idx}.png")
            title = title_template_simd.format(codepath_idx=codepath_idx)
            generate_boxplot(
                filtered_data, filtered_instructions, efficiency,
                output_path, title, y_label, num_runs, total_iterations, global_y_max,
                semantic_labels=filtered_labels
            )

def main():
    """Unified main function for both SIMD-specialized and non-SIMD kernels."""
    args = parse_args()
    
    csv_files = gather_csv_files(args.csv_dir, args.pattern)
    # TODO: The ASM file no longer needs tags on the instructions.
    asm_file = find_assembly_file(args.tensile_dir)

    schedule = schedule_from_yaml(args.yaml)
    loop_ranges = find_all_loop_ranges(asm_file, csv_files[0], schedule.num_code_paths)
    fill_schedule_with_latencies(schedule, loop_ranges, csv_files)

    # TODO: Efficiency calculation seems off when compared to manual calculation, Actual cycles are too high.
    stats = compute_and_report_efficiency(schedule, args.yaml)

    # Generate histograms with semantic labels (including MFMAs)
    generate_all_histograms(
        schedule, stats.efficiency,
        args.output_dir, stats.total_iterations,
    )
    
    if args.output_latency_json or args.output_excess_latency_table:
        latency_dict = create_latency_dict(schedule, stats.total_iterations)
    
    if args.output_latency_json:
        import json
        latency_json_path = os.path.join(args.output_dir, "latency_dict.json")
        with open(latency_json_path, "w") as f:
            json.dump(latency_dict, f, indent=2)
    
    if args.output_excess_latency_table:
        table_path = os.path.join(args.output_dir, "excess_latency_table.txt")
        table_str = generate_excess_latency_table(latency_dict, output_path=table_path)
        print(f"\n{table_str}")
    return 0


if __name__ == "__main__":
    exit(main())
