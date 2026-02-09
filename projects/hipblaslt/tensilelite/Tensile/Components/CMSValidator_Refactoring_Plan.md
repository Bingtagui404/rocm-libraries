# CMSValidator Refactoring Plan

This document outlines architectural improvements for the CMSValidator module and provides step-by-step implementation plans for each recommendation.

---

## Table of Contents

1. [Issue Summary](#issue-summary)
2. [Recommendations](#recommendations)
   - [R1: Split File Into Modules](#r1-split-file-into-modules)
   - [R2: Replace Float Indices with Composite Key](#r2-replace-float-indices-with-composite-key)
   - [R3: Make Mutation Order Explicit](#r3-make-mutation-order-explicit)
   - [R4: Extract Magic Numbers to Constants](#r4-extract-magic-numbers-to-constants)
   - [R5: Define Typed Context](#r5-define-typed-context)
   - [R6: Use Registry Pattern for Pack Handling](#r6-use-registry-pattern-for-pack-handling)
   - [R7: Define Formal ValidationPass Interface](#r7-define-formal-validationpass-interface)
   - [R8: Centralize Error Messages](#r8-centralize-error-messages)
   - [R9: Clarify Validation Logic Location](#r9-clarify-validation-logic-location)
   - [R10: Separate Timeline Responsibilities](#r10-separate-timeline-responsibilities)
   - [R11: Improve Test Infrastructure](#r11-improve-test-infrastructure)
   - [R12: Document Limitations Formally](#r12-document-limitations-formally)
   - [R13: Standardize ValidatorInstruction Class Hierarchy](#r13-standardize-validatorinstruction-class-hierarchy)
3. [Implementation Plans](#implementation-plans)

---

## Issue Summary

| # | Issue | Severity | Category |
|---|-------|----------|----------|
| 1 | 2100+ line file with mixed responsibilities | High | Maintainability |
| 2 | Float indices risk precision bugs | High | Correctness |
| 3 | Order-dependent mutations | High | Correctness |
| 4 | Magic numbers throughout | Medium | Maintainability |
| 5 | Untyped `context: dict` | Medium | Type Safety |
| 6 | Nested conditionals for pack modes | Medium | Extensibility |
| 7 | No formal validation pass interface | Low | Extensibility |
| 8 | Inline error message construction | Low | Maintainability |
| 9 | Mixed validation logic locations | Medium | Clarity |
| 10 | Timeline class has too many jobs | Medium | Maintainability |
| 11 | Testing infrastructure gaps | Medium | Testability |
| 12 | Undocumented limitations | Low | Documentation |
| 13 | Inconsistent instruction class interfaces | Medium | Type Safety / Maintainability |

---

## Recommendations

### R1: Split File Into Modules

**Current State**: Single 2100+ line file containing instruction classes, timeline management, 8 validation passes, pack handling, and utility functions.

**Target State**:
```
Tensile/Components/CMSValidator/
├── __init__.py              # Public API: isValid(), ValidationResult
├── instructions.py          # ValidatorInstruction base + all subclasses
├── timeline.py              # Timeline class
├── constants.py             # All magic numbers as named constants
├── context.py               # ValidationContext dataclass
├── errors.py                # Error message templates
├── passes/
│   ├── __init__.py          # Pass registry, run_all_passes()
│   ├── base.py              # ValidationPass protocol
│   ├── instruction_count.py
│   ├── ordering.py
│   ├── local_reads.py
│   ├── global_reads.py
│   ├── packs.py
│   └── scc_overlap.py
├── pack_handlers/
│   ├── __init__.py          # get_pack_handler() factory
│   ├── base.py              # PackHandler ABC
│   ├── bf16.py
│   ├── tf32.py
│   └── tf32_4x4mfma.py
└── utils/
    ├── __init__.py
    ├── mfma_reorder.py      # invert_mfma_reorder, find_earliest_mfma_execution
    └── index_transforms.py  # lr_needed_by_mfma, index_for_force_unroll_sub_iter
```

**Benefits**:
- Each file has single responsibility
- Easier to navigate and understand
- Enables parallel development
- Improves test isolation

---

### R2: Replace Float Indices with Composite Key

**Current State**:
```python
instruction.issued_at = 5.25  # vmfma_index=5, sub_index=1 of 4
```

The float index is built up incrementally across three stages:
1. **Construction** (`_populate_instructions`): `issued_at = idx_vmfma` (raw integer)
2. **`_insert`**: `issued_at += num_vmfma * loop_index` (loop offset hack to encode which loop)
3. **`_resolve_issued_at_indices`**: `issued_at += i_instruction / divisor` (sub-position via float fractions, with special-case handling for idx=-1 and idx=num_vmfma-1)

**Problems**:
- Float comparisons can fail due to precision
- Loop identity is encoded as an arithmetic offset rather than an explicit field
- Sub-index precision requires fragile float arithmetic with special cases
- Three mutation stages make the code hard to follow

**Target State**:
```python
@dataclass(frozen=True, order=True)
class SchedulePosition:
    """Represents a position in the schedule with sub-index precision.

    Fields are ordered for comparison: loop_index first (coarsest), then vmfma_index,
    then sub_index (finest). @dataclass(order=True) auto-generates __lt__, __le__,
    __gt__, __ge__, and __eq__ using tuple-style comparison over fields in declaration order.

    Args:
        loop_index: Which loop this position belongs to (0=ML-1, 1=ML, 2=NGL, 3=NLL).
        vmfma_index: The VMFMA index within the loop (-1 to num_vmfma-1).
        sub_index: Position within a VMFMA slot (0-based, for multiple instructions at same vmfma_index).
    """
    loop_index: int
    vmfma_index: int
    sub_index: int = 0

    @property
    def display_index(self) -> int:
        """Return the vmfma_index for user-facing messages."""
        return self.vmfma_index
```

**Key Design Decisions**:

1. **`frozen=True`**: The `SchedulePosition` is created in one shot inside `_insert`, where all three fields (loop_index, vmfma_index, sub_index) are known. The sub_index is simply the current length of the instruction list at that slot. This eliminates incremental mutation entirely.

2. **`order=True`**: Auto-generates all comparison methods (`<`, `<=`, `>`, `>=`, `==`) using tuple-style comparison over `(loop_index, vmfma_index, sub_index)`. No manual comparator methods needed.

3. **`loop_index` field**: Replaces the `num_vmfma * loop_index` offset hack in `_insert`. Loop identity is now explicit rather than encoded arithmetically.

4. **Eliminates `_resolve_issued_at_indices`**: This entire method becomes unnecessary since the sub_index is computed at insert time.

**Benefits**:
- No floating-point precision issues
- Immutable after creation (frozen) — no accidental mutation
- Explicit loop identity instead of arithmetic offset hack
- All comparison operators generated automatically
- Eliminates `_resolve_issued_at_indices` entirely
- Self-documenting code

---

### R3: Unified Timeline with Progressive Constraint Addition

**Current State**:
```python
# Each pass creates its own Timeline with a subset of instruction types
def verify_lrs_finished_before_vmfma(...):
    timeline = Timeline(["LRA0", "LRB0", ...], ...)  # Subset
    set_lr_needed_by_for_VMFMA(timeline, ...)
    apply_swaits(timeline)
    return validate_timeline(timeline)

def verify_grs_not_too_early(...):
    timeline = Timeline(["GRA", "GRB", ...], ...)    # Different subset
    apply_swaits(timeline)                           # Called again!
    set_gr_must_start_after_from_lr0s(...)
    return validate_timeline(timeline)
```

**Problems**:
1. Timeline created 4 times with different instruction subsets
2. Transformations like `apply_swaits` called redundantly
3. Order dependencies are implicit and error-prone

**Target State** (Unified Timeline):
```python
def isValid(scheduleInfo, context):
    for code_path in range(scheduleInfo.numCodePaths):
        # Structural checks (no Timeline needed)
        if error := verify_correct_number_of_instructions(...): return error
        if error := verify_ascending_order(...): return error

        # Create SINGLE Timeline with ALL instruction types
        timeline = create_unified_timeline(scheduleInfo, kernel, code_path)

        # Pass 3: Add LocalRead constraints, then validate
        add_local_read_constraints(timeline, kernel, mfma_reorder)
        if error := validate_timeline(timeline):
            return False, error

        # Pass 4: Add Pack constraints, then validate
        add_pack_constraints(timeline, kernel, mfma_reorder)
        if error := validate_timeline(timeline):
            return False, error

        # Pass 5: Add GR-not-too-early constraints, then validate
        add_gr_not_too_early_constraints(timeline, swap_global_read_order)
        if error := validate_timeline(timeline):
            return False, error

        # Pass 6: Add GR-finish-before-LR constraints, then validate
        add_gr_finish_before_lr_constraints(timeline, swap_global_read_order)
        if error := validate_timeline(timeline):
            return False, error

        # More structural checks
        if error := verify_scc_overlap(...): return error
        if error := verify_gr_inc_order(...): return error

    return True, ""
```

**Key Insight**: Instruction `validate()` methods already handle unset constraints gracefully:
- `LocalRead.validate()`: `if self.needed_by.issued_at == float('inf'): return None`
- `GlobalRead._validate_must_start_after()`: `if self.must_start_after.done_idx() == float('-inf'): return None`

This means calling `validate_timeline()` after each pass works correctly - instructions with unset constraints pass, instructions with set constraints get validated.

**Benefits**:
- Single Timeline creation per code path (not 4)
- Each transformation called exactly once
- Clear pass ordering enforced by code structure
- Constraints accumulate; final Timeline represents fully validated schedule
- Reuses existing `validate_timeline()` function unchanged

---

### R4: Extract Magic Numbers to Constants

**Current State**:
```python
if 4 <= idx_in_group < 20:
    continue
pack.min_quad_cycles_before_result_used = 5
idx_in_group = pack.issue_index % 10
```

**Target State**:
```python
# constants.py
class PackGroupSizes:
    """Number of pack instructions per logical group."""
    BF16 = 2          # Packs per element pair
    TF32_REGULAR = 24 # 4 CVT0 + 16 middle + 4 CVT1
    TF32_4X4_MFMA = 10 # 4 CVT0 + 2 MFMA + 4 CVT1

class TF32PackIndices:
    """Index ranges within a TF32 pack group."""
    # Regular TF32 (groups of 24)
    CVT0_START = 0
    CVT0_END = 4
    MIDDLE_16_START = 4
    MIDDLE_16_END = 20
    CVT1_START = 20
    CVT1_END = 24

    # 4x4 MFMA TF32 (groups of 10)
    MFMA_4X4_START = 4
    MFMA_4X4_END = 6

class QuadCycleRequirements:
    """Minimum quad-cycles before result can be used (CDNA 4 ISA 7.6)."""
    CVT_BEFORE_MFMA = 2
    MFMA_4X4_BEFORE_CVT1 = 5
    STANDARD_MFMA_FINISH = 3
    MFMA_4X4_FINISH = 1

class LoopNames:
    """Loop iteration identifiers."""
    MAIN_LOOP_PREV = "ML-1"
    MAIN_LOOP = "ML"
    NO_GLOBAL_LOAD = "NGL"
    NO_LOCAL_LOAD = "NLL"
```

---

### R5: Define Typed Context

**Current State**:
```python
def isValid(scheduleInfo: 'ScheduleInfo', context: dict) -> tuple[bool, str]:
    kernel = context["kernel"]
    idMap = context.get("idMap")  # Optional? Unknown structure
```

**Target State**:
```python
@dataclass
class ValidationContext:
    """Typed context for CMS validation."""
    kernel: 'Solution'
    id_map: Optional[dict[str, list[Any]]] = None

    # Convenience properties to avoid repeated kernel lookups
    @property
    def swap_global_read_order(self) -> bool:
        return self.kernel.get("SwapGlobalReadOrder", False)

    @property
    def direct_to_lds(self) -> bool:
        return self.kernel.get("DirectToLds", False)

    @property
    def use_f32x_emulation(self) -> bool:
        return self.kernel.get("UseF32XEmulation", False)

    @property
    def use_mfma_f32x_emulation(self) -> bool:
        return self.kernel.get("UseMFMAF32XEmulation", False)

    @property
    def use_plr_pack(self) -> bool:
        return self.kernel.get("UsePLRPack", False)

    @property
    def force_unroll_sub_iter(self) -> bool:
        return self.kernel.get("ForceUnrollSubIter", False)

    @property
    def n_tiles_a(self) -> int:
        return self.kernel["MIWaveTileA"]

    @property
    def n_tiles_b(self) -> int:
        return self.kernel["MIWaveTileB"]
```

---

### R6: Use Registry Pattern for Pack Handling

**Current State**:
```python
def hook_up_packs(timeline, kernel, mfma_reorder):
    if is_tf32_emulation:
        if is_4x4mfma_tf32:
            _hook_up_packs_f32_mfma(packs, local_reads)
        else:
            _hook_up_packs_f32(packs, all_middle_16_packs, local_reads)
        _handle_min_pack_quad_cycles(packs, is_4x4mfma_tf32)
    else:
        _hook_up_packs_bf16(packs, local_reads)
```

**Problem**: Nested conditionals become unwieldy as pack modes grow. Adding a new mode requires modifying the central dispatch logic.

**Target State** (Registry with Decorators):
```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class PackContext:
    """Everything a pack handler might need."""
    packs: list[Pack]
    local_reads: list[LocalRead]
    all_packs_in_loop: list[Pack]  # For middle-16 lookups
    kernel: dict
    mfmas_by_index: dict[int, MFMA]
    mfma_reorder: list[int]
    num_vmfma: int
    loop_index: int

# Registry
_PACK_HANDLERS: dict[str, Callable[[PackContext], None]] = {}

def pack_handler(mode: str):
    """Register a function as a pack handler for a given mode."""
    def decorator(fn: Callable[[PackContext], None]):
        _PACK_HANDLERS[mode] = fn
        return fn
    return decorator

def get_pack_mode(kernel: dict) -> str:
    """Determine pack mode from kernel configuration."""
    if kernel.get("UseMFMAF32XEmulation"):
        return "tf32_4x4mfma"
    if kernel.get("UseF32XEmulation"):
        return "tf32"
    return "bf16"

def handle_packs(ctx: PackContext) -> None:
    """Dispatch to the appropriate pack handler."""
    mode = get_pack_mode(ctx.kernel)
    if mode not in _PACK_HANDLERS:
        raise ValueError(f"Unknown pack mode: {mode}")
    _PACK_HANDLERS[mode](ctx)
```

**Handler implementations** (each is a decorated function):
```python
@pack_handler("bf16")
def _handle_bf16(ctx: PackContext) -> None:
    """BF16: each pack depends on 2 consecutive LRs."""
    num_element_pairs = len(ctx.local_reads) // 2
    for pack in ctx.packs:
        element_idx = pack.issue_index % num_element_pairs
        lr1 = ctx.local_reads[element_idx * 2]
        lr2 = ctx.local_reads[element_idx * 2 + 1]
        pack.must_start_after = max(lr1, lr2, key=lambda lr: lr.issued_at)
    _set_pack_needed_by(ctx.packs, ctx.mfmas_by_index, ...)

@pack_handler("tf32")
def _handle_tf32(ctx: PackContext) -> None:
    """TF32: groups of 24 with CVT0 -> middle-16 -> CVT1 chain."""
    _hook_up_packs_f32(ctx.packs, ctx.all_packs_in_loop, ctx.local_reads)
    _set_min_quad_cycles(ctx.packs, cycles=2)
    _set_pack_needed_by(ctx.packs, ctx.mfmas_by_index, ...)

@pack_handler("tf32_4x4mfma")
def _handle_tf32_4x4mfma(ctx: PackContext) -> None:
    """TF32 4x4 MFMA: groups of 10 with CVT0 -> MFMA -> CVT1 chain."""
    _hook_up_packs_f32_mfma(ctx.packs, ctx.local_reads)
    _set_min_quad_cycles(ctx.packs, cycles=5)
    _set_pack_needed_by(ctx.packs, ctx.mfmas_by_index, ...)
```

**Adding a new mode** requires only a new decorated function:
```python
@pack_handler("fp8")
def _handle_fp8(ctx: PackContext) -> None:
    """FP8: whatever FP8 needs."""
    ...
```

**Benefits**:
- No class hierarchy or ABC boilerplate
- Self-registering handlers (decorator makes intent clear)
- Handlers can live in separate files and register on import
- Easy to discover all modes: `_PACK_HANDLERS.keys()`
- Easy to test: call function directly with mock PackContext
- Pythonic pattern (same as Flask routes, pytest fixtures, Click commands)

**Comparison to class-based Strategy Pattern**:

| Aspect | Classes | Registry |
|--------|---------|----------|
| Add new mode | New class + factory update | Add decorated function |
| Boilerplate | ABC, abstractmethod, inheritance | One decorator |
| Discoverability | Grep for subclasses | `_PACK_HANDLERS.keys()` |
| Testing | Instantiate class, call methods | Call function with mock context |
| IDE support | Good | Good (dataclass + type hints) |

---

### R7: Keep Functions, Not Classes for Passes

**Current State**:
```python
# Functions with implicit contract
def verify_lrs_finished_before_vmfma(schedule_info, context, code_path):
    ...
    return True, ""
```

**Recommendation**: Keep using functions rather than introducing a formal ValidationPass class/protocol.

**Rationale**:
- The unified timeline approach (R3) already provides clear structure
- Functions are simpler and match the existing codebase style
- No need for class boilerplate when functions work well
- The pass ordering is explicit in the `isValid()` function itself

**Target State** (with R3 unified timeline):
```python
# Constraint-adding functions (new)
def add_local_read_constraints(timeline: Timeline, kernel, mfma_reorder) -> None:
    """Add LR.needed_by and LR.guaranteed_by constraints."""
    set_lr_needed_by_for_VMFMA(timeline, kernel, mfma_reorder)
    apply_swaits(timeline)
    apply_barriers(timeline)

def add_pack_constraints(timeline: Timeline, kernel, mfma_reorder) -> None:
    """Add Pack constraints."""
    hook_up_packs(timeline, kernel, mfma_reorder)
    estimate_quad_cycles(timeline, kernel)

def add_gr_not_too_early_constraints(timeline: Timeline, swap_global_read_order) -> None:
    """Add GR.must_start_after constraints."""
    set_gr_must_start_after_from_lr0s(timeline, swap_global_read_order)
    apply_must_start_after_barriers(timeline)

def add_gr_finish_before_lr_constraints(timeline: Timeline, swap_global_read_order) -> None:
    """Add GR.needed_by constraints."""
    set_gr_needed_by_from_lrs(timeline, swap_global_read_order)

# Structural check functions (keep existing)
def verify_correct_number_of_instructions(schedule_info, context, code_path) -> tuple[bool, str]: ...
def verify_ascending_order(schedule_info, context, code_path) -> tuple[bool, str]: ...
def verify_scc_overlap(schedule_info, context, code_path) -> tuple[bool, str]: ...
def verify_gr_inc_order(schedule_info, context, code_path) -> tuple[bool, str]: ...
```

**Benefits**:
- No new abstractions to learn
- Explicit ordering in `isValid()` is clear and auditable
- Easy to add new passes: just add a function and call it in the right place

---

### R8: Centralize Error Messages

**Current State**:
```python
return f"{name} @ idx={issued_at} is not valid. It is guaranteed by the SWait @ idx={guaranteed_by} which is after..."
```

**Target State** (module-level functions, no class):
```python
# Error message functions - keep near top of CMSValidator.py
# (or in errors.py if file is split per R1)

def _error_issued_too_late(
    name: str,
    issued_at: int,
    needed_by_name: str,
    needed_by_at: int,
    context: str = ""
) -> str:
    """Format error for instruction issued after its consumer."""
    msg = (
        f"{name} @ idx={issued_at} issued too late, "
        f"must be issued before {needed_by_name} @ idx={needed_by_at}"
    )
    if context:
        msg += f" {context}"
    return msg + "."


def _error_issued_too_early(
    name: str,
    issued_at: int,
    must_start_after_name: str,
    must_start_after_at: int
) -> str:
    """Format error for instruction issued before its dependency."""
    return (
        f"{name} @ idx={issued_at} issued too early, "
        f"must be issued after {must_start_after_name} @ idx={must_start_after_at}."
    )


def _error_missing_barrier(
    before_name: str,
    before_at: int,
    after_name: str,
    after_at: int,
    required_order: str
) -> str:
    """Format error for missing synchronization barrier."""
    return (
        f"Missing SBarrier between {before_name} @ idx={before_at} "
        f"and {after_name} @ idx={after_at}. "
        f"Required order: {required_order}."
    )


def _error_no_guarantee(name: str, issued_at: int) -> str:
    """Format error for instruction with no SWaitCnt guarantee."""
    return f"{name} @ idx={issued_at} has no SWaitCnt guaranteeing completion."


def _error_wrong_instruction_count(name: str, actual: int, expected: int) -> str:
    """Format error for incorrect number of instructions."""
    return f"{name} has {actual} instructions, but {expected} are required."


def _error_quad_cycle_violation(
    name: str,
    issued_at: int,
    needed_by_name: str,
    needed_by_at: int,
    required: int,
    actual: int
) -> str:
    """Format error for insufficient quad-cycle gap."""
    return (
        f"{name} @ idx={issued_at} has insufficient gap before "
        f"{needed_by_name} @ idx={needed_by_at}. "
        f"Required: {required} quad-cycles, actual: {actual}."
    )
```

**Usage example**:
```python
def validate(self) -> Optional[str]:
    if self.issued_at >= self.needed_by.issued_at:
        return _error_issued_too_late(
            name=self.name,
            issued_at=floor(self.issued_at),
            needed_by_name=self.needed_by.name,
            needed_by_at=floor(self.needed_by.issued_at)
        )
    return None
```

---

### R9: Clarify Validation Logic Location

**Current State**: Mixed - some validation in `Instruction.validate()`, some in standalone functions.

**Target State**: All validation in instruction classes (self-validating objects):

```python
@dataclass
class LocalRead(ValidatorInstruction):
    # ... fields ...

    def validate(self) -> Optional[str]:
        """Validate this instruction against its constraints."""
        if not self._has_constraints():
            return None

        if not self._is_guaranteed_before_needed():
            return self._format_timing_error()

        return None

    def _has_constraints(self) -> bool:
        return self.needed_by.issued_at != float('inf')

    def _is_guaranteed_before_needed(self) -> bool:
        return self.guaranteed_by < self.needed_by.issued_at

    def _format_timing_error(self) -> str:
        return ValidationErrors.instruction_issued_too_late(
            name=self.name,
            issued_at=self._display_index(),
            needed_by_name=self.needed_by.name,
            needed_by_at=self.needed_by._display_index()
        )
```

**Validation passes** then only:
1. Build the timeline
2. Set up constraints (needed_by, guaranteed_by, etc.)
3. Call `validate_timeline()` which iterates and calls each instruction's `validate()`

---

### R10: Separate Timeline Responsibilities

**Current State**: Timeline handles parsing, storage, iteration management, index resolution, and lookups.

**Target State**:

```python
# schedule_parser.py
class ScheduleParser:
    """Parses ScheduleInfo into ValidatorInstructions."""

    def parse(
        self,
        schedule_info: 'ScheduleInfo',
        instruction_names: list[str],
        code_path: int,
        context: ValidationContext
    ) -> list[ValidatorInstruction]:
        """Parse schedule into instruction list."""
        ...

# loop_manager.py
class LoopManager:
    """Manages loop iterations (ML-1, ML, NGL, NLL)."""

    def __init__(self, base_instructions: list[ValidatorInstruction], context: ValidationContext):
        self._loops = self._create_loops(base_instructions, context)

    def get_loop(self, loop_name: str) -> 'LoopIteration':
        ...

    def all_instructions(self) -> Iterator[ValidatorInstruction]:
        """Iterate all instructions across all loops."""
        ...

# timeline.py
class Timeline:
    """Immutable view of scheduled instructions."""

    def __init__(self, loop_manager: LoopManager):
        self._loop_manager = loop_manager
        self._index = self._build_index()

    def get_instructions(self, name: str, loop: str) -> Sequence[ValidatorInstruction]:
        ...

    def get_instructions_combined(self, name: str) -> Sequence[ValidatorInstruction]:
        ...

    def get_at_vmfma_index(self, index: int, loop: str) -> Sequence[ValidatorInstruction]:
        ...
```

---

### R11: Improve Test Infrastructure

**Current State**:
```python
if "idMap" not in context:
    printWarning("idMap not found in context. Skipping...")
    return True, ""
```

**Target State**:

```python
# tests/fixtures.py
class ValidationContextFactory:
    """Factory for creating test ValidationContext objects."""

    @staticmethod
    def default() -> ValidationContext:
        return ValidationContext(
            kernel=KernelFixtures.default_kernel(),
            id_map=None
        )

    @staticmethod
    def with_tf32() -> ValidationContext:
        kernel = KernelFixtures.default_kernel()
        kernel["UseF32XEmulation"] = True
        kernel["UseDirect32XEmulation"] = True
        return ValidationContext(kernel=kernel)

    @staticmethod
    def with_full_id_map(schedule_info: 'ScheduleInfo') -> ValidationContext:
        return ValidationContext(
            kernel=KernelFixtures.default_kernel(),
            id_map=IdMapBuilder.from_schedule(schedule_info)
        )

class ScheduleInfoBuilder:
    """Builder for creating test ScheduleInfo objects."""

    def __init__(self):
        self._opt_schedule = {}
        self._num_mfma = 16
        self._mfma_reorder = []

    def with_local_reads(self, lra0: list[int], lrb0: list[int]) -> 'ScheduleInfoBuilder':
        self._opt_schedule["LRA0"] = [lra0]
        self._opt_schedule["LRB0"] = [lrb0]
        return self

    def with_global_reads(self, gra: list[int], grb: list[int]) -> 'ScheduleInfoBuilder':
        self._opt_schedule["GRA"] = [gra]
        self._opt_schedule["GRB"] = [grb]
        return self

    def build(self) -> 'ScheduleInfo':
        ...

# In tests
def test_lr_finished_before_vmfma():
    schedule = (ScheduleInfoBuilder()
        .with_local_reads([0, 1, 2], [0, 1, 2])
        .with_sync_at([3])
        .build())
    context = ValidationContextFactory.default()

    result = verify_lrs_finished_before_vmfma(schedule, context, code_path=0)

    assert result.passed
```

---

### R12: Document Limitations Formally

**Current State**: TODOs and limitations scattered in code comments.

**Target State**: Create `LIMITATIONS.md`:

```markdown
# CMSValidator Known Limitations

## Unsupported Configurations

### UseDirect32XEmulation = False with UseF32XEmulation = True
- **Status**: Not supported
- **Error**: Raises ValueError
- **Reason**: Non-direct TF32 emulation uses different pack sequences not yet modeled

### ForceUnrollSubIter with Register Reuse
- **Status**: Partially supported
- **Issue**: Does not fully account for VGPR reuse patterns
- **Tracking**: TODO in timeline.py line 489

## Validation Gaps

### False Negatives (schedule appears valid but may fail)
1. Quad-cycle estimation is conservative (doesn't model all stalls)
2. SBarrier timing assumes instant synchronization

### False Positives (schedule appears invalid but works)
1. None currently known

## Architecture-Specific Behavior

### CDNA 4 (gfx950)
- Quad-cycle requirements from ISA section 7.6 are enforced
- 4x4 MFMA TF32 path is validated

### CDNA 3 and earlier
- Quad-cycle requirements not enforced (different ISA)
```

---

### R13: Standardize ValidatorInstruction Class Hierarchy

**Current State**: The `ValidatorInstruction` base class is minimal (just `name`, `issued_at`, `validate()`, `done_idx()`) and subclasses diverge significantly in their field types, constraint patterns, and error message formatting. This leads to several concrete problems:

**Problem 1: `needed_by` type mismatch across subclasses**
```python
class LocalRead(ValidatorInstruction):
    needed_by: ValidatorInstruction = ...   # An instruction object

class Pack(ValidatorInstruction):
    needed_by: ValidatorInstruction = ...   # An instruction object

class GlobalRead(ValidatorInstruction):
    needed_by: float = float('inf')         # A bare float!
```
`GlobalRead.needed_by` is a `float` (the `issued_at` of the first LR1/3), while `LocalRead.needed_by` and `Pack.needed_by` are `ValidatorInstruction` references. This means:
- Error formatting code cannot be shared (one does `self.needed_by.name`, the other can't).
- `validate_timeline()` can't make any assumptions about constraint fields.
- `estimate_quad_cycles()` must use `hasattr()` checks instead of type-safe access.

**Problem 2: `num_vmfma` stored redundantly on each instruction**
```python
class LocalRead(ValidatorInstruction):
    num_vmfma: int          # Same value for all instances

class Pack(ValidatorInstruction):
    num_vmfma: int          # Same value for all instances

class GlobalRead(ValidatorInstruction):
    num_vmfma: int          # Same value for all instances
```
Every `LocalRead`, `Pack`, and `GlobalRead` stores the same `num_vmfma` value. It's used exclusively for display formatting (`floor(self.issued_at) % self.num_vmfma`) and cross-iteration detection (`self.needed_by.issued_at > self.num_vmfma`). Both of these uses disappear entirely if R2 (SchedulePosition) is implemented, since SchedulePosition would encode the vmfma index directly without needing modular arithmetic.

**Problem 3: Duplicated display-index computation**

The pattern `floor(self.issued_at) % self.num_vmfma` appears **19 times** across `validate()` methods, with a special case for idx=-1 appearing **3 times**:
```python
# This exact pattern (or minor variant) appears in LocalRead, Pack, and GlobalRead:
issued_at = floor(self.issued_at) % self.num_vmfma

# This special-case for idx=-1 appears in LocalRead and Pack:
if self.num_vmfma - 1 + 0.5 <= (value % self.num_vmfma) < self.num_vmfma:
    display_index = -1
else:
    display_index = floor(value) % self.num_vmfma
```

**Problem 4: Inconsistent error message formats**

Each class formats errors differently, making them hard to parse programmatically or visually:
```python
# LocalRead:
f"{self.name} @ idx={issued_at} is not valid. There are no guarantees on when it will be done."
f"{self.name} @ idx={issued_at} issued too late, must be guaranteed before {self.needed_by.name} @ idx={needed_by}{context_str} but only guaranteed @ idx={guaranteed_by}."

# Pack:
f"{self.name} @ idx={issued_at} issued too early, must be issued after idx={must_start_after_at} (because of {self.must_start_after.name} issued @ idx={must_start_after_issued_at})."
f"{self.name} @ idx={issued_at} issued too late, must be issued before {self.needed_by.name} @ idx={needed_by_at}."
f"{self.name} @ idx={issued_at} has wrong interleaving. Should have been followed by ..."
f"{self.name} @ idx={issued_at} has too little gap between it and ..."
f"{self.name} at index {issued_at} is not valid."  # Note: "at index" not "@ idx="!

# GlobalRead._validate_must_start_after():
f"{name} @ idx={issued_at} is issued too early. Must be issued after idx=..."
f"There is an SBarrier missing between the SWaitCnt @ idx=..."

# GlobalRead._validate_needed_by():
f"{name} @ idx={issued_at} is not valid. There are no guarantees on when it will be done."
f"{name} @ idx={issued_at} is not valid. There is no SBarrier acting on it."
f"{name} @ idx={issued_at} is not valid. It is guaranteed by the SWait @ idx=..."

# SWait:
f"SWait at index {floor(self.issued_at)} is invalid: ..."  # Uses "at index", no modulo wrapping

# Barrier:
f"Barrier at index {floor(self.issued_at)} is not valid. Must be >= -1."  # Uses "at index"
```

Note the inconsistencies: "at index" vs "@ idx=", "issued too early" vs "is issued too early", "is not valid" appearing in different positions, some messages explaining the fix ("Order must be X") and others not.

**Problem 5: `estimate_quad_cycles()` uses `hasattr()` checks**
```python
# Current: runtime duck-typing
if not hasattr(instruction, "needed_by") or instruction.needed_by is None:
    continue
if not hasattr(instruction, "min_quad_cycles_before_result_used"):
    continue
```
These `hasattr()` checks exist because the base class doesn't define `needed_by` or `min_quad_cycles_before_result_used`, so there's no type-safe way to check if an instruction has constraints.

**Target State**:

1. **Unify `GlobalRead.needed_by` to `ValidatorInstruction`** (matching `LocalRead` and `Pack`):
```python
class GlobalRead(ValidatorInstruction):
    needed_by: ValidatorInstruction = field(default_factory=lambda: MFMA(float('inf')))
    # Instead of: needed_by: float = float('inf')
```
Update `set_gr_needed_by_from_lrs()` to assign the LR1/3 instruction object rather than its `issued_at`:
```python
# Before:
for _, gr in grs:
    gr.needed_by = LR_target.issued_at   # float

# After:
for _, gr in grs:
    gr.needed_by = LR_target              # ValidatorInstruction
```

2. **Eliminate `num_vmfma` from instruction classes** (requires R2: SchedulePosition):

With SchedulePosition, display indices are accessed directly:
```python
# Before (19 occurrences):
issued_at = floor(self.issued_at) % self.num_vmfma

# After:
issued_at = self.issued_at.display_index
```

And the special idx=-1 handling moves into SchedulePosition:
```python
@dataclass(frozen=True, order=True)
class SchedulePosition:
    vmfma_index: int
    sub_index: int = 0

    @property
    def display_index(self) -> int:
        return self.vmfma_index
```

Cross-iteration detection (`self.needed_by.issued_at > self.num_vmfma`) would be handled by adding iteration info to SchedulePosition or by a separate mechanism.

3. **Shared error formatting functions** (module-level, see R8):

Once `needed_by` is unified, error formatting functions can work uniformly across all instruction types:
```python
def _error_issued_too_late(name: str, issued_at: int, needed_by_name: str, needed_by_at: int, context: str = "") -> str:
    msg = f"{name} @ idx={issued_at} issued too late, must be issued before {needed_by_name} @ idx={needed_by_at}"
    if context:
        msg += f" {context}"
    return msg + "."

def _error_issued_too_early(name: str, issued_at: int, must_start_after_name: str, must_start_after_at: int) -> str:
    return f"{name} @ idx={issued_at} issued too early, must be issued after {must_start_after_name} @ idx={must_start_after_at}."

def _error_no_guarantee(name: str, issued_at: int) -> str:
    return f"{name} @ idx={issued_at} has no guarantee on when it will be done."
```

These are usable by any class:
```python
# LocalRead.validate():
return _error_issued_too_late(self.name, self.issued_at.display_index, self.needed_by.name, self.needed_by.issued_at.display_index, context_str)

# Pack.validate():
return _error_issued_too_late(self.name, self.issued_at.display_index, self.needed_by.name, self.needed_by.issued_at.display_index)

# GlobalRead._validate_needed_by():
return _error_issued_too_late(self.name, self.issued_at.display_index, self.needed_by.name, self.needed_by.issued_at.display_index)
```

**Benefits**:
- `needed_by` has a single type across all instruction classes, enabling shared code
- `num_vmfma` is eliminated from instruction classes (absorbed into SchedulePosition via R2)
- Display index computation is centralized (19 occurrences reduced to SchedulePosition.display_index)
- Error messages are consistent and testable
- `hasattr()` checks in `estimate_quad_cycles()` are replaced with type-safe access
- Cross-iteration detection logic is standardized

**Relationship to other recommendations**:
- **Depends on R2** (SchedulePosition) for eliminating `num_vmfma`
- **Depends on R8** (error message centralization) for shared error functions
- **Enables R9** (clarify validation logic) by making instruction interfaces consistent
- **Can be done concurrently with R2** since both touch the same fields

---

## Implementation Plans

### Plan for R1: Split File Into Modules

**Estimated Effort**: Large (2-3 days)

**Step 1**: Create directory structure
```bash
mkdir -p Tensile/Components/CMSValidator/{passes,pack_handlers,utils}
touch Tensile/Components/CMSValidator/{__init__,instructions,timeline,constants,context,errors}.py
touch Tensile/Components/CMSValidator/passes/{__init__,base}.py
touch Tensile/Components/CMSValidator/pack_handlers/{__init__,base}.py
touch Tensile/Components/CMSValidator/utils/__init__.py
```

**Step 2**: Extract constants (do first - no dependencies)
- Move `MAIN_LOOP_PREV`, `MAIN_LOOP`, etc. to `constants.py`
- Add `PackGroupSizes`, `TF32PackIndices`, `QuadCycleRequirements`
- Update imports in original file

**Step 3**: Extract instruction classes
- Move `ValidatorInstruction`, `LocalRead`, `GlobalRead`, `Pack`, `MFMA`, `SWait`, `Barrier`, `SNop` to `instructions.py`
- Update imports

**Step 4**: Extract utility functions
- Move `invert_mfma_reorder`, `find_earliest_mfma_execution` to `utils/mfma_reorder.py`
- Move `lr_needed_by_mfma`, `index_for_force_unroll_sub_iter`, `_transform_index_*` to `utils/index_transforms.py`
- Move `schedule_get` to `utils/__init__.py`

**Step 5**: Extract Timeline class
- Move `Timeline`, `apply_barriers`, `apply_swaits`, etc. to `timeline.py`
- Keep transformation functions with Timeline for now

**Step 6**: Extract validation passes
- Create one file per pass in `passes/`
- Move each `verify_*` function to appropriate file
- Create `passes/__init__.py` with pass registry

**Step 7**: Update main module
- `CMSValidator/__init__.py` exports `isValid`, `ValidationResult`
- Create backward-compatible `CMSValidator.py` that imports from package

**Step 8**: Update all imports throughout codebase
- Search for `from Tensile.Components.CMSValidator import`
- Update to new paths

**Step 9**: Run tests and fix any issues

---

### Plan for R2: Replace Float Indices with Composite Key

**Estimated Effort**: Medium (1 day)

**Step 1**: Define SchedulePosition class
```python
@dataclass(frozen=True, order=True)
class SchedulePosition:
    """Fields ordered for tuple-style comparison: loop_index > vmfma_index > sub_index."""
    loop_index: int
    vmfma_index: int
    sub_index: int = 0

    @property
    def display_index(self) -> int:
        return self.vmfma_index
```

**Step 2**: Update ValidatorInstruction base class
- Change `issued_at: Union[int, float]` to `issued_at: SchedulePosition`
- Update `done_idx()` return type

**Step 3**: Update `Timeline._insert()` to create SchedulePosition in one shot
The sub_index is the current length of the instruction list at that slot (i.e., how many instructions are already there). The loop_index and vmfma_index are already known. This replaces both the loop offset hack and the need for `_resolve_issued_at_indices`.
```python
def _insert(self, vmfma_index: int, instruction: ValidatorInstruction, kernel: 'Solution') -> None:
    for loop in self.loops:
        if self._should_add(instruction, loop, kernel):
            _instruction = deepcopy(instruction)

            loop_index = self.loops.index(loop)
            sub_index = len(self._instructions_at_index[loop][vmfma_index + 1])
            _instruction.issued_at = SchedulePosition(
                loop_index=loop_index,
                vmfma_index=vmfma_index,
                sub_index=sub_index
            )

            # Adjust for NLL/NGL shifts (SWait handling unchanged).
            if isinstance(_instruction, SWait):
                if _instruction.vlcnt != -1:
                    vlcnt = max(0, _instruction.vlcnt - self.vlcnt_shift[loop])
                    _instruction.vlcnt = vlcnt
                if _instruction.dscnt != -1 and self.nll_zero_dscnt \
                   and loop in [NO_LOCAL_LOAD_LOOP]:
                    _instruction.dscnt = 0

            self._instructions_at_index[loop][vmfma_index + 1].append(_instruction)
```

**Step 4**: Remove `Timeline._resolve_issued_at_indices()`
- Delete the method entirely
- Remove the call from `Timeline.__init__`

**Step 5**: Update all comparisons
- Find all `issued_at < `, `issued_at > `, `issued_at >= `, `issued_at <= `
- These should work unchanged due to `order=True` generating tuple-style comparisons
- Replace `float('inf')` and `float('-inf')` sentinel values with sentinel SchedulePosition instances (e.g., `SchedulePosition(loop_index=999, vmfma_index=999)` for inf)

**Step 6**: Update `floor()` calls and display formatting
- `floor(self.issued_at) % self.num_vmfma` becomes `self.issued_at.display_index`
- `f"idx={issued_at}"` stays the same, but `issued_at` is now `self.issued_at.display_index`
- The special-case idx=-1 handling (`num_vmfma - 1 + 0.5 <= ...`) is eliminated — `display_index` returns `vmfma_index` directly, which is already -1

**Step 7**: Remove `num_vmfma` from instruction classes
- With `loop_index` explicit, the `num_vmfma * loop_index` offset is gone
- Cross-iteration detection (`self.needed_by.issued_at > self.num_vmfma`) becomes `self.needed_by.issued_at.loop_index > self.issued_at.loop_index`
- The `num_vmfma` field on `LocalRead`, `Pack`, and `GlobalRead` can be removed

**Step 8**: Run tests and fix edge cases
```bash
pytest Tensile/Tests/unit/test_CMSValidator*.py -v
```

---

### Plan for R3: Unified Timeline with Progressive Constraint Addition

**Estimated Effort**: Medium (1 day)

**Step 1**: Add `create_unified_timeline()` function
```python
ALL_INSTRUCTION_NAMES = [
    "LRA0", "LRB0", "LRA1", "LRB1", "LRA3", "LRB3",
    "GRA", "GRB",
    "PackA0", "PackB0", "PackA1", "PackB1", "PackA3", "PackB3",
    "SYNC", "SNOP",
]

def create_unified_timeline(schedule_info, kernel, code_path) -> Timeline:
    """Create a single Timeline with all instruction types."""
    available = set(schedule_info.optSchedule.keys())
    names = [n for n in ALL_INSTRUCTION_NAMES if n in available]
    return Timeline(names, code_path, schedule_info, kernel)
```

**Step 2**: Add constraint-adding wrapper functions
```python
def add_local_read_constraints(timeline, kernel, mfma_reorder) -> None:
    set_lr_needed_by_for_VMFMA(timeline, kernel, mfma_reorder)
    apply_swaits(timeline)  # Sets guaranteed_by for BOTH LRs and GRs
    apply_barriers(timeline)

def add_pack_constraints(timeline, kernel, mfma_reorder) -> None:
    if kernel.get("UseF32XEmulation") and not kernel.get("UseDirect32XEmulation"):
        return  # Skip - not supported
    hook_up_packs(timeline, kernel, mfma_reorder)
    estimate_quad_cycles(timeline, kernel)

def add_gr_not_too_early_constraints(timeline, swap_global_read_order) -> None:
    set_gr_must_start_after_from_lr0s(timeline, swap_global_read_order)
    apply_must_start_after_barriers(timeline)

def add_gr_finish_before_lr_constraints(timeline, swap_global_read_order) -> None:
    set_gr_needed_by_from_lrs(timeline, swap_global_read_order)
```

**Step 3**: Rewrite `isValid()` to use unified timeline
```python
def isValid(scheduleInfo, context):
    kernel = context["kernel"]
    mfma_reorder = scheduleInfo.mfmaReorder or []
    swap = kernel.get("SwapGlobalReadOrder", False)

    for code_path in range(scheduleInfo.numCodePaths):
        # Structural checks
        status, msg = verify_correct_number_of_instructions(...)
        if not status: return False, f"Code path {code_path}: {msg}"
        status, msg = verify_ascending_order(...)
        if not status: return False, f"Code path {code_path}: {msg}"

        # Create unified timeline
        timeline = create_unified_timeline(scheduleInfo, kernel, code_path)

        # Add constraints and validate after each
        add_local_read_constraints(timeline, kernel, mfma_reorder)
        if error := validate_timeline(timeline):
            return False, f"Code path {code_path}: {error}"

        add_pack_constraints(timeline, kernel, mfma_reorder)
        if error := validate_timeline(timeline):
            return False, f"Code path {code_path}: {error}"

        add_gr_not_too_early_constraints(timeline, swap)
        if error := validate_timeline(timeline):
            return False, f"Code path {code_path}: {error}"

        add_gr_finish_before_lr_constraints(timeline, swap)
        if error := validate_timeline(timeline):
            return False, f"Code path {code_path}: {error}"

        # More structural checks
        status, msg = verify_scc_overlap(...)
        if not status: return False, f"Code path {code_path}: {msg}"
        status, msg = verify_gr_inc_order(...)
        if not status: return False, f"Code path {code_path}: {msg}"

    return True, ""
```

**Step 4**: Remove old `verify_*` functions that created their own Timelines
- `verify_lrs_finished_before_vmfma()` - replaced by pass 3
- `verify_packs_start_and_end_at_correct_indices()` - replaced by pass 4
- `verify_grs_not_too_early()` - replaced by pass 5
- `verify_grs_finish_before_lrs()` - replaced by pass 6

**Step 5**: Run tests to verify no regressions
```bash
pytest Tensile/Tests/unit/test_CMSValidator*.py -v
```

---

### Plan for R4: Extract Magic Numbers to Constants

**Estimated Effort**: Small (2-4 hours)

**Step 1**: Create constants.py with all constant classes (see R4 above)

**Step 2**: Search for magic numbers in CMSValidator.py
```bash
grep -n "== 24\|== 10\|== 4\|== 20\|< 4\|< 20\|>= 4" CMSValidator.py
```

**Step 3**: Replace each occurrence with named constant
- `24` → `PackGroupSizes.TF32_REGULAR`
- `10` → `PackGroupSizes.TF32_4X4_MFMA`
- `4 <= idx < 20` → `TF32PackIndices.MIDDLE_16_START <= idx < TF32PackIndices.MIDDLE_16_END`

**Step 4**: Add imports at top of file

**Step 5**: Run tests to verify no regressions

---

### Plan for R5: Define Typed Context

**Estimated Effort**: Small (2-4 hours)

**Step 1**: Create context.py with ValidationContext dataclass (see R5 above)

**Step 2**: Update isValid() signature
```python
def isValid(scheduleInfo: 'ScheduleInfo', context: Union[dict, ValidationContext]) -> tuple[bool, str]:
    if isinstance(context, dict):
        context = ValidationContext.from_dict(context)
    ...
```

**Step 3**: Add from_dict() class method for backward compatibility
```python
@classmethod
def from_dict(cls, d: dict) -> 'ValidationContext':
    return cls(
        kernel=d["kernel"],
        id_map=d.get("idMap")
    )
```

**Step 4**: Update internal functions to use ValidationContext
- Replace `context["kernel"]` with `context.kernel`
- Replace `context["kernel"]["SwapGlobalReadOrder"]` with `context.swap_global_read_order`

**Step 5**: Run tests

---

### Plan for R6: Use Registry Pattern for Pack Handling

**Estimated Effort**: Medium (1 day)

**Step 1**: Define PackContext dataclass and registry infrastructure
```python
# In CMSValidator.py or new pack_handlers.py
@dataclass
class PackContext:
    packs: list[Pack]
    local_reads: list[LocalRead]
    all_packs_in_loop: list[Pack]
    kernel: dict
    mfmas_by_index: dict[int, MFMA]
    mfma_reorder: list[int]
    num_vmfma: int
    loop_index: int

_PACK_HANDLERS: dict[str, Callable[[PackContext], None]] = {}

def pack_handler(mode: str):
    def decorator(fn):
        _PACK_HANDLERS[mode] = fn
        return fn
    return decorator

def get_pack_mode(kernel: dict) -> str:
    if kernel.get("UseMFMAF32XEmulation"):
        return "tf32_4x4mfma"
    if kernel.get("UseF32XEmulation"):
        return "tf32"
    return "bf16"
```

**Step 2**: Convert existing handler functions to decorated handlers
- Add `@pack_handler("bf16")` to `_hook_up_packs_bf16` (or wrapper)
- Add `@pack_handler("tf32")` to `_hook_up_packs_f32` (or wrapper)
- Add `@pack_handler("tf32_4x4mfma")` to `_hook_up_packs_f32_mfma` (or wrapper)
- Update function signatures to accept `PackContext`

**Step 3**: Update `hook_up_packs()` to use registry
```python
def hook_up_packs(timeline: Timeline, kernel: dict, mfma_reorder: list[int]) -> None:
    mode = get_pack_mode(kernel)
    if mode not in _PACK_HANDLERS:
        raise ValueError(f"Unknown pack mode: {mode}")

    mfmas_by_index = {int(m.issued_at): m for _, m in timeline.get_instructions_combined("MFMA")}

    for i_loop, loop in enumerate(timeline.loops):
        packs_by_name = _gather_packs(timeline, loop)
        all_packs_in_loop = [p for packs in packs_by_name.values() for p in packs]

        for pack_name, packs in packs_by_name.items():
            local_reads = _get_lrs_for_pack(timeline, kernel.get("UsePLRPack"), pack_name, loop)
            if not local_reads:
                continue

            ctx = PackContext(
                packs=packs,
                local_reads=local_reads,
                all_packs_in_loop=all_packs_in_loop,
                kernel=kernel,
                mfmas_by_index=mfmas_by_index,
                mfma_reorder=mfma_reorder,
                num_vmfma=timeline.num_vmfma,
                loop_index=i_loop,
            )
            _PACK_HANDLERS[mode](ctx)
```

**Step 4**: Run tests to verify no regressions

**Step 5**: (Optional) If file splitting (R1) is done, move handlers to separate files
- Each handler file imports `pack_handler` decorator and self-registers on import
- Main module imports handler files to trigger registration

---

### Plan for R7: Keep Functions (No Classes Needed)

**Estimated Effort**: None (no action required)

With the unified timeline approach (R3), the pass structure is already explicit and clear in `isValid()`. No need to introduce a formal ValidationPass interface.

The constraint-adding functions created in R3 serve as the "passes":
- `add_local_read_constraints()`
- `add_pack_constraints()`
- `add_gr_not_too_early_constraints()`
- `add_gr_finish_before_lr_constraints()`

And the existing structural check functions remain:
- `verify_correct_number_of_instructions()`
- `verify_ascending_order()`
- `verify_scc_overlap()`
- `verify_gr_inc_order()`

**Recommendation**: Skip R7. The unified timeline (R3) already provides the structure we need.

---

### Plan for R8: Centralize Error Messages

**Estimated Effort**: Small (3-4 hours total, split across 2 PRs)

This refactoring is done in two separate PRs to isolate test fixes from the main implementation.

---

#### PR1: Standardize Error Message Formats

**Goal**: Make error messages consistent across all instruction classes without extracting helper functions yet.

**Step 1**: Identify all error message patterns
```bash
grep -n "return f\"" CMSValidator.py | head -30
```

**Step 2**: Define the canonical format for each error type:

| Error Type | Canonical Format |
|------------|------------------|
| Issued too late | `{name} @ idx={issued_at} issued too late, must be issued before {needed_by_name} @ idx={needed_by_at}.` |
| Issued too early | `{name} @ idx={issued_at} issued too early, must be issued after {must_start_after_name} @ idx={must_start_after_at}.` |
| No guarantee | `{name} @ idx={issued_at} has no guarantee on when it will be done.` |
| Missing barrier | `{name} @ idx={issued_at} is missing an SBarrier. Order must be {required_order}.` |
| Quad-cycle violation | `{name} @ idx={issued_at} has insufficient gap before {needed_by_name} @ idx={needed_by_at}. Required: {required} quad-cycles, actual: {actual}.` |
| Wrong interleaving | `{name} @ idx={issued_at} has wrong interleaving. Expected {expected_name} @ idx={expected_at}, got {actual_name} @ idx={actual_at}.` |

**Step 3**: Update each class to use the canonical format:
- `LocalRead.validate()` - lines 109, 121
- `Pack.validate()` - lines 185, 190, 199, 204, 206
- `GlobalRead._validate_must_start_after()` - lines 263, 272, 277
- `GlobalRead._validate_needed_by()` - lines 296, 303, 307, 311, 314
- `SWait.validate()` - line 346
- `Barrier.validate()` - line 358

**Step 4**: Run tests
```bash
pytest Tensile/Tests/unit/test_CMSValidator*.py -v
```

**Step 5**: If tests fail due to hardcoded string expectations, update the test expectations to match the new canonical format.

**Step 6**: Create PR with title: "CMSValidator: Standardize error message formats"

---

#### PR2: Extract Helper Functions

**Goal**: Extract the now-standardized error messages into reusable helper functions.

**Step 1**: Add error message functions near top of CMSValidator.py (see R8 target state above)

**Step 2**: Update each instruction class to call the helper functions instead of inline f-strings

**Step 3**: Run tests to verify no regressions
```bash
pytest Tensile/Tests/unit/test_CMSValidator*.py -v
```

**Step 4**: Create PR with title: "CMSValidator: Extract error messages to helper functions"

---

### Plan for R9: Clarify Validation Logic Location

**Estimated Effort**: Medium (1 day)

**Step 1**: Document the chosen pattern (all validation in instruction classes)

**Step 2**: Review each instruction's validate() method
- Ensure it handles all constraints for that instruction type
- Move any external validation logic into the class

**Step 3**: Simplify standalone validation passes
- They should only: build timeline, set constraints, call validate_timeline()

**Step 4**: Add helper methods to instruction classes for constraint checking

**Step 5**: Run tests

---

### Plan for R10: Separate Timeline Responsibilities

**Estimated Effort**: Large (2-3 days)

**Step 1**: Create ScheduleParser class
- Extract instruction parsing logic from Timeline.__init__
- Handle DirectToLds, mfmaReorder, etc.

**Step 2**: Create LoopManager class
- Handle ML-1, ML, NGL, NLL iteration creation
- Handle instruction filtering per loop type

**Step 3**: Simplify Timeline class
- Accept LoopManager in constructor
- Focus on query/lookup functionality only

**Step 4**: Update all Timeline usages

**Step 5**: Run tests

---

### Plan for R11: Improve Test Infrastructure

**Estimated Effort**: Medium (1-2 days)

**Step 1**: Create tests/fixtures.py with factory classes

**Step 2**: Create ScheduleInfoBuilder for test data

**Step 3**: Create KernelFixtures with common kernel configurations

**Step 4**: Remove printWarning + skip pattern
```python
# Before
if "idMap" not in context:
    printWarning("...")
    return True, ""

# After
if context.id_map is None:
    raise ValueError("id_map required for instruction count validation")
```

**Step 5**: Update existing tests to use new fixtures

**Step 6**: Add tests for edge cases that were previously skipped

---

### Plan for R12: Document Limitations Formally

**Estimated Effort**: Small (1-2 hours)

**Step 1**: Create LIMITATIONS.md in CMSValidator directory

**Step 2**: Search for TODOs and limitations in code
```bash
grep -n "TODO\|FIXME\|not supported\|skip" CMSValidator.py
```

**Step 3**: Document each limitation with:
- Status (not supported, partially supported, known issue)
- Description
- Workaround if any
- Tracking reference

**Step 4**: Add reference to LIMITATIONS.md in module docstring

---

### Plan for R13: Standardize ValidatorInstruction Class Hierarchy

**Estimated Effort**: Medium (1 day, but best done alongside R2 and R8)

**Important**: This refactoring has strong dependencies on R2 (SchedulePosition) and R8 (error messages). The recommended approach is to implement all three together as a single coherent change, or in the order: R2 -> R13 -> R8.

---

#### Phase 1: Unify `needed_by` type on GlobalRead (can be done standalone)

**Step 1**: Change `GlobalRead.needed_by` from `float` to `ValidatorInstruction`
```python
# Before:
needed_by: float = float('inf')

# After:
needed_by: ValidatorInstruction = field(default_factory=lambda: MFMA(float('inf')))
```

**Step 2**: Update `set_gr_needed_by_from_lrs()` to assign the instruction object
```python
# Before (line 801):
for _, gr in grs:
    gr.needed_by = LR_target.issued_at

# After:
for _, gr in grs:
    gr.needed_by = LR_target
```

**Step 3**: Update `GlobalRead._validate_needed_by()` to use `self.needed_by.issued_at` and `self.needed_by.name` instead of using `self.needed_by` directly as a float
```python
# Before:
if self.needed_by == float('inf'):
if self.issued_at < self.guaranteed_by < self.needed_by:
needed_by = floor(self.needed_by) % self.num_vmfma

# After:
if self.needed_by.issued_at == float('inf'):
if self.issued_at < self.guaranteed_by < self.needed_by.issued_at:
needed_by = floor(self.needed_by.issued_at) % self.num_vmfma
```

**Step 4**: Update error messages in `_validate_needed_by()` to use `self.needed_by.name` instead of hardcoded "LR1"
```python
# Before:
f"... which is after the first corresponding LR1 @ idx={needed_by}. Order must be {name} -> SWait -> SBarrier -> LR1."

# After:
f"... which is after {self.needed_by.name} @ idx={needed_by}. Order must be {name} -> SWait -> SBarrier -> {self.needed_by.name}."
```

**Step 5**: Run tests
```bash
pytest Tensile/Tests/unit/test_CMSValidator*.py -v
```

---

#### Phase 2: Eliminate `num_vmfma` (requires R2: SchedulePosition)

This phase should be done as part of or immediately after R2.

**Step 1**: Ensure SchedulePosition (from R2) provides a `display_index` property
```python
@dataclass(frozen=True, order=True)
class SchedulePosition:
    vmfma_index: int
    sub_index: int = 0

    @property
    def display_index(self) -> int:
        return self.vmfma_index
```

**Step 2**: Replace all `floor(self.issued_at) % self.num_vmfma` with `self.issued_at.display_index` (19 occurrences)

**Step 3**: Replace all `floor(self.X.issued_at) % self.num_vmfma` patterns with `self.X.issued_at.display_index` for referenced instructions (needed_by, must_start_after, etc.)

**Step 4**: Handle the idx=-1 special case in SchedulePosition construction (in `Timeline._resolve_issued_at_indices()`) rather than in each `validate()` method

**Step 5**: Handle cross-iteration detection. Currently uses `self.needed_by.issued_at > self.num_vmfma`. Options:
- Add an `iteration` field to SchedulePosition
- Add a `is_next_iteration(self, other: SchedulePosition) -> bool` method
- Keep a separate mechanism outside the instruction classes

**Step 6**: Remove `num_vmfma` field from `LocalRead`, `Pack`, and `GlobalRead` dataclasses

**Step 7**: Remove `num_vmfma` parameter from Timeline's instruction construction calls

**Step 8**: Run tests

---

#### Phase 3: Add shared error formatting (concurrent with or after R8)

**Step 1**: Add error formatting functions as described in R8

**Step 2**: Update all `validate()` methods to call the shared functions instead of inline f-strings

**Step 3**: Verify error messages are consistent across all instruction types

**Step 4**: Run tests and update any test expectations that depend on exact error message strings

---

## Recommended Implementation Order

1. **R3: Unified Timeline** (high value, creates single Timeline with progressive constraints)
2. **R4: Extract Magic Numbers** (quick win, low risk)
3. **R5: Define Typed Context** (quick win, improves IDE support)
4. **R13: Standardize Class Hierarchy** (Phase 1: unify `needed_by` type — standalone, no dependencies; Phase 2: eliminate `num_vmfma` — after R2; Phase 3: shared error formatting — after R8)
5. **R2: Replace Float Indices** (fixes potential correctness issue, enables R13 Phase 2)
6. **R8: Centralize Error Messages** (quick win, enabled by R13 Phase 1's unified `needed_by` type)
7. **R12: Document Limitations** (quick win, documentation only)
8. **R9: Clarify Validation Logic** (already mostly done by R3, further enabled by R13)
9. **R1: Split File Into Modules** (large effort, do after other changes stabilize)
10. **R6: Registry Pattern for Packs** (medium effort, can do standalone or with R1)
11. **R10: Separate Timeline** (large effort, do last)
12. **R11: Improve Test Infrastructure** (ongoing, do incrementally)

**Note**: R7 (ValidationPass Interface) is no longer needed - the unified timeline approach (R3) provides sufficient structure using simple functions.

See `CMSValidator_UnifiedTimeline_Plan.md` for detailed implementation steps for R3.
