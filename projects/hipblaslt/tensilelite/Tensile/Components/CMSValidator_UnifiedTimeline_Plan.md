# Unified Timeline Architecture Implementation Plan

This document outlines how to refactor CMSValidator to use a single Timeline object with ordered passes that progressively add constraints and validate.

---

## Overview

### Current Architecture
- 8 separate verification passes
- Each pass creates its own `Timeline` with a subset of instruction types
- Each pass applies its own transformations (apply_swaits, set_lr_needed_by, etc.)
- Each pass calls `validate_timeline()` which validates ALL instructions

### Target Architecture
- **Single Timeline** created once with ALL instruction types
- **Ordered pass functions** that add constraints to the shared Timeline
- **Single `validate_timeline()` call** after each pass (reuse existing function)
- Timeline accumulates all constraints; final state represents fully validated schedule

### Key Insight
The instruction `validate()` methods already handle unset constraints gracefully:
- `LocalRead.validate()`: `if self.needed_by.issued_at == float('inf'): return None`
- `GlobalRead._validate_must_start_after()`: `if self.must_start_after.done_idx() == float('-inf'): return None`
- `Pack.validate()`: checks against default values

This means we can call `validate_timeline()` after each pass - instructions with unset constraints pass validation, instructions with set constraints get validated.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Validation timing | Fail-fast | Stop at first error, easier debugging |
| Pass ordering | Fixed list | Simple, matches current code, small number of passes |
| Pass implementation | Functions | No classes needed, simpler, matches existing code |
| Validation method | Reuse `validate_timeline()` | Already validates all instructions correctly |

---

## PR Implementation Plan

The implementation is split into 4 self-contained PRs. Each leaves the code in a working state and can be reviewed independently.

### PR 0: Infrastructure + Signature Refactor

**Goal:** Add unified timeline infrastructure and refactor verify functions to accept external timeline.

**Changes:**

1. Add `ALL_INSTRUCTION_NAMES` constant:
```python
ALL_INSTRUCTION_NAMES = [
    "LRA0", "LRB0", "LRA1", "LRB1", "LRA3", "LRB3",
    "GRA", "GRB",
    "PackA0", "PackB0", "PackA1", "PackB1", "PackA3", "PackB3",
    "SYNC", "SNOP",
]
```

2. Add `create_unified_timeline()` function:
```python
def create_unified_timeline(
    schedule_info: 'ScheduleInfo',
    kernel: 'Solution',
    code_path: int
) -> Timeline:
    """Create a single Timeline with all instruction types."""
    available_names = set(schedule_info.optSchedule.keys())
    names_to_add = [n for n in ALL_INSTRUCTION_NAMES if n in available_names]
    return Timeline(names_to_add, code_path, schedule_info, kernel)
```

3. Refactor `verify_*` functions to accept `timeline` parameter (instead of creating internally):

**Before:**
```python
def verify_lrs_finished_before_vmfma(schedule_info, context, code_path):
    kernel = context["kernel"]
    timeline = Timeline([...LR names...], code_path, schedule_info, kernel)
    # ... validation logic ...
```

**After:**
```python
def verify_lrs_finished_before_vmfma(timeline, schedule_info, context, code_path):
    kernel = context["kernel"]
    # timeline passed in, not created here
    # ... validation logic ...
```

4. Update `isValid()` to create timelines and pass them to verify functions:
```python
def isValid(schedule_info, context):
    # ...
    # Each verify call gets its own unified timeline (for now)
    timeline = create_unified_timeline(schedule_info, kernel, code_path)
    status, msg = verify_lrs_finished_before_vmfma(timeline, schedule_info, context, code_path)

    timeline = create_unified_timeline(schedule_info, kernel, code_path)
    status, msg = verify_packs_start_and_end_at_correct_indices(timeline, schedule_info, context, code_path)
    # ... etc
```

**Tests:** All existing tests pass (behavior unchanged)

---

### PR 1: Extract All Constraint Wrapper Functions

**Goal:** Extract constraint-adding logic from all 4 verify functions into reusable wrappers. Introduce `ValidatorPassContext` so all wrappers have uniform signatures.

**Changes:**

#### 1. Add `ValidatorPassContext` dataclass:

```python
@dataclass
class ValidatorPassContext:
    """Context object containing all values needed by validator passes."""
    kernel: 'Solution'
    mfma_reorder: list[int]
    swap_global_read_order: bool
```

#### 2. `add_local_read_constraints()` from `verify_lrs_finished_before_vmfma()`

```python
def add_local_read_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add LR.needed_by and LR.guaranteed_by constraints to the provided timeline."""
    set_lr_needed_by_for_VMFMA(timeline, ctx.kernel, ctx.mfma_reorder)
    apply_swaits(timeline)  # Sets guaranteed_by for BOTH LRs and GRs
    apply_barriers(timeline)

def verify_lrs_finished_before_vmfma(timeline, schedule_info, context, code_path):
    kernel = context["kernel"]
    ctx = ValidatorPassContext(
        kernel=kernel,
        mfma_reorder=schedule_info.mfmaReorder or [],
        swap_global_read_order=kernel.get("SwapGlobalReadOrder", False),
    )
    add_local_read_constraints(timeline, ctx)
    error = validate_timeline(timeline)
    return (False, error) if error else (True, "")
```

#### 3. `add_pack_constraints()` from `verify_packs_start_and_end_at_correct_indices()`

```python
def add_pack_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add Pack.needed_by, Pack.must_start_after, and quad-cycle constraints."""
    if ctx.kernel.get("UseF32XEmulation", False) and not ctx.kernel.get("UseDirect32XEmulation", False):
        return  # Skip - not supported
    hook_up_packs(timeline, ctx.kernel, ctx.mfma_reorder)
    estimate_quad_cycles(timeline, ctx.kernel)

def verify_packs_start_and_end_at_correct_indices(timeline, schedule_info, context, code_path):
    kernel = context["kernel"]
    ctx = ValidatorPassContext(
        kernel=kernel,
        mfma_reorder=schedule_info.mfmaReorder or [],
        swap_global_read_order=kernel.get("SwapGlobalReadOrder", False),
    )
    add_pack_constraints(timeline, ctx)
    error = validate_timeline(timeline)
    return (False, error) if error else (True, "")
```

#### 4. `add_gr_not_too_early_constraints()` from `verify_grs_not_too_early()`

```python
def add_gr_not_too_early_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add GR.must_start_after and GR.must_start_after_barriered_at constraints."""
    set_gr_must_start_after_from_lr0s(timeline, ctx.swap_global_read_order)
    apply_must_start_after_barriers(timeline)

def verify_grs_not_too_early(timeline, schedule_info, context, code_path):
    kernel = context["kernel"]
    ctx = ValidatorPassContext(
        kernel=kernel,
        mfma_reorder=schedule_info.mfmaReorder or [],
        swap_global_read_order=kernel.get("SwapGlobalReadOrder", False),
    )
    apply_swaits(timeline)  # Still needed for LR.guaranteed_by
    add_gr_not_too_early_constraints(timeline, ctx)
    error = validate_timeline(timeline)
    return (False, error) if error else (True, "")
```

#### 5. `add_gr_finish_before_lr_constraints()` from `verify_grs_finish_before_lrs()`

```python
def add_gr_finish_before_lr_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add GR.needed_by and GR.barriered_at constraints."""
    set_gr_needed_by_from_lrs(timeline, ctx.swap_global_read_order)

def verify_grs_finish_before_lrs(timeline, schedule_info, context, code_path):
    kernel = context["kernel"]
    ctx = ValidatorPassContext(
        kernel=kernel,
        mfma_reorder=schedule_info.mfmaReorder or [],
        swap_global_read_order=kernel.get("SwapGlobalReadOrder", False),
    )
    apply_swaits(timeline)  # Still needed for GR.guaranteed_by
    add_gr_finish_before_lr_constraints(timeline, ctx)
    error = validate_timeline(timeline)
    return (False, error) if error else (True, "")
```

**Tests:** All existing tests pass

---

### PR 2: Share Single Unified Timeline Across Passes

**Goal:** Switch `isValid()` to create one timeline per code path and share it across all passes. Use a loop for timeline-based passes with a unified context object.

**Changes:**

1. Add `ValidatorPassContext` dataclass to hold all values needed by any pass:
```python
@dataclass
class ValidatorPassContext:
    """Context object containing all values needed by validator passes."""
    kernel: 'Solution'
    mfma_reorder: list[int]
    swap_global_read_order: bool
```

2. Update wrapper function signatures to take `(timeline, ctx)`:
```python
def add_local_read_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add LR.needed_by and LR.guaranteed_by constraints."""
    set_lr_needed_by_for_VMFMA(timeline, ctx.kernel, ctx.mfma_reorder)
    apply_swaits(timeline)
    apply_barriers(timeline)

def add_pack_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add Pack constraints."""
    if ctx.kernel.get("UseF32XEmulation", False) and not ctx.kernel.get("UseDirect32XEmulation", False):
        return
    hook_up_packs(timeline, ctx.kernel, ctx.mfma_reorder)
    estimate_quad_cycles(timeline, ctx.kernel)

def add_gr_not_too_early_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add GR.must_start_after constraints."""
    set_gr_must_start_after_from_lr0s(timeline, ctx.swap_global_read_order)
    apply_must_start_after_barriers(timeline)

def add_gr_finish_before_lr_constraints(timeline: Timeline, ctx: ValidatorPassContext) -> None:
    """Add GR.needed_by constraints."""
    set_gr_needed_by_from_lrs(timeline, ctx.swap_global_read_order)
```

3. Define the list of timeline-based constraint passes (now just functions):
```python
TIMELINE_PASSES = [
    add_local_read_constraints,
    add_pack_constraints,
    add_gr_not_too_early_constraints,
    add_gr_finish_before_lr_constraints,
]
```

4. Update `isValid()` to loop through passes:
```python
def isValid(scheduleInfo: 'ScheduleInfo', context: dict) -> tuple[bool, str]:
    """
    Return True if all validation rules pass.

    Creates a single Timeline and runs all passes in order.
    Each pass adds constraints, then validate_timeline() checks everything.
    """
    if scheduleInfo.isValidationDisabled():
        # ... existing skip logic ...
        return True, message

    kernel = context["kernel"]

    for code_path in range(scheduleInfo.numCodePaths):
        # === Structural checks (no Timeline needed yet) ===
        # Future: these can be converted to Timeline-based passes too

        structural_checks = [
            verify_correct_number_of_instructions,
            verify_ascending_order,
            verify_scc_overlap,
            verify_gr_inc_order,
        ]
        for check in structural_checks:
            status, message = check(scheduleInfo, context, code_path)
            if not status:
                return False, f"Code path {code_path}: {message}"

        # === Timeline-based checks ===

        # Create context with all values needed by passes
        ctx = ValidatorPassContext(
            kernel=kernel,
            mfma_reorder=scheduleInfo.mfmaReorder or [],
            swap_global_read_order=kernel.get("SwapGlobalReadOrder", False),
        )

        # Create single unified timeline for this code path
        timeline = create_unified_timeline(scheduleInfo, kernel, code_path)

        # Run each constraint pass and validate after each
        for add_constraints in TIMELINE_PASSES:
            add_constraints(timeline, ctx)
            if error := validate_timeline(timeline):
                return False, f"Code path {code_path}: {error}"

    return True, ""
```

**Note:** After this PR, the 4 old `verify_*` functions become dead code (but remain in codebase).

**Future Work:** The structural checks (`verify_correct_number_of_instructions`, `verify_ascending_order`, `verify_scc_overlap`, `verify_gr_inc_order`) can eventually be rewritten to use the same Timeline object and `ValidatorPassContext`, making all validation passes uniform.

**Tests:** All existing tests pass (behavior identical)

---

### PR 3: Remove Dead Code

**Goal:** Clean up the old verify functions that are no longer called.

**Changes:**

Remove the following functions:
- `verify_lrs_finished_before_vmfma()`
- `verify_packs_start_and_end_at_correct_indices()`
- `verify_grs_not_too_early()`
- `verify_grs_finish_before_lrs()`

**Tests:** All existing tests pass

---

## PR Summary

| PR | Description | Type | Size |
|----|-------------|------|------|
| 0 | Add constants, `create_unified_timeline()`, refactor verify signatures | Infrastructure + Refactor | Medium |
| 1 | Extract all 4 `add_*_constraints()` wrapper functions | Extract method | Medium |
| 2 | Share unified timeline, call wrappers directly | Architecture change | Small-Medium |
| 3 | Remove dead verify_* functions | Cleanup (pure deletion) | Small |

---

## Pass Ordering and Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Pass                        │ Adds to Timeline           │ Requires         │
├─────────────────────────────┼────────────────────────────┼──────────────────┤
│ 1. verify_correct_number    │ (no timeline)              │ (none)           │
│ 2. verify_ascending_order   │ (no timeline)              │ (none)           │
│ 3. add_local_read_constr    │ LR.needed_by               │ (none)           │
│                             │ LR.guaranteed_by           │                  │
│                             │ GR.guaranteed_by           │                  │
│ 4. add_pack_constraints     │ Pack.needed_by             │ LR.needed_by     │
│                             │ Pack.must_start_after      │                  │
│                             │ Pack.quad_cycles           │                  │
│ 5. add_gr_not_too_early     │ GR.must_start_after        │ LR.guaranteed_by │
│                             │ GR.must_start_after_barr   │                  │
│ 6. add_gr_finish_before_lr  │ GR.needed_by               │ GR.guaranteed_by │
│                             │ GR.barriered_at            │                  │
│ 7. verify_scc_overlap       │ (no timeline)              │ (none)           │
│ 8. verify_gr_inc_order      │ (no timeline)              │ (none)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Dependencies:**
- Pass 4 (`add_pack_constraints`) requires `LR.needed_by` from pass 3
- Pass 5 (`add_gr_not_too_early`) requires `LR.guaranteed_by` from pass 3
- Pass 6 (`add_gr_finish_before_lr`) requires `GR.guaranteed_by` from pass 3

**Note:** `apply_swaits` in pass 3 sets `guaranteed_by` for BOTH LocalReads and GlobalReads, so passes 5 and 6 don't need to call it again.

---

## Functions Summary

### Functions/Classes to Add
- `ALL_INSTRUCTION_NAMES` - Constant listing all instruction names (PR 0)
- `create_unified_timeline()` - Create Timeline with all instruction types (PR 0)
- `ValidatorPassContext` - Dataclass holding all values needed by passes (PR 1)
- `add_local_read_constraints(timeline, ctx)` - Wrapper calling existing functions (PR 1)
- `add_pack_constraints(timeline, ctx)` - Wrapper calling existing functions (PR 1)
- `add_gr_not_too_early_constraints(timeline, ctx)` - Wrapper calling existing functions (PR 1)
- `add_gr_finish_before_lr_constraints(timeline, ctx)` - Wrapper calling existing functions (PR 1)
- `TIMELINE_PASSES` - List of constraint pass functions (PR 2)

### Functions to Keep (unchanged)
- `validate_timeline()` - Reused after each pass
- `apply_swaits()` - Called once in pass 3
- `apply_barriers()` - Called once in pass 3
- `set_lr_needed_by_for_VMFMA()` - Called in pass 3
- `hook_up_packs()` - Called in pass 4
- `estimate_quad_cycles()` - Called in pass 4
- `set_gr_must_start_after_from_lr0s()` - Called in pass 5
- `apply_must_start_after_barriers()` - Called in pass 5
- `set_gr_needed_by_from_lrs()` - Called in pass 6
- `verify_correct_number_of_instructions()` - Pass 1
- `verify_ascending_order()` - Pass 2
- `verify_scc_overlap()` - Pass 7
- `verify_gr_inc_order()` - Pass 8

### Functions to Remove (PR 3)
- `verify_lrs_finished_before_vmfma()` - Replaced by pass 3 + validate_timeline
- `verify_packs_start_and_end_at_correct_indices()` - Replaced by pass 4 + validate_timeline
- `verify_grs_not_too_early()` - Replaced by pass 5 + validate_timeline
- `verify_grs_finish_before_lrs()` - Replaced by pass 6 + validate_timeline

---

## Benefits

1. **Single Timeline creation**: Parse schedule once, not 4 times
2. **Constraint accumulation**: Final Timeline has all constraints proven valid
3. **Simpler code**: No new classes, reuses existing `validate_timeline()`
4. **Clear pass ordering**: Dependencies documented, enforced by code order
5. **Easier debugging**: Can inspect Timeline state between passes

---

## Files to Modify

1. **CMSValidator.py**:
   - PR 0: Add constants, `create_unified_timeline()`, refactor verify signatures
   - PR 1: Add all 4 wrapper functions, update verify functions to use them
   - PR 2: Rewrite `isValid()` to use unified timeline flow
   - PR 3: Remove old `verify_*` functions

---

## Verification

After each PR, run the existing test suite to verify no regressions:

```bash
pytest Tensile/Tests/unit/test_CMSValidator*.py -v
```

The validation results should be identical to the current implementation at every step.
