# Known Bugs and Oversights in Ansatz.py

This document catalogs bugs and control flow issues identified in `quop_mpi/Ansatz.py` during code review on 26 January 2026.

**Last Updated**: 26 January 2026 (Bug fixes applied)

---

## Critical Bugs

### 1. ✅ FIXED - Missing `self.stop` Initialization (Line 229)

**Location**: `Ansatz.__init__()`, line 229

**Issue**: The comment and code were incorrectly merged on a single line:
```python
# variables managed by the 'system' class self.stop = False  # synchronise ranks during optimisation
```

**Fix Applied**: Split into two lines:
```python
# variables managed by the 'system' class
self.stop = False  # synchronise ranks during optimisation
```

**Status**: ✅ Fixed and verified by test `test_stop_attribute_initialized`

---

### 2. Undefined `self.n_free_params`, `self.free_params`, `__place_free_params`

**Locations**: Lines 1076, 2174, 2182, 2187

**Issue**: These identifiers are used in the parallel Jacobian code but never defined anywhere in the class:
```python
for var in range(self.n_free_params):                              # line 1076
self.variational_parameters = self.__place_free_params(...)       # line 2174
partials.append(self.jacobian.call(self.free_params[var]))        # line 2182
jacobian = np.zeros(self.n_free_params, dtype=np.float64)         # line 2187
```

**Impact**: Any call to `set_parallel_jacobian()` followed by `execute()` will fail with `AttributeError`. The parallel Jacobian feature is completely broken.

**Status**: ❌ Not fixed - requires design investigation

---

### 3. ✅ FIXED - Undefined `self.rank` in Algorithm Subclasses

**Locations**: 
- `quop_mpi/algorithm/combinatorial/qaoa.py`, line 72
- `quop_mpi/algorithm/combinatorial/qwoa.py`, line 71

**Issue**: Error messages referenced `self.rank` which was never defined in `Ansatz.__init__()`:
```python
"Rank {}: Solution qualities not defined.".format(self.rank)
```

**Fix Applied**: Changed to use `self.MPI_COMM_WORLD.Get_rank()`:
```python
"Rank {}: Solution qualities not defined.".format(self.MPI_COMM_WORLD.Get_rank())
```

**Status**: ✅ Fixed and verified by tests `test_qaoa_error_message_formatting`, `test_qwoa_error_message_formatting`

---

### 4. `destroy()` Condition is Inverted (Line 1282)

**Location**: `Ansatz.destroy()`, line 1282

**Issue**: The early-return condition logic is inverted:
```python
if not self.reset or not self.setup_called:
    return
```

After `setup()` completes:
- `self.reset = False` (line 1255)
- `self.setup_called = True` (line 1256)

So the condition evaluates as: `if not False or not True` → `if True or False` → `True`

This means `destroy()` **always returns early** after setup has been called.

**Expected**: The condition should likely be:
```python
if not self.setup_called:
    return
```
Or the logic needs to be rethought entirely.

**Impact**: Resources are never freed. Memory leaks occur and MPI communicators are not released.

---

### 5. `setup_parallel` Flag Never Set to `False`

**Locations**: Lines 262, 1292-1294

**Issue**: The flag `self.setup_parallel = True` is:
- Initialized to `True` in `__init__()` (line 262)
- Reset to `True` in `destroy()` (line 1294)
- **Never set to `False` anywhere**

In `destroy()`:
```python
if not self.setup_parallel:      # Always True!
    self.__post_parallel()
    self.setup_parallel = True
```

**Impact**: `__post_parallel()` (which calls `self.subcomms.free()` to release MPI subcommunicators) is **never called**, causing MPI resource leaks.

---

## Low Severity Issues

### 6. Duplicate `self.optimiser` Assignment (Lines 220-224)

**Issue**: Copy-paste duplication:
```python
self.optimiser = (
    None  # optimiser: sp_minimize, sp_basin_hopping or nlopt_minimize
)
self.optimiser = (
    None  # optimiser: sp_minimize, sp_basin_hopping or nlopt_minimize
)
```

**Impact**: Harmless but indicates copy-paste error during development.

---

### 7. `self.samples` Immediately Overwritten (Lines 280, 285)

**Issue**: Variable assigned then immediately overwritten:
```python
self.samples = []           # line 280
# ...
self.samples = None         # line 285
```

**Impact**: Harmless but confusing; the empty list is never used.

---

### 8. `self.setup_called` Initialized Twice (Lines 242, 302)

**Issue**: Duplicate initialization:
```python
self.setup_called = False   # line 242
# ... 60 lines later ...
self.setup_called = False   # line 302
```

**Impact**: Harmless but indicates messy/unorganized initialization code.

---

## Summary Table

| # | Severity | Issue | Lines | Status |
|---|----------|-------|-------|--------|
| 1 | **Critical** | `self.stop` not initialized (merged with comment) | 229 | ✅ Fixed |
| 2 | **Critical** | `n_free_params`, `free_params`, `__place_free_params` undefined | 1076, 2174, 2182, 2187 | ❌ Not fixed |
| 3 | **Critical** | `self.rank` undefined in subclasses | qaoa.py:72, qwoa.py:71 | ✅ Fixed |
| 4 | **Critical** | `destroy()` condition inverted, cleanup never runs | 1282 | ❌ Not fixed |
| 5 | **Critical** | `setup_parallel` never `False`, `__post_parallel()` never called | 262, 1292-1294 | ❌ Not fixed |
| 6 | Low | Duplicate `self.optimiser` assignment | 220-224 | ⚪ Low priority |
| 7 | Low | `self.samples` overwritten | 280, 285 | ⚪ Low priority |
| 8 | Low | `self.setup_called` duplicate init | 242, 302 | ⚪ Low priority |

---

## Recommendations

1. **Immediate**: Fix bugs #1-5 as they cause runtime failures or resource leaks
2. **Code Quality**: Refactor `__init__()` to organize attribute initialization logically
3. **Testing**: Add unit tests for `set_parallel_jacobian()` path which is currently completely broken
4. **Architecture**: Consider simplifying the `setup_*` flag system - it's error-prone and hard to trace
