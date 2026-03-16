# QuOp_MPI Code Standard

This document defines the mandatory engineering rules for the native QuOp_MPI
codebase. It applies to Fortran, HIP/C++, CMake, and Python-facing native
bindings.

This is not a style guide. A separate `STYLE_GUIDE.md` should cover formatting,
naming, comment style, and layout preferences. This document covers correctness,
portability, lifecycle, ownership, and backend-contract rules.

## Status

- New code MUST follow this document.
- Modified code SHOULD be brought into compliance when touched.
- Existing violations are technical debt, not precedent.

## Keywords

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are used in the RFC 2119
sense.

## Scope

This standard applies to:

- `src/`
- `src_tests/`
- the immediate Python adapter layer that directly wraps or normalizes native
  behavior, including `quop_mpi/_lib/` and `quop_mpi/_utils/_comm_size.py`
- native build logic in `CMakeLists.txt` and `cmake/`

It is especially important for code that affects:

- communicator creation or lifetime
- partitioning and layout negotiation
- behavior exposed through the immediate Python interface to native code
- HIP/device memory ownership
- build/link dependency resolution

For this document, the "immediate Python interface" means thin Python code that
directly owns native pointers, calls compiled entry points, or normalizes
backend behavior immediately above the compiled layer. It does not mean
higher-level algorithm or application logic.

## Architecture References

This standard is intended to align with the architectural intent described in:

- `docs/source/backend_architecture.rst`
- `quop_mpi_layout_flowchart.md`
- `comm_info_t.md`
- `docs/communicator_structure.md`
- `docs/BUILD_SYSTEM.md`

If code or docs disagree, the disagreement MUST be resolved explicitly rather
than ignored.

## Initial Standardization Priorities

The highest-priority areas to standardize, based on the current codebase, are:

1. Authority and ownership of `quop_mpi_layout_t`, `split_info_t`, contexts,
   propagators, and communicators.
2. Backend parity between MPI and wavefront for Python-visible operations.
3. MPI datatype policy for `int32`, `int64`, `real(dp)`, and `complex(dp)`.
4. Negotiation and lifecycle rules for setup, plan, destroy, and re-negotiate.
5. Target-based CMake and correct dependency/link declaration.
6. Regression-test requirements for layout, communicator, and backend changes.

## 1. Architectural Authority

### 1.1 Single source of truth

- `discover_topology()` output is the authoritative source of hardware-intrinsic
  topology.
- `split_info_t` is the authoritative source of worker-group splitting prior to
  `negotiate()`.
- `quop_mpi_layout_t` is the authoritative source of post-negotiate:
  - communicator hierarchy
  - host partitioning
  - device partitioning
  - allocation requirements
  - partition table
  - transfer geometry derived from the negotiated layout
- Contexts and propagators MAY cache layout fields for performance, but those
  cached copies MUST be treated as derived state.
- Contexts and propagators MUST NOT independently redefine layout semantics or
  recompute authoritative allocation/partition values when the layout already
  provides them.

### 1.2 Required meanings of core fields

The following meanings MUST be consistent across all backends:

- `system_size`: global logical state size
- `n_processes`: active host ranks in `SUBCOMM`
- `local_i`: host elements owned by this rank
- `local_i_offset`: global 0-based offset of the first host element on this rank
- `alloc_local`: required host allocation length for this rank
- `device_local_i`: device elements owned by this rank
- `device_local_i_offset`: global 0-based offset of the first device element
- `device_alloc_local`: required device allocation length for this rank
- `device_n_processes`: active device ranks on the local node that both own a
  device and have data

`device_n_processes` MUST NOT mean "all visible GPU ranks on the node". It MUST
mean "active ranks with both GPU access and non-zero device workload".

On the MPI backend, and on ranks where device participation is not applicable,
device-related state MUST remain defined and deterministic rather than
undefined:

- `device_n_processes = 0`
- `device_local_i = 0`
- `device_local_i_offset = 0`
- `device_alloc_local = 0`
- `DEVCOMM = MPI_COMM_NULL`
- `DEVCOMM_NODE = MPI_COMM_NULL`
- GPU-topology fields that are not meaningful off-wavefront MUST remain at
  documented zero/false defaults

### 1.3 Partition table convention

Until intentionally redesigned, the following convention is mandatory:

- `local_i_offset` is 0-based
- `partition_table` stores 1-based cumulative boundaries
- the relationship between offsets and `partition_table` MUST be documented and
  preserved across wrappers and tests

Any future redesign of this convention MUST be done project-wide and MUST
include wrapper, test, and documentation updates.

## 2. Ownership and Lifetime

### 2.1 Single owner rule

Every communicator, host allocation, device allocation, and opaque pointer MUST
have exactly one owner responsible for freeing it.

Borrowed resources MUST be documented as borrowed and MUST NOT be freed by the
borrower.

### 2.2 Communicator ownership

- `MPI_COMM` passed into QuOp_MPI is never owned by the layout and MUST NOT be
  freed by QuOp_MPI.
- `split_info_t` owns pre-negotiate worker communicators and retains ownership
  of `ROOTCOMM` and `JACCOMM`.
- Ownership of `SUBCOMM` transfers from `split_info_t` to `quop_mpi_layout_t`
  during `negotiate()`.
- `quop_mpi_layout_t` owns `SUBCOMM`, `NODECOMM`, `DEVCOMM`, and
  `DEVCOMM_NODE` after negotiation.
- Contexts borrow communicator handles from the layout and MUST NOT free them.
- Propagators borrow communicator handles from the context/layout and MUST NOT
  free them.

Communicator presence policy:

- `SUBCOMM` MUST exist on all active ranks on all backends.
- `NODECOMM` MUST exist on all active ranks on all backends.
- `DEVCOMM` and `DEVCOMM_NODE` are wavefront-specific and MUST be
  `MPI_COMM_NULL` when device participation is not applicable.

### 2.3 Memory ownership

- The context owns the state/observable/work buffers it allocates.
- Propagators MUST NOT privately allocate replacement state buffers that escape
  context ownership unless the ownership transfer is explicit and documented.
- Any device scratch requirement shared by multiple propagators MUST be
  negotiated through the layout/context contract, not hidden in one propagator.

## 3. Negotiation and Lifecycle

### 3.1 Mutation window

- Layout mutation is allowed only during negotiation and other explicitly
  defined unlocked phases.
- Once the layout is locked, backend code MUST NOT silently mutate partitioning,
  communicator membership, or allocation contracts.
- If a new operation requires layout mutation after lock, it MUST go through an
  explicit unlock/re-negotiate path.

In particular, the block decomposition is immutable after lock:

- `n_processes`
- `local_i`
- `local_i_offset`
- `alloc_local`
- `partition_table`
- device-side decomposition fields derived from the negotiated layout

These quantities MUST NOT be changed on a locked layout.

### 3.2 Derived-state updates

If code changes any part of the negotiated device distribution, it MUST update
all dependent state consistently before the layout is used:

- `device_local_i`
- `device_local_i_offset`
- `device_alloc_local`
- `device_n_processes`
- `local_i`
- `local_i_offset`
- `alloc_local`
- communicator membership if active device ranks changed
- `partition_table` if host partitioning changed and the layout is expected to
  remain finalized/consumable
- any transfer tables derived from those values

Partial updates are not allowed.

If host partitioning changes, the mutating path is responsible for rebuilding
`partition_table` before the layout is considered finalized or lockable.

### 3.3 Setup and destroy requirements

- `setup()` and `destroy()` routines MUST be safe after partial initialization.
- `destroy()` routines MUST be safe to call more than once.
- Deallocation/freeing MUST be guarded with `allocated(...)` or
  `associated(...)` checks as appropriate.
- Pointers MUST be initialized to `null()` and allocatables MUST start
  unallocated.
- Guard flags such as `planned`, `work_allocated`, and
  `reduction_allocated` SHOULD reflect actual resource state, not intent.

### 3.4 No collectives in destructors

- Destructor-like routines and final cleanup helpers MUST NOT contain MPI
  collectives or barriers unless the API is explicitly documented as a
  collective teardown entry point.
- If collective synchronization is required during teardown, it MUST happen in a
  higher-level explicit API where the collective contract is visible.

### 3.5 Allocation authority

- Context allocation sizes MUST come from the negotiated layout fields
  (`alloc_local`, `device_alloc_local`) rather than ad hoc recomputation.
- If a backend requires padded or oversized allocation for FFT/scratch reasons,
  that requirement MUST be written into the layout during negotiation.

## 4. Backend Contract Parity

### 4.1 Public semantics

MPI and wavefront backends MUST present the same semantics for equivalent
operations at the Python boundary and at shared Fortran wrapper entry points.

This includes:

- return-value visibility on root vs all active ranks
- meaning of layout fields
- error-handling behavior for equivalent failure modes
- ownership and lifetime expectations

Backend-specific optimization is allowed. Backend-specific public behavior drift
is not.

### 4.2 Wrapper contract consistency

For a shared wrapper entry point such as:

- `get_expectation_value`
- `get_state_norm`
- `get_state`
- `set_state`
- `get_observables`
- `set_observables`

all backends MUST agree on:

- whether the operation is collective
- which communicator it is collective over
- whether the result is defined on root only or on all active ranks
- what happens on excluded ranks

If the Python layer intentionally normalizes different backend internals, that
normalization MUST be documented and tested.

For scalar getter operations exposed through shared wrappers, the project
standard is:

- results MUST be defined on all active ranks in `SUBCOMM`
- backends SHOULD prefer `MPI_Allreduce` over root-only reduction followed by a
  separate broadcast

This applies in particular to:

- `get_expectation_value`
- `get_state_norm`

## 5. MPI Datatypes and Collective Rules

### 5.1 Kind-based MPI policy

MPI datatype selection MUST follow the Fortran kind of the payload, not the
host C ABI type.

Current project-wide intent:

- default `integer` payloads -> `MPI_INTEGER`
- `integer(int64)` payloads -> project-approved 64-bit integer datatype
- `real(dp)` payloads -> `MPI_DOUBLE`
- `complex(dp)` payloads -> `MPI_DOUBLE_COMPLEX`

With the current codebase, `integer(int64)` payloads should use `MPI_INTEGER8`
unless and until the project adopts a different validated 64-bit mapping.

`MPI_LONG` MUST NOT be used for `integer(int64)` payloads unless the payload is
intentionally modeling the C `long` ABI and the portability tradeoff is
explicitly documented.

### 5.2 Consistency over convenience

- A given payload kind MUST use the same MPI datatype policy across the repo.
- New code MUST NOT mix `MPI_LONG` and `MPI_INTEGER8` for logically identical
  `integer(int64)` arrays.
- If the project introduces wrapper constants/helpers for MPI kinds, new code
  MUST use them.

### 5.3 Collective documentation

Any subroutine that performs MPI collectives MUST state:

- whether it is collective
- which communicator(s) it is collective over
- whether `MPI_COMM_NULL` is allowed and how it is handled
- whether root-only outputs are valid on non-root ranks

This requirement applies to both code comments and user/developer-facing docs
for non-obvious routines.

### 5.4 Collective failure policy

- Backend/library entry points SHOULD return explicit status or error codes for
  invalid input, unsupported configuration, violated preconditions, and other
  failures that can be reported without leaving ranks in an inconsistent
  collective state.
- The immediate Python interface SHOULD translate backend status codes into
  Python exceptions or equivalent user-facing failures.
- Irrecoverable failures inside an active collective operation MAY use
  `MPI_Abort`, but only when continued execution would likely leave ranks
  inconsistent, deadlocked, or unable to recover collectively.
- Precondition and validation errors MUST NOT use `MPI_Abort` if they can be
  detected and reported before entering the unsafe collective state.
- The failure policy for a given API MUST be consistent across backends.

## 6. Error Handling and Diagnostics

### 6.1 Output channel

- Diagnostics from native code MUST use `error_unit` for errors and warnings.
- New code MUST NOT use hard-coded unit numbers such as `write(0, ...)`.

### 6.2 Error-message shape

Error messages SHOULD include, where relevant:

- subsystem or module name
- relevant communicator/rank context
- the violated invariant or failed operation

Messages SHOULD be concise and actionable.

### 6.3 Shelling out

Core library code MUST NOT invoke shell commands or `EXECUTE_COMMAND_LINE` on
normal execution paths.

Exceptions MAY be made for opt-in diagnostics or developer tooling, but they
MUST be clearly isolated from the normal library runtime and documented.

### 6.4 Environment variables

- Environment-variable parsing MUST validate values explicitly.
- Invalid values MUST produce a deterministic warning or error.
- New environment variables MUST be documented in project docs.

## 7. Fortran, C, and HIP Interop

### 7.1 `bind(C)` discipline

Every `bind(C)` interface MUST declare:

- exact C kinds
- `value` where required
- explicit `intent`
- ownership assumptions for pointer arguments

Opaque `c_ptr` handles MUST have their ownership documented at the API
boundary.

### 7.2 Interop isolation

- Pointer conversion with `transfer`, `c_f_pointer`, and `c_loc` SHOULD be kept
  close to wrapper or boundary code.
- Core business logic SHOULD operate on typed Fortran data, not raw opaque
  handles.

### 7.3 Device API discipline

- The correct device MUST be selected before device allocation, free, copy, or
  kernel launch when rank-local device state matters.
- Hot paths such as repeated propagation or repeated reductions SHOULD avoid
  per-call `hipMalloc`/`hipFree`.
- Preallocated buffers used by kernels MUST have an explicit size contract and a
  documented owner.

## 8. CMake and Build Rules

### 8.1 Target-based CMake

New build logic MUST use target-based CMake:

- `target_link_libraries`
- `target_include_directories`
- `target_compile_options`
- `set_target_properties`

Global `include_directories()` and mutable global compiler flags SHOULD be
avoided unless the setting is truly toolchain-wide and documented.

### 8.2 Correct dependency declaration

- Link actual imported targets or actual library variables.
- Boolean/status variables MUST NOT be used where a library path or target is
  required.
- Fortran module dependencies MUST be declared explicitly; build correctness
  MUST NOT depend on serialized compilation.

### 8.3 Preprocessor and module rules

- Files that require preprocessing MUST receive preprocessing flags through
  target-local build rules where practical.
- Targets producing Fortran modules MUST export module directories in a
  consistent way for downstream consumers.
- Custom wrapper-generation or custom-command patterns MUST be documented when
  they differ from normal target patterns.

## 9. Tests and Regression Requirements

### 9.1 Required test coverage

Any change that affects one of the following MUST include or update regression
coverage in the best test infrastructure currently available:

- communicator creation or lifetime
- layout negotiation
- host/device partitioning
- backend-visible semantics
- wrapper return semantics
- build/link dependency logic

### 9.2 Test placement

- For now, native regression coverage should primarily be added in `src_tests/`.
- When dedicated wrapper-level or Python-level test infrastructure is expanded,
  behavior in the immediate Python interface should gain coverage there as well.
- Changes that should behave the same across MPI and wavefront SHOULD gain
  backend-parity coverage in whatever harness can currently exercise that
  behavior.

### 9.3 Rank-count coverage

MPI tests SHOULD include at least one non-power-of-2 rank count (for example 3
or 5). Partitioning bugs frequently surface only when the global state size does
not divide evenly across ranks, so power-of-2-only testing leaves a significant
blind spot.

### 9.4 Bug-fix rule

Every bug fix in native code SHOULD add a regression test unless the failure is
not reproducible in CI or test infrastructure cannot exercise it. In that case,
the reason MUST be stated in the PR or commit message.

## 10. Documentation Requirements

### 10.1 Architecture changes

Changes to communicator hierarchy, layout-field meaning, lifecycle, or topology
handling MUST update the relevant architecture docs in the same change set.

### 10.2 Non-obvious invariants

Code MUST document non-obvious invariants at the declaration or update site.
This includes cases such as:

- why a field counts active data-owning GPU ranks instead of visible GPUs
- why padding/allocation differs from logical element count
- why a communicator exists or is rebuilt
- why a failure path must abort collectively

### 10.3 Debug features

Debug-only environment variables, dumps, and diagnostics MUST be documented and
MUST be safe to ignore in production runs.

## 11. Review Checklist for Native Changes

Any PR or change touching native code SHOULD answer these questions:

1. Which object owns each communicator and allocation touched by this change?
2. Does this change alter authoritative layout state or only derived cached
   state?
3. Are MPI datatype choices consistent with the payload kinds?
4. Does the MPI backend still match wavefront at the public API boundary?
5. Is teardown safe after partial initialization and repeated destroy calls?
6. Does this change introduce any new environment variables, and are they
   validated and documented per Section 6.4?
7. Are all required tests added or updated?
8. Do architecture/build docs need an update?

## 12. Adoption Notes

This document is expected to evolve as the codebase is cleaned up. The first
adoption work should focus on:

- replacing mixed MPI datatype usage for `int64`
- aligning backend semantics for shared wrapper APIs
- changing scalar getter semantics to all active `SUBCOMM` ranks
- enforcing idempotent cleanup paths
- creating `NODECOMM` on all backends for active ranks
- preventing post-lock mutation of negotiated block decomposition
- making layout allocation fields authoritative everywhere
- removing avoidable ownership ambiguity in communicator and buffer lifetimes
