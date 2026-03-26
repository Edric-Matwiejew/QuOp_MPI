# QuOp_QUISA Fortran Style Guide

This document defines the Fortran style conventions for QuOp_QUISA.

It is intentionally style-focused: naming, formatting, layout, declarations,
and documentation patterns. Correctness, ownership, lifecycle, backend
contracts, and portability requirements are defined in `CODE_STANDARD.md`.

## Status

- This version is descriptive-first.
- New Fortran code in `native/` and `native_tests/` MUST follow this guide.
- Modified Fortran code SHOULD be brought into compliance when touched.
- Existing inconsistencies are tracked as migration work, not precedent.

## Scope

This guide applies to:

- `native/**/*.f90`
- `native_tests/**/*.f90`
- Fortran test sources in `native/sparse_propagators/tests/src/**/*.f90`

This guide does not apply to:

- generated Fortran source
- third-party/vendor source copied into the repository

## Keywords

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are used in the RFC 2119
sense.

## 1. Naming

### 1.1 General naming style

- Modules, procedures, variables, and derived types SHOULD use
  `lowercase_with_underscores`.
- Public API names inherited from established interfaces MAY retain existing
  names until migrated.
- Constants SHOULD use uppercase names when they are true compile-time
  parameters and used as symbolic constants.

### 1.2 File and module alignment

- A source file SHOULD contain one primary module.
- File names SHOULD match primary module names where practical.
- New modules MUST use lowercase module names.

### 1.3 Migration status

All module and procedure names in the codebase now follow
`lowercase_with_underscores`. No naming-convention migrations are currently
outstanding.

## 2. File and Module Layout

Preferred module layout:

1. `module ...`
2. `use` imports
3. `implicit none`
4. visibility (`private`, then explicit `public :: ...`)
5. `contains`
6. procedures
7. `end module ...`

Rules:

- Modules MUST declare `implicit none`.
- Modules SHOULD default to `private` and export only explicit public symbols.
- `end` statements SHOULD include construct names (for example,
  `end subroutine my_subroutine`).

## 3. Imports and Kind Usage

### 3.1 `iso_fortran_env`

- Numeric kinds MUST be explicit.
- Import kinds directly from `iso_fortran_env`:
  - `real32`, `real64` (and `real128` where required)
  - `int32`, `int64`
- Kind aliases (`sp => real32`, `dp => real64`, etc.) MUST NOT be introduced
  in new code. Existing aliases have been removed.

### 3.2 Import hygiene

- Prefer `use ..., only: ...` for non-trivial modules.
- Avoid duplicate symbols in `only:` lists.
- Keep import ordering stable within a file once established.

### 3.3 Literals and kind consistency

- Real literals SHOULD use explicit kind suffixes (`1.0_real64`, `0.5_real32`).
- Complex literals SHOULD be created with explicit kind
  (`cmplx(a, b, kind=real64)` or equivalent).
- Mixed-kind arithmetic SHOULD be avoided; convert explicitly at boundaries.
- New code SHOULD avoid relying on default real kind in numerical kernels.

## 4. Declarations and Procedure Signatures

### 4.1 Dummy arguments

- All dummy arguments MUST include `intent(...)`.
- Use `intent(in)`, `intent(out)`, and `intent(inout)` precisely.
- Use `optional` only when argument omission is part of the procedure interface.

### 4.2 Declaration order

Within a procedure, use this order:

1. local `use` imports (if needed)
2. `implicit none`
3. parameters/constants
4. dummy argument declarations
5. local scalar declarations
6. local array/pointer/allocatable declarations
7. executable statements

### 4.3 Interop-specific declarations

For wrapper/interoperability code:

- Continue using explicit `iso_c_binding` types (`c_ptr`, `c_size_t`, etc.).
- Continue using `!f2py` directives in wrapper-facing procedures.
- Keep f2py comments immediately adjacent to corresponding declarations.

### 4.4 Procedure attributes

- Procedures SHOULD be marked `pure`, `elemental`, or `recursive` when that is
  semantically true.
- Use `contiguous` for assumed-shape arrays when contiguity is required.
- Use `target` only when pointer association is actually needed.
- `save` SHOULD be avoided except for intentional persistent state.

### 4.5 Interfaces and polymorphism

- `interface` and `abstract interface` blocks SHOULD be named clearly and kept
  close to the procedures/types they describe.
- Type-bound procedure names SHOULD follow the same
  `lowercase_with_underscores` convention.
- `class(*)` and broad polymorphism SHOULD be used sparingly and documented
  with expected dynamic types in comments.

### 4.6 Derived type layout

- Derived types SHOULD default to `private` components unless public component
  access is required.
- Type component declarations SHOULD be grouped by role
  (configuration/state/scratch) for readability.
- Constructor/helper procedure naming SHOULD be consistent within each module.

## 5. Formatting

### 5.1 Indentation

- Standard indentation is 4 spaces per block level.
- Tabs MUST NOT be used for indentation.
- Legacy files with 3-space indentation SHOULD be normalized when touched.

### 5.2 Continuation and long lines

- Use free-form continuation with trailing `&` and leading continuation where
  helpful for readability.
- Target line length is 120 characters.
- Exceeding 120 characters is allowed only for narrow cases where wrapping
  would significantly harm clarity (for example, long f2py directives or
  HIP-call argument lists).

### 5.3 Whitespace

- Use a single space after commas.
- Use spaces around binary operators.
- Keep vertical spacing modest and purposeful.
- Remove trailing whitespace.

### 5.4 Syntax form preferences

- Prefer `end do`, `end if`, and `end select` over legacy collapsed forms.
- Prefer one statement per line; avoid semicolon-separated statements.
- For new code, prefer bracket array constructors (`[ ... ]`) over
  `(/ ... /)`.

## 6. Arrays, Memory, and Initialization Style

- Pointer declarations SHOULD initialize to `=> null()`.
- Deallocation SHOULD be guarded with `allocated(...)` or `associated(...)`.
- Prefer explicit shape/intent declarations that make data flow and mutability
  obvious at call boundaries.

## 7. Comments and Documentation

- Use `!` for inline comments.
- Use `!>` documentation blocks for public modules/procedures and complex
  algorithms.
- Comments SHOULD explain intent and invariants, not restate obvious syntax.
- Keep copyright/license headers unchanged where present.

### 7.1 Public API documentation template

For new public procedures, documentation SHOULD include:

- Purpose
- Inputs/outputs (arguments and intent)
- Return value (if function)
- Side effects or assumptions

## 8. Conditional Compilation and Backend Blocks

- Use `#ifdef USE_HIP` blocks consistently for HIP-specific code paths.
- Keep host and HIP paths structurally parallel when possible.
- Minimize preprocessor scope to the smallest readable block.

### 8.1 Preprocessor naming and layout

- Preprocessor macro names SHOULD be uppercase with underscores.
- `#ifdef`/`#endif` blocks SHOULD be indented to match surrounding code style.
- Keep comments on conditional blocks concise and focused on why the branch
  exists.

## 9. Tests

Fortran tests should follow the same style conventions as production Fortran:

- naming
- declaration discipline
- line-length targets
- comment conventions

Test-specific assertion verbosity MAY exceed normal line-length targets when
breaking lines would significantly reduce readability of failure messages.

## 10. Formatter and Linter Policy

### 10.1 Formatter

- The project formatter for Fortran source is `fprettify`.
- `fprettify` is used to enforce layout/style consistency only.
- Formatter runs SHOULD target files in this guide's scope:
  - `native/**/*.f90`
  - `native_tests/**/*.f90`
  - `native/sparse_propagators/tests/src/**/*.f90`

Recommended local usage:

```bash
fprettify -r native native_tests
fprettify -r native/sparse_propagators/tests/src
```

### 10.2 Linter

- `fprettify` is not a linter and MUST NOT be treated as one.
- Primary linting for Fortran code is compiler-warning linting.
- CI SHOULD compile Fortran targets with warnings enabled and treat warnings
  as actionable.

Recommended warning baseline (gfortran):

```bash
-Wall -Wextra -Wimplicit-interface -Wsurprising -Wcharacter-truncation -Wconversion-extra
```

Recommended debug/test check baseline (gfortran):

```bash
-fcheck=all
```

### 10.3 CI Policy

- CI SHOULD run formatting checks in non-mutating mode and fail on diffs.
- CI SHOULD run a warning-enabled Fortran build as lint gating.
- Teams MAY add additional static tools (for example, `fortitude`) as
  supplementary checks, but compiler-warning linting remains the required
  baseline.

Example CI-style check sequence:

```bash
# 1) format check (fails if reformat is needed)
fprettify -r native native_tests native/sparse_propagators/tests/src
git diff --exit-code

# 2) warning-enabled configure/build (example flags)
cmake -S . -B build-lint \
  -DCMAKE_Fortran_FLAGS="-Wall -Wextra -Wimplicit-interface -Wsurprising -Wcharacter-truncation -Wconversion-extra"
cmake --build build-lint -j
```

### 10.4 Tooling artifacts

- The repository SHOULD include a formatter configuration artifact
  (for example, `.fprettify`) so local and CI formatting are consistent.
- Teams MAY provide a pre-commit hook for formatter and lint checks.

## 11. Character Set and Encoding

- Fortran source files MUST be encoded as UTF-8 without BOM.
- Identifiers, keywords, operators, and preprocessor symbols MUST use ASCII
  characters only.
- String literals used for runtime diagnostics, logging, and external
  interfaces SHOULD be ASCII unless non-ASCII text is required.
- Comments and documentation SHOULD be ASCII by default; non-ASCII text MAY be
  used when necessary (for example, names or citations).
- Invisible or confusable Unicode characters (for example, zero-width spaces)
  MUST NOT appear in source files.

## 12. Non-Goals

This guide does not define:

- ownership/lifecycle contracts
- error-code semantics
- communicator or layout architecture policy
- backend behavioral parity requirements

Those are governed by `CODE_STANDARD.md` and related architecture documents.

## 13. Adoption and Review Checklist

For Fortran code review, check:

1. Module/procedure naming follows lowercase convention (or is listed exception).
2. `implicit none` is present in module and procedures.
3. Dummy arguments use explicit `intent(...)`.
4. Kind usage is explicit and consistent.
5. Indentation and continuation style are consistent in touched regions.
6. Public-facing or complex procedures have sufficient `!>` or inline docs.
7. Any exception introduced is documented in the PR rationale.

## 14. Known Legacy Exceptions

The following exceptions are currently known and accepted until migration:

- A small number of lines over 120 chars in f2py directives and HIP call sites.
- Bare `use` imports (without `only:`) for external library modules
  (`hipfort`, `shafft`, `MPI`) and f2py-generated interface modules
  (`context`, `propagator`).

These exceptions SHOULD be reduced opportunistically during normal maintenance.
