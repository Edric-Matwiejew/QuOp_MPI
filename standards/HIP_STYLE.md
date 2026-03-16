# QuOp_QUISA HIP/C++ Style Guide

This document defines HIP/C++ style conventions for GPU kernel sources and
host-side launcher code in QuOp_QUISA.

It is style-focused: naming, formatting, organization, and documentation.
Correctness, ownership/lifecycle contracts, and backend behavior requirements
are defined in `CODE_STANDARD.md`.

## Status

- This version is descriptive-first.
- New HIP/C++ code MUST follow this guide.
- Modified HIP/C++ code SHOULD be brought into compliance when touched.
- Existing inconsistencies are technical debt, not precedent.

## Scope

This guide applies to:

- `src/**/*.cpp`
- `src/**/*.hpp`
- `src/**/*.h`
- HIP kernel code compiled via HIP/CUDA language modes in CMake

This guide does not apply to:

- generated sources
- third-party/vendor code

## Keywords

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are used in the RFC 2119
sense.

## 1. File Organization

- Prefer one logical kernel family per file (vector, SpMV, reduction,
  chebyshev, wavefront-specific kernels).
- Keep device helpers in a shared header when reused across kernel families.
- Keep launcher wrappers in the same file as the corresponding kernel family.

Current baseline examples:

- `src/sparse_propagators/src/kernels/hip_vector_kernels.cpp`
- `src/sparse_propagators/src/kernels/hip_spmv_kernels.cpp`
- `src/sparse_propagators/src/kernels/hip_reduction_kernels.cpp`
- `src/sparse_propagators/src/kernels/hip_common.hpp`

## 2. Naming

### 2.1 Functions and kernels

- Kernel and helper function names SHOULD use `lowercase_with_underscores`.
- Launcher wrappers SHOULD use `launch_<kernel_name>_kernel`.
- Legacy launchers MAY keep established names until migrated.

### 2.2 Types and constants

- Use explicit-width or explicit-domain scalar types where practical
  (`size_t`, `double`, `hipDoubleComplex`, fixed-width integer types where
  width matters).
- File-local constants SHOULD be uppercase (`BLOCKSIZE`) when they are true
  constants.
- Prefer `constexpr` for compile-time constants in new code.

## 3. Interop Boundary (Fortran <-> C++)

- Fortran-callable launchers MUST use `extern "C"`.
- Launcher signatures SHOULD remain plain-C ABI friendly.
- Keep launcher naming stable once exposed through Fortran wrappers.

## 4. Includes and Header Hygiene

- Include HIP runtime and complex headers explicitly in implementation files
  that use them.
- Shared helpers SHOULD live in headers with include guards.
- Header guards MUST be uppercase and file-specific.

Current baseline:

- `src/sparse_propagators/src/kernels/hip_common.hpp` uses include guards.

## 5. Formatting

### 5.1 Indentation and braces

- Use 2-space indentation in HIP/C++ files (matches current kernel sources).
- Tabs MUST NOT be used.
- Opening braces for function definitions SHOULD remain on the same line.
- Keep brace style consistent within each file.

### 5.2 Spacing and pointer style

- Use one space after commas.
- Use spaces around binary operators.
- Pointer style SHOULD be `type* name` (current majority style).
- Keep one statement per line unless compact initialization is clearly
  beneficial.

### 5.3 Line length

- Target line length is 100-120 characters.
- Exceeding this is acceptable for kernel launcher invocations and unavoidable
  ABI-heavy signatures.

## 6. Kernel Structure and Idioms

- Use standard grid-stride loops for per-element kernels.
- Name launch geometry variables consistently (`grid_size`, `idx`, etc.).
- Use shared-memory reduction patterns consistently when reducing per-block.
- Keep synchronization points explicit and minimal (`__syncthreads()`).
- Keep index base assumptions documented in comments where relevant
  (for example, hash table positions or permutation indexing).

## 7. Comments and Documentation

- Use concise `//` comments for local intent.
- Use section separators for large kernel files where helpful.
- For non-trivial kernels, comment the mathematical operation and data layout
  assumptions.
- Comments SHOULD explain intent/invariants, not repeat obvious code.

## 8. Conditional Compilation and Portability

- Keep HIP/CUDA platform-specific logic in CMake and thin wrappers where
  possible.
- Minimize preprocessor conditionals in kernel bodies.
- When conditionals are required, keep blocks small and clearly commented.

## 9. Character Set and Encoding

- Source files MUST be UTF-8 without BOM.
- Identifiers and code tokens MUST use ASCII only.
- Comments SHOULD be ASCII by default; non-ASCII MAY be used when necessary.
- Invisible/confusable Unicode characters MUST NOT be used.

## 10. Formatter and Linter Policy

### 10.1 Formatter

- The project formatter for HIP/C++ SHOULD be `clang-format`.
- Formatting configuration is defined in `.clang-format` at repository root.
- Format checks SHOULD be limited to HIP/C++ files in scope to avoid unrelated
  churn.

Recommended default base for this project:

- `BasedOnStyle: LLVM`
- `IndentWidth: 2`
- `ColumnLimit: 110`
- `PointerAlignment: Left`

Recommended local usage:

```bash
clang-format -i src/wavefront/hip_kernels.cpp
clang-format -i src/sparse_propagators/src/kernels/*.cpp src/sparse_propagators/src/kernels/*.hpp
```

### 10.2 Linter

- Primary linting SHOULD be compiler-warning linting with HIP/C++ builds.
- `clang-tidy` MAY be added as supplemental linting once a baseline config is
  established.
- CI SHOULD treat newly introduced warnings as actionable.

### 10.3 CI policy

- CI SHOULD run format checks in non-mutating mode and fail on diffs.
- CI SHOULD run a warning-enabled HIP/C++ build configuration.
- CI MAY run `clang-tidy` on changed files once suppression policy is defined.

Example CI-style check sequence:

```bash
# 1) format check (fails if reformat is needed)
clang-format --dry-run --Werror src/wavefront/hip_kernels.cpp
clang-format --dry-run --Werror src/sparse_propagators/src/kernels/*.cpp src/sparse_propagators/src/kernels/*.hpp

# 2) warning-enabled HIP/C++ build (example)
cmake -S . -B build-hip-lint \
  -DCMAKE_CXX_FLAGS="-Wall -Wextra"
cmake --build build-hip-lint -j
```

## 11. Non-Goals

This guide does not define:

- communicator/layout architecture contracts
- ownership/lifecycle rules for buffers/communicators
- error semantics across language boundaries
- backend parity policy

Those are governed by `CODE_STANDARD.md` and architecture documents.

## 12. Adoption Checklist

For HIP/C++ code review, check:

1. Naming follows `lowercase_with_underscores` conventions.
2. Interop launchers use `extern "C"` and stable ABI-friendly signatures.
3. Formatting is consistent (2-space indent, brace style, pointer style).
4. Kernel loops follow grid-stride patterns where applicable.
5. Non-trivial kernels include intent/invariant comments.
6. Any exception is documented in the PR rationale.

## 13. Known Legacy Exceptions

The following are currently accepted until cleaned up:

- mixed constant declaration style (`const static` vs `constexpr`)
- occasional long launcher signatures
- small formatting drift in older kernel files

These SHOULD be reduced opportunistically during normal maintenance.
