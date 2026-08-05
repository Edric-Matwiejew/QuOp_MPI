# QuOp_QUISA CMake Style Guide

This document defines style conventions for CMake files in QuOp_QUISA.

It is style-focused: file structure, naming, formatting, option layout,
function/macro conventions, and diagnostics style. Build correctness,
dependency contracts, lifecycle semantics, and backend behavior requirements are
defined in `CODE_STANDARD.md`.

## Status

- This version is descriptive-first.
- New CMake code MUST follow this guide.
- Modified CMake code SHOULD be brought into compliance when touched.
- Existing inconsistencies are technical debt, not precedent.

## Scope

This guide applies to:

- `CMakeLists.txt`
- `native/**/CMakeLists.txt`
- `native/tests/**/CMakeLists.txt`
- `cmake/**/*.cmake`

This guide does not apply to:

- generated CMake fragments in build directories
- third-party vendored build files

## Keywords

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are used in the RFC 2119
sense.

## 1. File Organization

Preferred order for `CMakeLists.txt`:

1. `cmake_minimum_required(...)`
2. `project(...)` (top-level) or local preamble (subdirs)
3. cache options/variables
4. environment/path normalization
5. dependency discovery (`find_package`, modules)
6. target definitions and target properties
7. subdirectory wiring (`add_subdirectory`)
8. testing registration (`enable_testing`, `add_test`) where applicable

Rules:

- Top-level files SHOULD use section headers to separate major phases.
- Subdirectory files SHOULD keep one coherent responsibility per directory.
- Shared helper logic SHOULD live in `cmake/*.cmake` modules when reused.

## 2. Naming

### 2.1 Variables and options

- CMake cache options SHOULD use uppercase snake case (for example,
  `WAVEFRONT_BACKEND`, `GPU_AWARE_MPI`).
- Local temporary variables SHOULD use a leading underscore (for example,
  `_hipfort_target`, `_inc_dirs`).
- Cache variable names SHOULD remain stable once public to users.

### 2.2 Functions and macros

- Helper function/macro names SHOULD use lower_snake_case.
- Function names SHOULD be action-oriented (for example,
  `compile_and_link_hipfort_mpi`).
- Prefer `function()` over `macro()` unless macro scoping is explicitly needed.

### 2.3 Targets

- Target names SHOULD be descriptive and consistent with directory purpose.
- Reused helper/object targets SHOULD use stable names across directories.

## 3. Options and Cache Variables

- User-facing configuration MUST use `option(...)` or cached `set(... CACHE ...)`.
- Options SHOULD include clear help text.
- Enumerated cache strings SHOULD define allowed values with
  `set_property(... STRINGS ...)`.
- Environment-variable fallbacks MAY be used, but command-line/cache settings
  SHOULD remain the primary source of truth.

## 4. Dependency Discovery and Linking Style

- Prefer `find_package(... REQUIRED)` for external dependencies.
- Prefer target-based linking (`target_link_libraries`) over global linker flags.
- Include directories SHOULD be attached to targets
  (`target_include_directories`) rather than global `include_directories` where
  practical.
- Build definitions/options SHOULD be attached to targets
  (`target_compile_definitions`, `target_compile_options`) where practical.

## 5. Conditionals and Platform Branches

- Platform/toolchain branches SHOULD be explicit and easy to scan
  (`if(HIP_PLATFORM STREQUAL "nvidia") ... elseif(...) ... endif()`).
- Keep nested conditionals shallow where possible.
- Extract repeated branch logic into helper functions/modules.

## 6. Diagnostics and Messages

- Use `message(STATUS ...)` for high-level configuration progress.
- Use `message(WARNING ...)` for recoverable misconfiguration.
- Use `message(FATAL_ERROR ...)` for unrecoverable configuration failures.
- Diagnostic text SHOULD include actionable remediation hints when possible.

## 7. Formatting

### 7.1 Indentation and wrapping

- Use 2 spaces for indentation in CMake blocks.
- Tabs MUST NOT be used for indentation.
- Keep argument lists vertically aligned when wrapping long commands.
- Keep one argument per line in long lists (for example long
  `target_link_libraries` lists).

### 7.2 Spacing and command style

- Use one space after command names before opening `(`.
- Prefer uppercase CMake keywords for condition operators (`STREQUAL`, `ON`,
  `OFF`) and keep variable names as defined.
- Keep command naming style consistent within a file.

### 7.3 Line length

- Target line length is 100-120 characters.
- Longer lines MAY be accepted for unavoidable path/link lines when splitting
  significantly harms readability.

## 8. Character Set and Encoding

- CMake files MUST be UTF-8 without BOM.
- Identifiers and code tokens MUST use ASCII only.
- Comments SHOULD be ASCII by default; non-ASCII MAY be used when necessary.
- Invisible/confusable Unicode characters MUST NOT be used.

## 9. CMake Modules (`cmake/*.cmake`)

- Modules SHOULD start with a short header describing purpose and outputs.
- Public module options/variables SHOULD be documented near declaration.
- Use `include_guard(GLOBAL)` for reusable modules.
- Helper functions SHOULD avoid leaking local variables into parent scope
  unless explicitly required.

## 10. Tests and CTest Integration

- Test executables SHOULD be registered using `add_test(...)`.
- MPI tests SHOULD explicitly show launcher usage and rank count semantics.
- Repeated test wiring SHOULD be factored into helper functions where practical.

## 11. Formatter and Linter Policy

### 11.1 Formatter

- Preferred formatter: `cmake-format`.
- A repository-level `.cmake-format.yaml` SHOULD be added when enforcement is
  enabled.

Recommended local usage:

```bash
cmake-format -i CMakeLists.txt native/**/CMakeLists.txt native/tests/**/CMakeLists.txt cmake/**/*.cmake
```

### 11.2 Linter

- Preferred linter: `cmakelint`.
- Linting SHOULD be advisory-first, then tightened over time.

Recommended local usage:

```bash
cmakelint CMakeLists.txt native/**/CMakeLists.txt native/tests/**/CMakeLists.txt cmake/**/*.cmake
```

### 11.3 CI policy

- CI SHOULD run non-mutating checks and fail on violations.

Suggested sequence:

```bash
cmake-format --check CMakeLists.txt native/**/CMakeLists.txt native/tests/**/CMakeLists.txt cmake/**/*.cmake
cmakelint CMakeLists.txt native/**/CMakeLists.txt native/tests/**/CMakeLists.txt cmake/**/*.cmake
```

## 12. Non-Goals

This guide does not define:

- architecture authority and ownership contracts
- backend parity requirements
- dependency policy decisions beyond style presentation
- runtime lifecycle semantics

Those are governed by `CODE_STANDARD.md` and architecture documents.

## 13. Adoption Checklist

For CMake review, check:

1. File organization follows the standard phase ordering.
2. User options/cache variables are clearly named and documented.
3. Dependency and include settings are target-based where practical.
4. Platform branches are explicit and readable.
5. Diagnostics are actionable and severity-appropriate.
6. Formatting is consistent (indentation, wrapping, line length).
7. Any exception is documented in PR rationale.

## 14. Known Legacy Exceptions

The following are currently accepted until cleaned up:

- mixed indentation style (spaces and occasional tabs) in some older files
- selective use of global directory-level include/link commands
- long platform-specific link lines

These SHOULD be reduced opportunistically during normal maintenance.
