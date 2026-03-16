# QuOp_QUISA Python Style Guide

This document defines Python style conventions for QuOp_QUISA.

It is style-focused: naming, formatting, imports, typing, documentation, and
module layout. Correctness, ownership/lifecycle contracts, backend parity, and
interface semantics are defined in `CODE_STANDARD.md`.

## Status

- This version is descriptive-first.
- New Python code MUST follow this guide.
- Modified Python code SHOULD be brought into compliance when touched.
- Existing inconsistencies are technical debt, not precedent.

## Scope

This guide applies to:

- `quop_mpi/**/*.py`
- `tests/**/*.py`
- selected project Python scripts under repository root when they are part of
  normal development workflows

This guide does not apply to:

- generated code
- third-party/vendor code

## Keywords

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are used in the RFC 2119
sense.

## 1. Naming

### 1.1 General naming style

- Classes MUST use `PascalCase`.
- Functions and methods MUST use `snake_case`.
- Variables and attributes MUST use `snake_case`.
- Module names SHOULD use `snake_case`.

### 1.2 Existing module-name exceptions

The codebase currently includes public modules with capitalized filenames
(for example `Ansatz.py`, `Unitary.py`).

- These MAY remain for compatibility.
- New modules SHOULD use lowercase `snake_case` names.

### 1.3 Private naming

- Internal modules SHOULD use a leading underscore (for example `_logging.py`).
- Module-private helper symbols MAY use a leading underscore.
- Double-underscore module-private names SHOULD be used sparingly.

## 2. File Organization

Typical order:

1. module docstring (if present)
2. `from __future__ import annotations` (when used)
3. standard-library imports
4. third-party imports
5. local/package imports
6. optional `TYPE_CHECKING` imports
7. module constants/type aliases
8. classes and functions

## 3. Imports

- Imports MUST be grouped as stdlib, third-party, then local.
- `TYPE_CHECKING` imports SHOULD be used to avoid runtime import cycles.
- Keep import style consistent within a file once established.
- Avoid wildcard imports.

## 4. Typing

- New and modified public APIs SHOULD include type annotations.
- Prefer modern built-in generics (`list[int]`, `dict[str, Any]`) over
  `typing.List`/`typing.Dict`.
- Prefer PEP 604 union syntax (`A | B`) over `typing.Union[A, B]`.
- `TYPE_CHECKING` guards SHOULD be used for forward references when needed.

## 5. Docstrings and Comments

### 5.1 Docstring style

- Numpy-style docstrings are the project standard.
- Public classes/functions SHOULD include at least summary + parameters/returns
  where applicable.
- Tests MAY use shorter docstrings.

### 5.2 Comment style

- Comments SHOULD explain intent/invariants, not restate obvious code.
- Keep comments concise and close to the relevant logic.

## 6. Formatting

### 6.1 Baseline

- Use 4-space indentation.
- Tabs MUST NOT be used.
- Target line length is 100 characters.
- Longer lines MAY be accepted for unavoidable cases (long URLs,
  complex signatures, or readability-preserving test assertions).

### 6.2 Strings and layout

- Keep one statement per line.
- Use trailing commas in multiline literals/calls when it improves diff quality.
- Keep vertical whitespace purposeful and minimal.

## 7. Exceptions and Errors

- Raise the most specific built-in exception type practical (`ValueError`,
  `TypeError`, `RuntimeError`, etc.).
- Error messages SHOULD include enough context to diagnose caller misuse.

## 8. Tests

- Test files SHOULD be named `test_*.py`.
- Test functions/methods SHOULD be named `test_*`.
- Pytest markers (for example `@pytest.mark.mpi`) SHOULD be explicit where
  behavior depends on environment/runtime.
- Assertions SHOULD be clear and deterministic.
- Debug/diagnostic scripts (for example `debug_*.py`) MAY use non-test naming
  when they are not part of the standard test discovery path.

## 9. Character Set and Encoding

- Python source files MUST be UTF-8 without BOM.
- Identifiers and code tokens MUST use ASCII only.
- Comments/docstrings SHOULD be ASCII by default; non-ASCII MAY be used when
  necessary (for example, names or citations).
- Invisible/confusable Unicode characters MUST NOT be used.

## 10. Formatter and Linter Policy

### 10.1 Formatter

- Preferred formatter: `black`.
- Format target: Python files in this guide's scope.

Recommended local usage:

```bash
python -m black quop_mpi tests setup.py
```

### 10.2 Linter

- Preferred linter: `ruff`.
- Import sorting SHOULD be handled via `ruff` (isort-compatible rules).
- Lint checks SHOULD be advisory-first, then tightened over time.

Recommended local usage:

```bash
python -m ruff check quop_mpi tests setup.py
```

### 10.3 Suggested tool configuration

When ready to enforce in CI, use `pyproject.toml` settings similar to:

```toml
[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N"]
```

### 10.4 CI policy

- CI SHOULD run non-mutating checks and fail on violations.
- Suggested sequence:

```bash
python -m black --check quop_mpi tests setup.py
python -m ruff check quop_mpi tests setup.py
```

## 11. Non-Goals

This guide does not define:

- communicator/layout architecture contracts
- ownership/lifecycle rules for native handles or MPI resources
- backend behavior parity semantics
- failure policy across native/Python boundaries

Those are governed by `CODE_STANDARD.md` and architecture documents.

## 12. Adoption Checklist

For Python code review, check:

1. Naming conventions are followed.
2. Imports are grouped and ordered consistently.
3. New/changed public APIs are typed.
4. Public APIs use Numpy-style docstrings.
5. Formatting is consistent with line-length target and indentation rules.
6. Exceptions are specific and messages are actionable.
7. Test naming and markers are explicit and appropriate.

## 13. Known Legacy Exceptions

The following are currently accepted until cleaned up:

- mixed historical module filename casing (`Ansatz.py`, `Unitary.py`)
- occasional long docstring lines
- mixed use of modern and older typing idioms in untouched files

These SHOULD be reduced opportunistically during normal maintenance.
