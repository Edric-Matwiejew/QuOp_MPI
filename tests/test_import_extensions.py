#!/usr/bin/env python3
"""
Test that all compiled Python extension modules can be imported.

Run with:
    python tests/test_import_extensions.py

Or on a cluster with MPI:
    srun -N 1 -n 1 --gpus=1 python tests/test_import_extensions.py
"""

import importlib
import sys
import traceback

import pytest

from quop_mpi import config

MODULES_TO_TEST = [
    # Backend-agnostic
    ("quop_mpi._lib.comm_info_wrapper", "Comm info wrapper"),
    (
        f"quop_mpi._lib.{config.backend}",
        f"{config.backend} backend package",
    ),
]

ACTIVE_BACKEND_PACKAGE = f"quop_mpi._lib.{config.backend}"
ACTIVE_CONTEXT_MODULE = f"{ACTIVE_BACKEND_PACKAGE}.{config.backend}_context"


def _import_module(module_path, description):
    try:
        __import__(module_path)
        print(f"[PASS] {description}: {module_path}")
        return True
    except ImportError as e:
        print(f"[FAIL] {description}: {module_path}")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


def _import_backend_context():
    try:
        package = importlib.import_module(ACTIVE_BACKEND_PACKAGE)
        assert package.context is importlib.import_module(ACTIVE_CONTEXT_MODULE)
        print(f"[PASS] {config.backend} backend context export")
        return True
    except (ImportError, AssertionError) as e:
        print(f"[FAIL] {config.backend} backend context export")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


@pytest.mark.parametrize("module_path,description", MODULES_TO_TEST)
def test_import_extensions(module_path, description):
    assert _import_module(module_path, description)


def test_backend_package_exports_context():
    assert _import_backend_context()


def main():
    print("=" * 60)
    print("Testing Python extension module imports")
    print("=" * 60)

    results = []

    print("\n--- Required Modules ---")
    for module_path, description in MODULES_TO_TEST:
        results.append(_import_module(module_path, description))
    results.append(_import_backend_context())

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} modules imported successfully")

    if passed == total:
        print("All extension modules loaded successfully!")
        return 0
    else:
        print("Some modules failed to import. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
