#!/usr/bin/env python3
"""Patch the Python source of an installed QuOp_MPI environment in place.

Copies ``*.py`` files from ``src/quop_mpi/`` in this repository over the
matching files in the installed ``quop_mpi`` package of an active QuOp
environment (one created with ``environments/install.sh``).

This is a fast iteration aid: it lets you exercise pure-Python changes
against an existing install without re-running the full build/install
pipeline.  Compiled extensions (``*.so``, ``*.pyd``), ``__pycache__``
directories, and any non-Python files are never touched -- if you have
modified Fortran/C/HIP sources you still need a real rebuild.

The installed location is discovered from environment variables set by
the activation script, in this order of preference:

* ``QUOP_VENV_DIR``      -- venv created by the QuOp installer
* ``VIRTUAL_ENV``        -- generic active venv
* ``QUOP_INSTALL_ROOT``  -- fall back to ``<root>/venv`` if it exists

Run with ``--dry-run`` to preview the file list without copying.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys
import sysconfig
from pathlib import Path

PACKAGE = "quop_mpi"
REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PKG = REPO_ROOT / "src" / PACKAGE


def _resolve_venv() -> Path:
    for var in ("QUOP_VENV_DIR", "VIRTUAL_ENV"):
        value = os.environ.get(var)
        if value:
            path = Path(value)
            if path.is_dir():
                return path

    install_root = os.environ.get("QUOP_INSTALL_ROOT")
    if install_root:
        candidate = Path(install_root) / "venv"
        if candidate.is_dir():
            return candidate

    sys.exit(
        "error: no QuOp environment detected. Source the install's "
        "activation script (or set QUOP_VENV_DIR / VIRTUAL_ENV) first."
    )


def _resolve_installed_package(venv: Path) -> Path:
    """Return the ``quop_mpi`` directory inside *venv*'s site-packages."""

    # Ask sysconfig but rebase onto the target venv so we don't rely on
    # the script being executed from inside the venv itself.
    purelib = sysconfig.get_path("purelib", vars={"base": str(venv), "platbase": str(venv)})
    candidates = [Path(purelib) / PACKAGE]

    # sysconfig sometimes points outside the venv on system Python; also
    # probe the conventional layout(s) directly.
    for lib_dir in sorted((venv / "lib").glob("python*")):
        candidates.append(lib_dir / "site-packages" / PACKAGE)
    for lib_dir in sorted((venv / "Lib").glob("site-packages")):  # Windows-ish
        candidates.append(lib_dir / PACKAGE)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    sys.exit(
        f"error: could not find installed '{PACKAGE}' package under {venv}.\n"
        "Searched:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def _iter_python_sources(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _plan(source_root: Path, dest_root: Path):
    plan = []
    for src in _iter_python_sources(source_root):
        rel = src.relative_to(source_root)
        dst = dest_root / rel
        if not dst.exists():
            status = "new"
        elif filecmp.cmp(src, dst, shallow=False):
            status = "same"
        else:
            status = "update"
        plan.append((rel, src, dst, status))
    plan.sort(key=lambda item: item[0].as_posix())
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy QuOp_MPI Python sources into an active install.",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="Show what would be copied without modifying the install.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="List unchanged files as well.",
    )
    parser.add_argument(
        "--prune-pycache", action="store_true",
        help="Delete __pycache__ directories under the installed package "
             "after copying so stale .pyc files do not shadow updates.",
    )
    args = parser.parse_args()

    if not SOURCE_PKG.is_dir():
        sys.exit(f"error: source package not found at {SOURCE_PKG}")

    venv = _resolve_venv()
    installed = _resolve_installed_package(venv)

    print(f"source : {SOURCE_PKG}")
    print(f"target : {installed}")
    print(f"venv   : {venv}")
    print()

    plan = _plan(SOURCE_PKG, installed)
    if not plan:
        print("no .py files found in source package; nothing to do")
        return 0

    counts = {"new": 0, "update": 0, "same": 0}
    for rel, src, dst, status in plan:
        counts[status] += 1
        if status == "same" and not args.verbose:
            continue
        marker = {"new": "+", "update": "~", "same": "="}[status]
        print(f"  {marker} {rel.as_posix()}")

        if status == "same" or args.dry_run:
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    print()
    print(
        f"summary: {counts['update']} updated, {counts['new']} new, "
        f"{counts['same']} unchanged"
        + (" (dry run)" if args.dry_run else "")
    )

    if args.prune_pycache and not args.dry_run:
        removed = 0
        for cache in installed.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
            removed += 1
        print(f"pruned {removed} __pycache__ director{'y' if removed == 1 else 'ies'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
