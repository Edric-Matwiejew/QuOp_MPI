#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

BASE_EXTENSION_STEMS = (
    "quop_mpi/_lib/cartesian",
    "quop_mpi/_lib/comm_info_wrapper",
    "quop_mpi/_lib/csr_generators",
    "quop_mpi/_lib/parallel_io",
)

BACKEND_EXTENSION_STEMS = {
    "mpi": (
        "quop_mpi/_lib/mpi/context_wrapper",
        "quop_mpi/_lib/mpi/mpi_diagonal_propagator",
        "quop_mpi/_lib/mpi/mpi_sparse_propagator",
        "quop_mpi/_lib/mpi/mpi_circulant_propagator",
        "quop_mpi/_lib/mpi/mpi_composite_propagator",
        "quop_mpi/_lib/mpi/mpi_momentum_propagator",
        "quop_mpi/_lib/mpi/mpi_transverse_field_propagator",
    ),
    "wavefront": (
        "quop_mpi/_lib/wavefront/context_wrapper",
        "quop_mpi/_lib/wavefront/wavefront_diagonal_propagator",
        "quop_mpi/_lib/wavefront/wavefront_sparse_propagator",
        "quop_mpi/_lib/wavefront/wavefront_circulant_propagator",
        "quop_mpi/_lib/wavefront/wavefront_composite_propagator",
        "quop_mpi/_lib/wavefront/wavefront_momentum_propagator",
        "quop_mpi/_lib/wavefront/wavefront_transverse_field_propagator",
    ),
}


class ValidationError(RuntimeError):
    pass


def required_extension_stems(backend: str) -> tuple[str, ...]:
    if backend not in BACKEND_EXTENSION_STEMS:
        raise ValidationError(f"Unsupported backend for validation: {backend}")
    return BASE_EXTENSION_STEMS + BACKEND_EXTENSION_STEMS[backend]


def locate_package_init(package_name: str = "quop_mpi") -> Path:
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        raise ValidationError(
            "Installed package validation failed: could not locate "
            f"'{package_name}' in the active environment."
        )
    return Path(spec.origin).resolve()


def ensure_not_source_tree(package_init: Path, project_root: Path) -> None:
    package_init = package_init.resolve()
    source_package_init = (project_root / "src" / "quop_mpi" / "__init__.py").resolve()
    if package_init == source_package_init:
        raise ValidationError(
            "Installed package validation failed: Python would import quop_mpi "
            f"from the source tree at {package_init}. "
            f"For a standard install, avoid adding {project_root / 'src'} "
            "to PYTHONPATH when checking the installed package."
        )


def has_extension_module(package_root: Path, module_stem: str) -> bool:
    relative_stem = Path(module_stem).relative_to("quop_mpi")
    stem_path = package_root / relative_stem
    return any(Path(f"{stem_path}{suffix}").exists() for suffix in EXTENSION_SUFFIXES)


def ensure_required_extensions(package_root: Path, backend: str) -> None:
    missing = [
        module_stem
        for module_stem in required_extension_stems(backend)
        if not has_extension_module(package_root, module_stem)
    ]
    if missing:
        raise ValidationError(
            "Installed package is missing required extension modules: "
            + ", ".join(missing)
        )


def validate_installed_package(project_root: Path, backend: str) -> Path:
    package_init = locate_package_init()
    ensure_not_source_tree(package_init, project_root)
    ensure_required_extensions(package_init.parent, backend)
    return package_init


def main() -> None:
    project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
    backend = os.environ["BACKEND"]
    print(validate_installed_package(project_root, backend))


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc
