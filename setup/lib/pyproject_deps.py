#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")
DOCS_BUILD_EXCLUDED_NAMES = {"mpi4py", "h5py"}


def requirement_name(requirement: str) -> str:
    match = REQUIREMENT_NAME_RE.match(requirement)
    if match is None:
        raise SystemExit(f"Invalid requirement entry: {requirement!r}")
    return match.group(1).lower().replace("_", "-")


def load_pyproject(pyproject_path: Path) -> dict:
    with pyproject_path.open("rb") as handle:
        return tomllib.load(handle)


def dedupe_requirements(requirements: list[str]) -> list[str]:
    deduped: dict[str, str] = {}

    for requirement in requirements:
        name = requirement_name(requirement)
        deduped[name] = requirement

    return list(deduped.values())


def docs_build_requirements(pyproject: dict) -> list[str]:
    project = pyproject.get("project", {})
    runtime_requirements = list(project.get("dependencies", []))
    docs_requirements = list(project.get("optional-dependencies", {}).get("docs", []))

    combined = dedupe_requirements(runtime_requirements + docs_requirements)
    return [
        requirement
        for requirement in combined
        if requirement_name(requirement) not in DOCS_BUILD_EXCLUDED_NAMES
    ]


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "docs-build":
        raise SystemExit("usage: pyproject_deps.py docs-build <pyproject.toml>")

    pyproject_path = Path(sys.argv[2])
    pyproject = load_pyproject(pyproject_path)

    for requirement in docs_build_requirements(pyproject):
        print(requirement)


if __name__ == "__main__":
    main()
