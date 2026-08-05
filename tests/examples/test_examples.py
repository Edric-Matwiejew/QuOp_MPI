"""Pytest tests for QuOp_MPI example scripts.

Each test copies the relevant example directory to a temporary location,
runs the script via MPI, and validates results against expected bounds.
"""

import json
import subprocess
from pathlib import Path

import pytest

from .run_example_tests import (
    _build_launch_cmd,
    _example_test_env,
    run_example,
    validate_result,
)

EXPECTED_RESULTS_FILE = Path(__file__).parent / "expected_results.json"

with open(EXPECTED_RESULTS_FILE, "r") as _f:
    EXPECTED = json.load(_f)


@pytest.mark.parametrize(
    "test_name",
    list(EXPECTED.keys()),
    ids=list(EXPECTED.keys()),
)
def test_example(test_name, tmp_path, nprocs, launcher):
    """Run an example script in a temp copy and check results."""
    import shutil

    config = EXPECTED[test_name]
    src = Path(__file__).parent.parent.parent / "examples" / config["example_dir"]
    work_dir = tmp_path / config["example_dir"]
    shutil.copytree(src, work_dir)

    # Run setup script if specified (e.g., to generate data files)
    if "setup_script" in config:
        setup_result = subprocess.run(
            ["python", config["setup_script"]],
            cwd=str(work_dir),
            env=_example_test_env(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert setup_result.returncode == 0, (
            f"Setup script failed:\n{setup_result.stderr}"
        )

    result = run_example(
        config["script"], work_dir, nprocs=nprocs, launcher=launcher
    )
    errors = validate_result(result, config)

    assert result["success"], (
        f"Example failed (returncode={result['returncode']})\n{result['output']}"
    )
    assert not errors, "\n".join(errors)
