"""Local pytest configuration for example tests."""

import shutil
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def pytest_addoption(parser):
    parser.addoption(
        "--nprocs",
        type=int,
        default=4,
        help="Number of MPI processes for example tests",
    )
    parser.addoption(
        "--launcher",
        default="mpiexec",
        choices=["mpiexec", "srun"],
        help="MPI launcher command (mpiexec or srun)",
    )


@pytest.fixture
def nprocs(request):
    return request.config.getoption("--nprocs")


@pytest.fixture
def launcher(request):
    return request.config.getoption("--launcher")


@pytest.fixture
def example_work_dir(request, tmp_path):
    """Copy an example directory to a temporary location and return the path."""
    example_dir_name = request.param
    src = EXAMPLES_DIR / example_dir_name
    dst = tmp_path / example_dir_name
    shutil.copytree(src, dst)
    return dst
