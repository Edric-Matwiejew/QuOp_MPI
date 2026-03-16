"""
Pytest configuration for propagator tests.

These tests are designed to run with both MPI and wavefront backends.
Use --backend option to switch backends:
    mpiexec -n 2 python -m pytest tests/ --with-mpi --backend mpi
    mpiexec -n 2 python -m pytest tests/ --with-mpi --backend wavefront
"""

import os

import pytest
from mpi4py import MPI

# Set OMP_NUM_THREADS=1 to prevent OpenMP thread contention with MPI
os.environ.setdefault("OMP_NUM_THREADS", "1")


def pytest_runtest_setup(item):
    """Skip tests that require more MPI processes than available."""
    for marker in item.iter_markers(name="requires_nprocs"):
        required_nprocs = marker.args[0]
        actual_nprocs = MPI.COMM_WORLD.Get_size()
        if actual_nprocs < required_nprocs:
            pytest.skip(
                f"Test requires {required_nprocs} MPI processes, but only {actual_nprocs} available"
            )
