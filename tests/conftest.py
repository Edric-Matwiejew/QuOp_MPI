"""
Shared pytest fixtures for QuOp_MPI tests.

MPI tests should be run with:
    mpiexec -n <nprocs> python -m pytest tests/mpi/
"""
import pytest
import numpy as np
from mpi4py import MPI


# =============================================================================
# MPI Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def mpi_comm():
    """MPI COMM_WORLD fixture."""
    return MPI.COMM_WORLD


@pytest.fixture(scope="session")
def mpi_rank(mpi_comm):
    """Current MPI rank."""
    return mpi_comm.Get_rank()


@pytest.fixture(scope="session")
def mpi_size(mpi_comm):
    """Total number of MPI processes."""
    return mpi_comm.Get_size()


@pytest.fixture(scope="session")
def is_root(mpi_rank):
    """True only on rank 0."""
    return mpi_rank == 0


# =============================================================================
# System Size Fixtures
# =============================================================================

@pytest.fixture
def small_system_size():
    """Small system size for quick tests (4 qubits)."""
    return 16


@pytest.fixture
def medium_system_size():
    """Medium system size (6 qubits)."""
    return 64


# =============================================================================
# Helper Functions
# =============================================================================

def mpi_barrier(comm):
    """Synchronize all MPI ranks."""
    comm.Barrier()


def assert_on_root(condition, message, comm):
    """Assert a condition, but only report from root to avoid duplicate output."""
    result = comm.gather(condition, root=0)
    if comm.Get_rank() == 0:
        assert all(result), message


def collect_to_root(value, comm):
    """Gather values from all ranks to root."""
    return comm.gather(value, root=0)
