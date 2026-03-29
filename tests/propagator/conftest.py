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
from quop_mpi import config

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


@pytest.fixture(scope="session")
def backend_name():
    """Return the active backend name for propagator test reporting."""
    return config.backend


@pytest.fixture(scope="session")
def mpi_comm():
    """MPI COMM_WORLD fixture for propagator tests."""
    return MPI.COMM_WORLD


def _scaled_grid_exponents(mpi_sizing, base_exponents):
    """Scale multivariable grid resolution while preserving dimensions."""
    exponents = [int(exponent) for exponent in base_exponents]
    extra_bits = max(0, (mpi_sizing.topology.world_size - 1).bit_length() - 1)
    for _ in range(extra_bits):
        smallest = min(exponents)
        index = exponents.index(smallest)
        exponents[index] += 1
    return exponents


@pytest.fixture
def circulant_small_system_size(small_system_size):
    """Small power-of-two size for lightweight circulant checks."""
    return small_system_size


@pytest.fixture
def circulant_medium_system_size(mpi_sizing):
    """Moderate power-of-two size for multi-rank circulant checks."""
    return mpi_sizing.power_of_two(base=32, min_per_rank=1)


@pytest.fixture
def composite_grid_ns_2d(mpi_sizing):
    """2D QMOA grid that scales while preserving dimensionality."""
    return _scaled_grid_exponents(mpi_sizing, [2, 2])


@pytest.fixture
def composite_grid_ns_1d(mpi_sizing):
    """1D QMOA grid with enough points for multi-rank FFT partitioning."""
    return _scaled_grid_exponents(mpi_sizing, [5])


@pytest.fixture
def composite_grid_ns_3d(mpi_sizing):
    """3D QMOA grid that preserves the original cubic shape."""
    return _scaled_grid_exponents(mpi_sizing, [2, 2, 2])


@pytest.fixture
def composite_grid_ns_2d_large(mpi_sizing):
    """Larger 2D QMOA grid for heavier multi-rank propagator coverage."""
    return _scaled_grid_exponents(mpi_sizing, [3, 3])
