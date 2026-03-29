"""
Stage 6 tests -- Property delegation: layout is the canonical source of truth.

T6.1 After setup(), alg.local_i == alg.layout.local_i
T6.2 After setup(), alg.partition_table identical to alg.layout.partition_table
T6.3 After destroy(), alg.layout is None
T6.4 alg.MPI_COMM is the SUBCOMM from layout

Run with:
    mpiexec -n 2 python -m pytest tests/mpi/test_property_delegation.py -v --with-mpi --backend mpi
"""

import numpy as np
import pytest

from tests.conftest import TestOracle


@pytest.fixture
def property_system_size(small_system_size):
    """Small representative size for delegation and layout-source checks."""
    return small_system_size


@pytest.mark.mpi
class TestPropertyDelegation:
    """Verify that layout is the canonical source of truth after setup()."""

    def _make_alg(self, comm, system_size):
        from quop_mpi.algorithm.combinatorial import QAOA

        oracle = TestOracle(system_size, n_marked=1)
        alg = QAOA(system_size, comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        return alg

    def test_local_i_matches_layout(self, mpi_comm, property_system_size):
        """T6.1: After setup(), alg.local_i == alg.layout.local_i."""
        alg = self._make_alg(mpi_comm, property_system_size)
        alg.setup()

        assert alg.layout is not None
        assert alg.local_i == alg.layout.local_i

        alg.destroy()

    def test_partition_table_matches_layout(self, mpi_comm, property_system_size):
        """T6.2: After setup(), alg.partition_table is identical to alg.layout.partition_table."""
        alg = self._make_alg(mpi_comm, property_system_size)
        alg.setup()

        assert alg.layout is not None
        np.testing.assert_array_equal(alg.partition_table, alg.layout.partition_table)

        alg.destroy()

    def test_layout_none_after_destroy(self, mpi_comm, property_system_size):
        """T6.3: After destroy(), alg.layout is None."""
        alg = self._make_alg(mpi_comm, property_system_size)
        alg.setup()
        assert alg.layout is not None

        alg.destroy()
        assert alg.layout is None

    def test_mpi_comm_is_subcomm(self, mpi_comm, property_system_size):
        """T6.4: alg.MPI_COMM is the SUBCOMM from layout (when active)."""
        alg = self._make_alg(mpi_comm, property_system_size)
        alg.setup()

        if alg.subcomms.in_subcomm():
            layout_subcomm = alg.layout.subcomm
            if layout_subcomm is not None:
                # Both should be the same communicator
                assert alg.MPI_COMM == layout_subcomm or (
                    alg.MPI_COMM.Get_rank() == layout_subcomm.Get_rank()
                    and alg.MPI_COMM.Get_size() == layout_subcomm.Get_size()
                )

        alg.destroy()
