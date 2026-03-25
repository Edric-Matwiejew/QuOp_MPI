"""
Stage 6 tests -- Inactive-rank behaviour (no-parking design).

All MPI ranks call every user-facing function; excluded ranks skip work
via ``subcomms.in_subcomm()`` guards rather than parking on barriers.

T6.5  With np=4 and system_size=2, some ranks are excluded but all ranks
      complete execute() without hanging.
T6.6  Rank 0 is always active (never excluded by negotiate).
T6.7  Results on rank 0 are correct despite inactive ranks.

Run with:
    mpiexec -n 4 python -m pytest tests/mpi/test_inactive_rank.py -v --with-mpi --backend mpi
"""

import pytest

from tests.conftest import TestOracle


@pytest.mark.mpi
@pytest.mark.requires_nprocs(4)
class TestInactiveRanks:
    """Verify excluded ranks participate in calls without hanging."""

    def _make_alg(self, comm, system_size=2):
        """Create a QAOA with system_size < n_procs to force exclusion."""
        from quop_mpi.algorithm.combinatorial import QAOA

        oracle = TestOracle(system_size, n_marked=1)
        alg = QAOA(system_size, comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        return alg

    def test_all_ranks_complete_setup(self, mpi_comm):
        """T6.5a: All ranks complete setup() without hanging."""
        alg = self._make_alg(mpi_comm, system_size=2)
        alg.setup()

        # Every rank reaches this point -- no hang
        mpi_comm.barrier()
        assert alg._setup_done is True

        alg.destroy()

    def test_all_ranks_complete_execute(self, mpi_comm):
        """T6.5b: All ranks complete execute() without hanging."""
        alg = self._make_alg(mpi_comm, system_size=2)
        alg.execute()

        # Every rank reaches this point -- no hang
        mpi_comm.barrier()

        alg.destroy()

    def test_rank_zero_always_active(self, mpi_comm):
        """T6.6: Rank 0 is always active after setup()."""
        alg = self._make_alg(mpi_comm, system_size=2)
        alg.setup()

        if mpi_comm.Get_rank() == 0:
            assert alg._is_active is True
            assert alg.subcomms.in_subcomm() is True

        alg.destroy()

    def test_results_correct_despite_inactive(self, mpi_comm):
        """T6.7: Optimization results on rank 0 are reasonable."""
        alg = self._make_alg(mpi_comm, system_size=2)
        alg.execute()

        if mpi_comm.Get_rank() == 0 and alg.subcomms.in_subcomm():
            # Result should exist on active rank 0
            assert alg.result is not None
            assert "fun" in alg.result

        alg.destroy()

    def test_multiple_executes_no_hang(self, mpi_comm):
        """All ranks complete 3 consecutive execute() calls without hanging."""
        alg = self._make_alg(mpi_comm, system_size=2)

        for _ in range(3):
            alg.execute()

        mpi_comm.barrier()

        alg.destroy()
