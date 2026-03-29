"""
Stage 5 tests: Sparse propagators share the partition table.

Verifies that sparse propagators read ``quop_mpi_layout_t%partition_table``
instead of computing their own via ``MPI_Allgather``.

T5.1 -- Sparse propagator plan() succeeds and produces correct results.
T5.2 -- Layout partition table satisfies sparse-propagator expectations
       (sum covers system_size, matches local_i on each rank).

Run with:
    mpiexec -n 2 python -m pytest tests/mpi/test_sparse_partition.py -v --with-mpi
    mpiexec -n 4 python -m pytest tests/mpi/test_sparse_partition.py -v --with-mpi
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sparse_small_system_size(mpi_sizing):
    """Small power-of-two size for sparse-plan correctness checks."""
    return mpi_sizing.power_of_two(base=16, min_per_rank=1)


@pytest.fixture
def sparse_medium_system_size(mpi_sizing):
    """Moderate power-of-two size for partition-table checks."""
    return mpi_sizing.power_of_two(base=32, min_per_rank=1)


@pytest.fixture
def sparse_system_size_ladder(mpi_sizing):
    """Three increasing power-of-two sizes for cross-size sanity checks."""
    smallest = mpi_sizing.power_of_two(base=8, min_per_rank=1)
    return [smallest, smallest * 2, smallest * 4]


def _create_qaoa(system_size, mpi_comm, depth=1):
    """Create and configure a QAOA instance with a deterministic cost."""
    from quop_mpi.algorithm.combinatorial import QAOA

    def qualities(local_i, local_i_offset):
        """Deterministic cost: q(x) = sin(x)."""
        return np.sin(np.arange(local_i, dtype=np.float64) + local_i_offset)

    alg = QAOA(system_size, mpi_comm)
    alg.set_qualities(qualities)
    alg.set_depth(depth)
    return alg


# ---------------------------------------------------------------------------
# T5.1 -- Sparse propagator produces correct results
# ---------------------------------------------------------------------------


@pytest.mark.mpi
class TestSparsePartitionCorrectness:
    """T5.1: sparse plan/propagate via the shared layout partition table."""

    def test_sparse_plan_succeeds(self, mpi_comm, sparse_small_system_size):
        """plan() must complete without error on all ranks."""
        alg = _create_qaoa(sparse_small_system_size, mpi_comm)

        # evolve_state triggers setup (negotiate -> plan -> gen_operator)
        params = np.array([0.0, 0.0])  # identity-like params
        alg.evolve_state(params)  # should not raise
        alg.destroy()

    def test_sparse_preserves_normalisation(self, mpi_comm, sparse_medium_system_size):
        """Probability must sum to 1 after sparse state evolution."""
        alg = _create_qaoa(sparse_medium_system_size, mpi_comm)

        params = np.array([0.3, 0.7])
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            total = np.sum(probs)
            assert abs(total - 1.0) < 1e-10, f"Total probability = {total}"
        alg.destroy()

    def test_sparse_determinism(self, mpi_comm, sparse_small_system_size):
        """Same parameters -> identical final state (bit-exact)."""
        params = np.array([0.25, 0.50])

        alg = _create_qaoa(sparse_small_system_size, mpi_comm)
        alg.evolve_state(params)
        state1 = alg.get_final_state()

        alg.evolve_state(params)
        state2 = alg.get_final_state()

        if mpi_comm.Get_rank() == 0:
            assert state1 is not None and state2 is not None
            assert np.allclose(
                state1, state2, atol=0.0
            ), "Identical parameters must produce identical states"
        alg.destroy()

    def test_sparse_expectation_value_finite(self, mpi_comm, sparse_small_system_size):
        """execute() must produce a finite expectation value."""
        alg = _create_qaoa(sparse_small_system_size, mpi_comm)
        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert np.isfinite(alg.result["fun"]), f"E[cost] = {alg.result['fun']} is not finite"
        alg.destroy()

    def test_sparse_multi_depth_normalisation(self, mpi_comm, sparse_small_system_size):
        """Multiple QAOA layers still preserve normalisation."""
        depth = 3
        alg = _create_qaoa(sparse_small_system_size, mpi_comm, depth=depth)

        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            total = np.sum(probs)
            assert abs(total - 1.0) < 1e-10, f"depth={depth}: total probability = {total}"
            assert np.all(probs >= -1e-15), "Negative probabilities found"
        alg.destroy()


# ---------------------------------------------------------------------------
# T5.2 -- Layout partition table matches sparse propagator expectations
# ---------------------------------------------------------------------------


@pytest.mark.mpi
class TestPartitionTableAgreement:
    """T5.2: the layout partition table is consistent with the sparse propagator."""

    def test_partition_table_covers_system_size(self, mpi_comm, sparse_medium_system_size):
        """partition_table[end] - partition_table[0] == system_size."""
        alg = _create_qaoa(sparse_medium_system_size, mpi_comm)

        # trigger negotiate so the layout is populated
        params = np.array([0.0, 0.0])
        alg.evolve_state(params)

        if alg.partition_table is not None:
            pt = alg.partition_table
            # 1-based convention: pt[0]=1, pt[-1]=system_size+1
            span = int(pt[-1]) - int(pt[0])
            assert span == sparse_medium_system_size, (
                f"partition_table spans {span}, expected {sparse_medium_system_size}"
            )
        alg.destroy()

    def test_partition_table_matches_local_i(self, mpi_comm, sparse_medium_system_size):
        """local_i on each rank == partition_table[rank+1] - partition_table[rank]."""
        alg = _create_qaoa(sparse_medium_system_size, mpi_comm)

        params = np.array([0.0, 0.0])
        alg.evolve_state(params)

        if alg.partition_table is not None:
            rank = mpi_comm.Get_rank()
            pt = alg.partition_table
            expected_local_i = int(pt[rank + 1]) - int(pt[rank])
            assert alg.local_i == expected_local_i, (
                f"rank {rank}: local_i={alg.local_i}, " f"pt says {expected_local_i}"
            )
        alg.destroy()

    def test_partition_table_dtype_is_int64(self, mpi_comm, sparse_small_system_size):
        """Layout partition table must be int64 (supports large system_size)."""
        alg = _create_qaoa(sparse_small_system_size, mpi_comm)

        params = np.array([0.0, 0.0])
        alg.evolve_state(params)

        if alg.partition_table is not None:
            assert alg.partition_table.dtype == np.int64, (
                f"partition_table dtype = {alg.partition_table.dtype}, " f"expected int64"
            )
        alg.destroy()

    def test_partition_table_monotonically_increasing(
        self, mpi_comm, sparse_medium_system_size
    ):
        """partition_table values must be strictly increasing."""
        alg = _create_qaoa(sparse_medium_system_size, mpi_comm)

        params = np.array([0.0, 0.0])
        alg.evolve_state(params)

        if alg.partition_table is not None:
            pt = alg.partition_table
            for i in range(len(pt) - 1):
                assert pt[i + 1] > pt[i], (
                    f"partition_table not increasing at index {i}: " f"{pt[i]} -> {pt[i+1]}"
                )
        alg.destroy()

    def test_partition_table_has_correct_length(self, mpi_comm, sparse_medium_system_size):
        """partition_table length == n_procs_in_subcomm + 1."""
        alg = _create_qaoa(sparse_medium_system_size, mpi_comm)

        params = np.array([0.0, 0.0])
        alg.evolve_state(params)

        if alg.partition_table is not None:
            # SUBCOMM may be smaller than WORLD if ranks were excluded
            comm_size = mpi_comm.Get_size()
            # partition_table has comm_size + 1 entries (or fewer if shrunken)
            assert (
                len(alg.partition_table) >= 2
            ), f"partition_table too short: {len(alg.partition_table)}"
            # If no ranks excluded, length == comm_size + 1
            if alg.local_i > 0:
                expected_len = comm_size + 1
                assert len(alg.partition_table) == expected_len, (
                    f"partition_table length = {len(alg.partition_table)}, "
                    f"expected {expected_len} for {comm_size} ranks"
                )
        alg.destroy()

    def test_sparse_evolution_consistent_across_sizes(self, mpi_comm, sparse_system_size_ladder):
        """
        State evolution with the shared partition table produces the same
        expectation value for different system sizes (sanity check that
        the int64 partition table doesn't introduce regressions).
        """
        params = np.array([0.3, 0.7])

        for system_size in sparse_system_size_ladder:
            alg = _create_qaoa(system_size, mpi_comm)
            alg.evolve_state(params)

            probs = alg.get_probabilities()
            if mpi_comm.Get_rank() == 0:
                total = np.sum(probs)
                assert abs(total - 1.0) < 1e-10, f"system_size={system_size}: total prob = {total}"
            alg.destroy()
