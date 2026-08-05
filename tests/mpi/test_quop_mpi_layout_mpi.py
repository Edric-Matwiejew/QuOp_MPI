"""
MPI tests for QuopMpiLayout (Stage 2).

These exercise collective operations (partition_table, validate,
assert_matches_comm) across multiple MPI ranks.

Run with:
    mpiexec -n 2 python -m pytest tests/mpi/test_quop_mpi_layout_mpi.py -v --with-mpi
    mpiexec -n 4 python -m pytest tests/mpi/test_quop_mpi_layout_mpi.py -v --with-mpi
"""

import pytest
from mpi4py import MPI

from quop_mpi._utils._comm_size import QuopMpiLayout


def _block_partition(system_size, size, rank):
    """Return (local_i, offset) for a standard block distribution."""
    base = system_size // size
    remainder = system_size % size
    local_i = base + (1 if rank < remainder else 0)
    offset = sum(base + (1 if r < remainder else 0) for r in range(rank))
    return local_i, offset


def _make_layout(comm, system_size, local_i, offset, alloc_local=None):
    return QuopMpiLayout.from_partition(
        comm,
        system_size=system_size,
        local_i=local_i,
        local_i_offset=offset,
        alloc_local=alloc_local,
    )


@pytest.fixture
def layout_prime_system_size(mpi_sizing):
    """Prime size that keeps partition-table tests uneven across larger MPI jobs."""
    return mpi_sizing.prime(base=97, min_per_rank=8)


@pytest.fixture
def layout_validation_system_size(mpi_sizing):
    """Moderate layout size that scales enough to keep validation tests representative."""
    return mpi_sizing.multiple(base=200, per_rank=16)


# =============================================================================
# T2.7 -- Collective partition_table on multiple ranks
# =============================================================================


@pytest.mark.mpi
class TestPartitionTableCollective:
    """T2.7: Create QuopMpiLayout on N ranks, verify partition_table collectively."""

    def test_uniform_partition(self, mpi_comm, mpi_rank, mpi_size, layout_prime_system_size):
        system_size = layout_prime_system_size
        local_i, offset = _block_partition(system_size, mpi_size, mpi_rank)
        layout = _make_layout(mpi_comm, system_size, local_i, offset)
        layout.build_partition_table()

        table = layout.partition_table
        assert table is not None
        assert len(table) == mpi_size + 1
        assert table[0] == 1
        assert table[-1] == system_size + 1

        # Verify table is consistent: table[r+1] - table[r] == local_i for this rank
        my_span = int(table[mpi_rank + 1] - table[mpi_rank])
        assert my_span == local_i

        layout.destroy()

    @pytest.mark.requires_nprocs(2)
    def test_uneven_partition(self, mpi_comm, mpi_rank, mpi_size):
        """System size not evenly divisible -- verify remainder distribution."""
        system_size = mpi_size * 3 + 1  # guaranteed remainder
        local_i, offset = _block_partition(system_size, mpi_size, mpi_rank)
        layout = _make_layout(mpi_comm, system_size, local_i, offset)
        layout.build_partition_table()

        table = layout.partition_table
        total = sum(int(table[i + 1] - table[i]) for i in range(mpi_size))
        assert total == system_size

        layout.destroy()


# =============================================================================
# T2.8 -- assert_matches_comm
# =============================================================================


@pytest.mark.mpi
class TestAssertMatchesComm:
    """T2.8: assert_matches_comm passes with matching comm, raises with mismatch."""

    def test_matching_comm(self, mpi_comm, mpi_rank, mpi_size):
        layout = _make_layout(mpi_comm, 10 * mpi_size, 10, mpi_rank * 10)

        # Should not raise -- same communicator
        layout.assert_matches_comm(mpi_comm)

        # Should not raise -- dup is congruent
        dup = mpi_comm.Dup()
        layout.assert_matches_comm(dup)
        dup.Free()

        layout.destroy()

    @pytest.mark.requires_nprocs(2)
    def test_mismatched_comm(self, mpi_comm, mpi_rank, mpi_size):
        """Split creates a structurally different communicator -> should raise."""
        layout = _make_layout(mpi_comm, 10 * mpi_size, 10, mpi_rank * 10)

        color = mpi_rank % 2
        split_comm = mpi_comm.Split(color, mpi_rank)

        # Split comm has fewer ranks than COMM_WORLD when nprocs > 1
        # and the groups differ -> MPI_UNEQUAL
        with pytest.raises(RuntimeError, match="mismatch"):
            layout.assert_matches_comm(split_comm)

        split_comm.Free()
        layout.destroy()

    def test_null_comm_raises(self, mpi_comm, mpi_rank, mpi_size):
        layout = _make_layout(mpi_comm, 10 * mpi_size, 10, mpi_rank * 10)

        with pytest.raises(RuntimeError, match="COMM_NULL"):
            layout.assert_matches_comm(MPI.COMM_NULL)

        layout.destroy()


# =============================================================================
# T2.9 -- validate (collective)
# =============================================================================


@pytest.mark.mpi
class TestValidate:
    """T2.9: validate on good partition -> no error; bad -> ValueError."""

    def test_good_partition(self, mpi_comm, mpi_rank, mpi_size, layout_validation_system_size):
        system_size = layout_validation_system_size
        local_i, offset = _block_partition(system_size, mpi_size, mpi_rank)
        layout = _make_layout(mpi_comm, system_size, local_i, offset)

        # Build partition table (required by validate_contiguity)
        layout.build_partition_table()

        # Should not raise
        layout.validate(system_size)

        layout.destroy()

    def test_bad_completeness(self, mpi_comm, mpi_rank, mpi_size, layout_validation_system_size):
        """sum(local_i) != system_size -> ValueError on all ranks."""
        system_size = layout_validation_system_size
        # Intentionally give everyone 1 too many
        local_i, offset = _block_partition(system_size, mpi_size, mpi_rank)
        bad_local_i = local_i + 1  # sum will exceed system_size
        bad_offset = offset + mpi_rank  # shifted to match
        layout = _make_layout(mpi_comm, system_size, bad_local_i, bad_offset)

        with pytest.raises(ValueError, match="completeness"):
            layout.validate(system_size)

        layout.destroy()

    def test_bad_offset(self, mpi_comm, mpi_rank, mpi_size, layout_validation_system_size):
        """Wrong offsets -> ValueError on all ranks."""
        system_size = layout_validation_system_size
        local_i, _ = _block_partition(system_size, mpi_size, mpi_rank)
        bad_offset = 999  # deliberately wrong
        layout = _make_layout(mpi_comm, system_size, local_i, bad_offset)

        # The allreduce pattern ensures ALL ranks raise
        with pytest.raises(ValueError):
            layout.validate(system_size)

        layout.destroy()

    def test_negative_local_i(self, mpi_comm, mpi_rank, mpi_size):
        """Negative local_i -> ValueError."""
        layout = _make_layout(mpi_comm, 100, -1, 0)

        with pytest.raises(ValueError, match="non-negative"):
            layout.validate(100)

        layout.destroy()


# =============================================================================
# T2.10 -- shrink (collective)
# =============================================================================


@pytest.mark.mpi
class TestShrink:
    """T2.10: shrink keeps communicator and layout state consistent."""

    @pytest.mark.requires_nprocs(2)
    def test_shrink_repartitions_active_ranks(self, mpi_comm, mpi_rank, mpi_size):
        system_size = max(32, mpi_size * 5)
        local_i, offset = _block_partition(system_size, mpi_size, mpi_rank)
        layout = _make_layout(mpi_comm, system_size, local_i, offset)

        half = mpi_size // 2
        layout.build_partition_table()
        layout.shrink(half)

        if mpi_rank < half:
            expected_local_i, expected_offset = _block_partition(system_size, half, mpi_rank)

            assert layout.subcomm is not None
            assert layout.n_processes == half
            assert layout.local_i == expected_local_i
            assert layout.local_i_offset == expected_offset
            assert layout.alloc_local == expected_local_i

            table = layout.partition_table
            assert table is not None
            assert len(table) == half + 1
            assert table[0] == 1
            assert table[-1] == system_size + 1

            layout.validate(system_size)
        else:
            assert layout.subcomm is None
            assert layout.n_processes == 0
            assert layout.local_i == 0
            assert layout.local_i_offset == 0
            assert layout.alloc_local == 0
            assert layout.partition_table is None

        layout.destroy()
