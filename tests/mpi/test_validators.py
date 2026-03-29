"""
T7.1 - T7.5: Python-level integration tests for QuopMpiLayout.validate().

The validate() method is collective over SUBCOMM.  It uses an
allreduce-then-raise pattern so that all ranks raise (or none do),
preventing MPI deadlocks inside ``pytest.raises()`` blocks.

Run with:
    mpiexec -n 2 python -m pytest tests/mpi/test_validators.py -v --with-mpi --backend mpi
"""

import numpy as np
import pytest
from quop_mpi._lib.comm_info_wrapper import comm_info_wrapper as _ciw

from quop_mpi._utils._comm_size import QuopMpiLayout

# -- Helpers ----------------------------------------------------------


@pytest.fixture
def validator_system_size(mpi_sizing):
    """Scale validator tests while keeping enough local_i for mismatch corruption."""
    return mpi_sizing.multiple(base=100, per_rank=16)


def _make_good_layout(comm, system_size):
    """Create a correctly-partitioned QuopMpiLayout on *comm*.

    Splits *system_size* evenly (with remainder on last rank).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    base = system_size // size
    remainder = system_size % size
    local_i = base + (1 if rank < remainder else 0)
    offset = sum(base + (1 if r < remainder else 0) for r in range(rank))
    return QuopMpiLayout.from_partition(
        comm,
        system_size=system_size,
        local_i=local_i,
        local_i_offset=offset,
        alloc_local=local_i,
    )


# -- T7.1: Good partition validates successfully ----------------------


@pytest.mark.mpi
class TestGoodPartition:
    def test_good_partition_passes(self, mpi_comm, validator_system_size):
        """T7.1 -- A correct, evenly-distributed partition validates."""
        layout = _make_good_layout(mpi_comm, system_size=validator_system_size)
        # Should not raise
        layout.validate(validator_system_size)
        layout.destroy()


# -- T7.2: Negative local_i -> ValueError -----------------------------


@pytest.mark.mpi
class TestNegativeLocalI:
    def test_negative_local_i(self, mpi_comm, validator_system_size):
        """T7.2 -- A negative local_i triggers ValueError on ALL ranks."""
        layout = _make_good_layout(mpi_comm, system_size=validator_system_size)
        # Inject negative local_i on rank 0 only -- but allreduce
        # guarantees all ranks raise.
        if mpi_comm.Get_rank() == 0:
            _ciw.set_partitioning(layout.handle, np.int64(-5), np.int64(0))
        with pytest.raises(
            ValueError, match="non-negative check|completeness|rank_ordering|validation failed"
        ):
            layout.validate(validator_system_size)
        layout.destroy()


# -- T7.3: Sum != system_size -> ValueError ----------------------------


@pytest.mark.mpi
class TestSumMismatch:
    def test_sum_not_equal_system_size(self, mpi_comm, validator_system_size):
        """T7.3 -- When sum(local_i) != system_size, all ranks raise."""
        layout = _make_good_layout(mpi_comm, system_size=validator_system_size)
        # Reduce rank 0's local_i so the total no longer matches system_size.
        if mpi_comm.Get_rank() == 0:
            old_li = layout.local_i
            _ciw.set_partitioning(layout.handle, np.int64(old_li - 10), np.int64(0))
        with pytest.raises(ValueError, match="completeness|validation failed"):
            layout.validate(validator_system_size)
        layout.destroy()


# -- T7.4: Non-monotone offsets -> ValueError -------------------------


@pytest.mark.mpi
class TestNonMonotoneOffsets:
    def test_offsets_not_monotone(self, mpi_comm, validator_system_size):
        """T7.4 -- Non-monotone offsets (not matching cumsum) raise ValueError."""
        layout = _make_good_layout(mpi_comm, system_size=validator_system_size)
        # Corrupt the offset on rank 1 (if it exists) to be wrong.
        rank = mpi_comm.Get_rank()
        if rank == 1:
            # Set offset to 0 instead of the correct cumulative value
            _ciw.set_partitioning(layout.handle, np.int64(layout.local_i), np.int64(0))
        # On a 2-rank run: rank 1 has wrong offset -> rank_ordering fails.
        # On 1-rank run, offsets are trivially monotone so this test
        # only makes sense with >= 2 ranks.
        if mpi_comm.Get_size() >= 2:
            with pytest.raises(ValueError, match="rank_ordering|validation failed"):
                layout.validate(validator_system_size)
        else:
            layout.validate(validator_system_size)  # trivially passes with 1 rank
        layout.destroy()


# -- T7.5: Gap in partition -> ValueError ------------------------------


@pytest.mark.mpi
class TestGapInPartition:
    def test_gap_between_ranks(self, mpi_comm, validator_system_size):
        """T7.5 -- A gap in the partition (offset > expected) raises ValueError."""
        layout = _make_good_layout(mpi_comm, system_size=validator_system_size)
        rank = mpi_comm.Get_rank()
        if rank == 1:
            # Move offset forward, creating a gap between rank 0 and rank 1
            _ciw.set_partitioning(
                layout.handle,
                np.int64(layout.local_i),
                np.int64(layout.local_i_offset + 5),
            )
        if mpi_comm.Get_size() >= 2:
            with pytest.raises(ValueError, match="rank_ordering|validation failed"):
                layout.validate(validator_system_size)
        else:
            layout.validate(validator_system_size)
        layout.destroy()
