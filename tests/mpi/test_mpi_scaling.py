"""
Tests for MPI scaling behavior across different process counts.

These tests help identify and isolate MPI-related issues that may occur
when running with different numbers of MPI processes.

Run with: mpiexec -n <N> python -m pytest tests/mpi/test_mpi_scaling.py -v --with-mpi

Known issue: Deadlocks have been observed with N > 2 processes.
These tests are designed to help isolate the root cause.
"""

import numpy as np
import pytest
from mpi4py import MPI

from tests.conftest import TestOracle


def get_mpi_info():
    """Return MPI rank and size."""
    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()


@pytest.fixture
def simple_oracle(mpi_sizing):
    """Scale the default oracle with topology while preserving problem density."""
    system_size = mpi_sizing.power_of_two(
        base=64,
        min_per_rank=1,
        min_per_node=16,
        min_per_gpu=8,
    )
    # Preserve the original 1/16 marked-state density from N=64, M=4.
    n_marked = max(4, system_size // 16)
    return TestOracle(system_size=system_size, n_marked=n_marked, seed=42)


@pytest.mark.mpi
class TestMPIBasicOperations:
    """Test basic MPI operations to ensure environment is working."""

    def test_mpi_comm_world_accessible(self, mpi_comm):
        """Verify MPI communicator is accessible."""
        assert mpi_comm is not None
        assert mpi_comm.Get_size() >= 1

    def test_mpi_barrier_works(self, mpi_comm):
        """Verify MPI barrier completes without deadlock."""
        mpi_comm.Barrier()

    def test_mpi_gather_works(self, mpi_comm):
        """Verify MPI gather completes without deadlock."""
        rank = mpi_comm.Get_rank()
        data = rank * 10

        result = mpi_comm.gather(data, root=0)

        if rank == 0:
            assert len(result) == mpi_comm.Get_size()
            for i, val in enumerate(result):
                assert val == i * 10

    def test_mpi_allgather_works(self, mpi_comm):
        """Verify MPI allgather completes without deadlock."""
        rank = mpi_comm.Get_rank()
        data = rank * 10

        result = mpi_comm.allgather(data)

        assert len(result) == mpi_comm.Get_size()


@pytest.mark.mpi
class TestAnsatzInitialization:
    """Test Ansatz initialization with various process counts."""

    def test_ansatz_creation(self, mpi_comm, medium_system_size):
        """Verify Ansatz can be created."""
        from quop_mpi import Ansatz

        # All ranks should be able to create an Ansatz
        alg = Ansatz(medium_system_size, mpi_comm)

        assert alg.system_size == medium_system_size
        alg.destroy()

    def test_qaoa_creation(self, mpi_comm, medium_system_size):
        """Verify QAOA can be created."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(medium_system_size, mpi_comm)

        assert alg.system_size == medium_system_size
        alg.destroy()

    def test_qwoa_creation(self, mpi_comm, medium_system_size):
        """Verify QWOA can be created."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(medium_system_size, mpi_comm)

        assert alg.system_size == medium_system_size
        alg.destroy()


@pytest.mark.mpi
class TestQualitiesSetting:
    """Test setting qualities/observables with various process counts."""

    def test_set_qualities_qaoa(self, mpi_comm, simple_oracle):
        """Verify set_qualities completes for QAOA."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.destroy()

    def test_set_qualities_qwoa(self, mpi_comm, simple_oracle):
        """Verify set_qualities completes for QWOA."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.destroy()

    def test_set_depth(self, mpi_comm, simple_oracle):
        """Verify set_depth completes."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(2)
        alg.destroy()


@pytest.mark.mpi
class TestSetupPhase:
    """Test the setup phase which initializes MPI subcommunicators."""

    def test_setup_completes_qaoa(self, mpi_comm, simple_oracle):
        """Verify QAOA setup completes without deadlock."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # This is often where subcommunicator issues arise
        alg.setup()

        assert alg.setup_called
        alg.destroy()

    def test_setup_completes_qwoa(self, mpi_comm, simple_oracle):
        """Verify QWOA setup completes without deadlock."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        alg.setup()

        assert alg.setup_called
        alg.destroy()

    def test_subcomms_after_setup(self, mpi_comm, simple_oracle):
        """Examine subcommunicator state after setup."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()

        rank = mpi_comm.Get_rank()

        # Gather diagnostic info about subcomms
        in_subcomm = alg.subcomms.in_subcomm()
        subcomm_index = alg.subcomms.get_subcomm_index() if in_subcomm else -1

        # Collect info from all ranks
        all_in_subcomm = mpi_comm.gather(in_subcomm, root=0)
        all_subcomm_index = mpi_comm.gather(subcomm_index, root=0)

        if rank == 0:
            # All ranks should be in a subcomm for simple cases
            assert all(all_in_subcomm), f"Not all ranks in subcomm: {all_in_subcomm}"
            # All should be in subcomm index 0
            assert all(
                idx == 0 for idx in all_subcomm_index
            ), f"Ranks in different subcomm indices: {all_subcomm_index}"
        alg.destroy()


@pytest.mark.mpi
class TestEvolveStatePhase:
    """Test evolve_state which may involve MPI communication."""

    def test_evolve_state_completes(self, mpi_comm, simple_oracle):
        """Verify evolve_state completes without deadlock."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)

        # evolve_state calls setup internally
        alg.evolve_state(params)
        alg.destroy()


@pytest.mark.mpi
class TestExecutePhase:
    """Test execute which runs the full optimization."""

    def test_execute_completes(self, mpi_comm, simple_oracle):
        """Verify execute completes without deadlock."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # This is where deadlock is observed with > 2 processes
        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
        alg.destroy()


@pytest.mark.mpi
class TestGatherPhase:
    """Test state gathering operations."""

    def test_get_probabilities_after_evolve(self, mpi_comm, simple_oracle):
        """Verify get_probabilities works after evolve_state."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        # This involves MPI gather
        probs = alg.get_probabilities()

        if mpi_comm.Get_rank() == 0:
            assert probs is not None
            assert len(probs) == simple_oracle.system_size
        alg.destroy()

    def test_get_final_state_after_evolve(self, mpi_comm, simple_oracle):
        """Verify get_final_state works after evolve_state."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        # This involves MPI gather
        state = alg.get_final_state()

        if mpi_comm.Get_rank() == 0:
            assert state is not None
            assert len(state) == simple_oracle.system_size
        alg.destroy()


@pytest.mark.mpi
class TestSystemSizeScaling:
    """Test with various system sizes relative to process count."""

    def test_system_size_equal_to_nprocs(self, mpi_comm, mpi_sizing):
        """Test when system_size equals number of processes."""
        from quop_mpi.algorithm.combinatorial import QAOA

        system_size = mpi_sizing.power_of_two(base=2, min_per_rank=1)

        oracle = TestOracle(system_size=system_size, n_marked=1, seed=42)

        alg = QAOA(system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert np.isfinite(alg.result["fun"])
            assert len(alg.result["x"]) == alg.n_free_params
        alg.destroy()

    def test_system_size_less_than_nprocs(self, mpi_comm, mpi_sizing):
        """Test when system_size is less than number of processes."""
        from quop_mpi.algorithm.combinatorial import QAOA

        size = mpi_comm.Get_size()
        if size <= 2:
            pytest.skip("Need more than 2 processes for this test")

        # Use the largest power of two below the world size to force exclusion.
        system_size = mpi_sizing.below_world_power_of_two(minimum=2)

        oracle = TestOracle(system_size=system_size, n_marked=1, seed=42)

        alg = QAOA(system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()
        alg.destroy()

    def test_system_size_much_larger_than_nprocs(self, mpi_comm, mpi_sizing):
        """Test when system_size is much larger than number of processes."""
        from quop_mpi.algorithm.combinatorial import QAOA

        system_size = mpi_sizing.power_of_two(base=32, min_per_rank=4, min_per_node=16)

        oracle = TestOracle(system_size=system_size, n_marked=4, seed=42)

        alg = QAOA(system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()
        alg.destroy()


@pytest.mark.mpi
class TestPartitionTable:
    """Test partition table correctness across processes."""

    def test_partition_table_covers_all_elements(self, mpi_comm, simple_oracle):
        """Verify partition table correctly distributes all elements."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()

        # Get partition info from Ansatz (not unitaries)
        partition_table = alg.partition_table

        # Gather partition tables from all ranks
        all_tables = mpi_comm.gather(
            list(partition_table) if partition_table is not None else None, root=0
        )

        if mpi_comm.Get_rank() == 0:
            # Filter out None values (ranks not in subcomm)
            valid_tables = [t for t in all_tables if t is not None]

            if len(valid_tables) > 0:
                # All ranks in subcomm should have the same partition table
                for table in valid_tables:
                    assert table == valid_tables[0], "Partition tables differ across ranks"

                # Partition table should account for all elements
                # partition_table uses 1-based Fortran indexing with cumsum + 1
                # So partition_table[-1] = system_size + 1
                assert (
                    valid_tables[0][-1] == simple_oracle.system_size + 1
                ), (
                    f"Partition table end {valid_tables[0][-1]} "
                    f"!= system size + 1 ({simple_oracle.system_size + 1})"
                )
        alg.destroy()

    def test_local_i_and_offset_consistency(self, mpi_comm, simple_oracle):
        """Verify local_i and local_i_offset are consistent across ranks."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()

        rank = mpi_comm.Get_rank()

        # Get local partition info from Ansatz
        local_i = alg.local_i
        local_i_offset = alg.local_i_offset

        # Gather from all ranks
        all_local_i = mpi_comm.gather(local_i, root=0)
        all_offsets = mpi_comm.gather(local_i_offset, root=0)

        if rank == 0:
            # Filter out None values
            valid_pairs = [
                (li, off)
                for li, off in zip(all_local_i, all_offsets, strict=False)
                if li is not None and off is not None
            ]

            if len(valid_pairs) > 0:
                # Sort by offset to verify continuity
                valid_pairs.sort(key=lambda x: x[1])

                # Verify offsets are cumulative sums of local_i
                expected_offset = 0
                for li, offset in valid_pairs:
                    assert (
                        offset == expected_offset
                    ), f"Offset {offset} != expected {expected_offset}"
                    expected_offset += li

                # Total should equal system size
                assert (
                    expected_offset == simple_oracle.system_size
                ), f"Total elements {expected_offset} != system size"
        alg.destroy()


@pytest.mark.mpi
class TestEdgeCaseSizes:
    """Test edge cases with unusual system sizes and process configurations."""

    def test_prime_system_size_qwoa(self, mpi_comm, mpi_sizing):
        """Test QWOA with prime system size (circulant supports any size)."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = mpi_sizing.prime(base=17, min_per_rank=1, min_per_node=8)

        def qualities(local_i, local_i_offset):
            return np.random.RandomState(42 + local_i_offset).random(local_i)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        alg.execute()

        # All ranks must call get_probabilities (it's a collective operation)
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            # Sum of probabilities should be 1
            np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10)
            # State should have evolved from uniform - check variance is non-zero
            assert np.std(probs) > 1e-6, "State did not evolve from uniform"

        alg.destroy()

    def test_larger_prime_system_size_qwoa(self, mpi_comm, mpi_sizing):
        """Test QWOA with larger prime system size."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = mpi_sizing.prime(base=23, min_per_rank=2, min_per_node=12)

        def qualities(local_i, local_i_offset):
            return np.random.RandomState(42 + local_i_offset).random(local_i)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        alg.execute()

        # All ranks must call get_probabilities (it's a collective operation)
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10)
            # State should have evolved from uniform - check variance is non-zero
            assert np.std(probs) > 1e-6, "State did not evolve from uniform"

        alg.destroy()

    def test_system_size_one_element_per_rank(self, mpi_comm):
        """Test when system size equals exactly number of processes."""
        from quop_mpi.algorithm.combinatorial import QWOA

        size = mpi_comm.Get_size()
        system_size = size  # Exactly one element per rank

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        alg.execute()

        # Total local_i across all ranks should equal system_size
        # Note: FFTW MPI may not give exactly 1 element per rank
        total_local_i = mpi_comm.allreduce(alg.local_i if alg.subcomms.in_subcomm() else 0)
        assert total_local_i == system_size

        # Verify result is valid - all ranks must call get_probabilities (collective)
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10)
            uniform_prob = 1.0 / system_size
            if system_size == 1:
                # A 1-element system cannot evolve - state must remain uniform
                assert np.allclose(probs, uniform_prob), "1-element state should remain uniform"
            else:
                # State should have evolved from uniform - not all probabilities equal
                assert not np.allclose(probs, uniform_prob), "State did not evolve from uniform"

        alg.destroy()

    def test_system_size_smaller_than_nprocs(self, mpi_comm, mpi_sizing):
        """Test when system size is smaller than number of processes.

        This tests that ranks are correctly excluded when system_size < nprocs,
        and verifies the fix for FFTW MPI's inability to handle size-1 DFTs.
        """
        from quop_mpi.algorithm.combinatorial import QWOA

        size = mpi_comm.Get_size()
        if size <= 1:
            pytest.skip("Need more than 1 process for this test")

        # System smaller than processes - tests the shrinking logic
        system_size = mpi_sizing.world_fraction(1, 2, minimum=1)

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        alg.execute()

        # Verify algorithm still works correctly
        # All ranks must call get_probabilities (collective)
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert len(probs) == system_size
            np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10)

        alg.destroy()

    def test_minimum_system_size(self, mpi_comm):
        """Test with minimum system size of 2."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = 2

        def qualities(local_i, local_i_offset):
            return np.array([0.0, 1.0])[local_i_offset : local_i_offset + local_i]

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        alg.execute()

        # All ranks must call get_probabilities (collective)
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert len(probs) == 2

        alg.destroy()

    def test_highly_uneven_partition_qwoa(self, mpi_comm, mpi_sizing):
        """Test with system size that leads to very uneven partition."""
        from quop_mpi.algorithm.combinatorial import QWOA

        # Force a remainder so the final partition is uneven across ranks.
        system_size = mpi_sizing.multiple(base=4, per_rank=3, per_node=8, remainder=1)

        def qualities(local_i, local_i_offset):
            return np.random.RandomState(42 + local_i_offset).random(local_i)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        alg.execute()

        # All ranks must call get_probabilities (collective)
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert len(probs) == system_size
            np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10)
            # State should have evolved from uniform - not all probabilities equal
            uniform_prob = 1.0 / system_size
            assert not np.allclose(probs, uniform_prob), "State did not evolve from uniform"

        alg.destroy()

    def test_large_prime_system_size(self, mpi_comm, mpi_sizing):
        """Test with larger prime system size."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = mpi_sizing.prime(base=127, min_per_rank=4, min_per_node=32)

        def qualities(local_i, local_i_offset):
            return np.random.RandomState(42 + local_i_offset).random(local_i)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        alg.evolve_state(np.array([0.1, 0.2]))

        # All ranks must call get_probabilities (it's a collective operation)
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10)
            # State should have evolved from uniform - not all probabilities equal
            uniform_prob = 1.0 / system_size
            assert not np.allclose(probs, uniform_prob), "State did not evolve from uniform"

        alg.destroy()


@pytest.mark.mpi
class TestEmptyPartitionHandling:
    """Test behavior when some ranks have empty partitions.

    Note: The circulant propagator (used by QWOA) requires system_size >= nprocs.
    This limitation comes from FFTW MPI which aborts when ranks receive 0 elements.
    Tests in this class verify behavior under valid configurations.
    """

    def test_all_ranks_participate_in_collective(self, mpi_comm, mpi_sizing):
        """Verify all ranks can participate in collective operations.

        Note: The circulant propagator requires system_size >= nprocs, so we
        use a system size that ensures all ranks have work.
        """
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = mpi_sizing.multiple(base=2, per_rank=2, per_node=16)

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        alg.execute()

        # All ranks must call get_probabilities (collective)
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10)
            # State should have evolved from uniform - not all probabilities equal
            uniform_prob = 1.0 / system_size
            assert not np.allclose(probs, uniform_prob), "State did not evolve from uniform"

        alg.destroy()


@pytest.mark.mpi
class TestPartitionConsistency:
    """Test partition table consistency across edge cases."""

    def test_partition_sums_to_system_size_prime(self, mpi_comm, mpi_sizing):
        """Verify partition table sums correctly for prime system size."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = mpi_sizing.prime(base=31, min_per_rank=1, min_per_node=8)

        def qualities(local_i, local_i_offset):
            return np.random.RandomState(42 + local_i_offset).random(local_i)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        alg.setup()

        # Gather local_i from all ranks
        all_local_i = mpi_comm.gather(alg.local_i if alg.subcomms.in_subcomm() else 0, root=0)

        if mpi_comm.Get_rank() == 0:
            total = sum(li for li in all_local_i if li is not None)
            assert total == system_size, f"Partition sum {total} != system size {system_size}"

        alg.destroy()

    def test_partition_offsets_are_correct(self, mpi_comm, mpi_sizing):
        """Verify partition offsets are correctly computed."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = mpi_sizing.prime(base=19, min_per_rank=1, min_per_node=8)

        def qualities(local_i, local_i_offset):
            return np.random.RandomState(42 + local_i_offset).random(local_i)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        alg.setup()

        rank = mpi_comm.Get_rank()

        # Gather local_i and offsets
        all_local_i = mpi_comm.gather(alg.local_i if alg.subcomms.in_subcomm() else None, root=0)
        all_offsets = mpi_comm.gather(
            alg.local_i_offset if alg.subcomms.in_subcomm() else None, root=0
        )

        if rank == 0:
            # Filter valid entries
            valid = [
                (li, off)
                for li, off in zip(all_local_i, all_offsets, strict=False)
                if li is not None and off is not None
            ]
            valid.sort(key=lambda x: x[1])

            # Verify offsets are consecutive
            expected = 0
            for li, off in valid:
                assert off == expected, f"Offset {off} != expected {expected}"
                expected += li

        alg.destroy()

    def test_state_correctness_with_prime_size(self, mpi_comm, mpi_sizing):
        """Verify state evolution is correct with prime system size."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = mpi_sizing.prime(base=13, min_per_rank=1, min_per_node=4)

        # Use simple qualities with known structure
        def qualities(local_i, local_i_offset):
            q = np.zeros(local_i)
            if local_i_offset == 0:
                q[0] = 1.0  # Mark state 0
            return q

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        alg.evolve_state(np.array([0.5, 0.3]))

        # All ranks must call get_final_state (it's a collective operation)
        state = alg.get_final_state()
        if mpi_comm.Get_rank() == 0:
            probs = np.abs(state) ** 2

            # State should be normalized
            np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10)

            # All probabilities should be non-negative
            assert np.all(probs >= 0)

        alg.destroy()


@pytest.mark.mpi
class TestSubcommManagement:
    """Test subcommunicator management edge cases."""

    def test_subcomm_free_called_properly(self, mpi_comm, simple_oracle):
        """Verify subcommunicator is freed without error."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()

        # Explicitly delete - should free subcomms properly
        alg.destroy()

    def test_multiple_algorithm_instances(self, mpi_comm, simple_oracle):
        """Verify multiple algorithm instances can be created and destroyed."""
        from quop_mpi.algorithm.combinatorial import QAOA

        for _ in range(3):
            alg = QAOA(simple_oracle.system_size, mpi_comm)
            alg.set_qualities(simple_oracle.qualities_function())
            alg.set_depth(1)
            alg.execute()
            alg.destroy()

    def test_sequential_algorithms_different_sizes(self, mpi_comm, simple_oracle):
        """Test creating algorithms with different sizes sequentially (power of 2 only for QAOA)."""
        from quop_mpi.algorithm.combinatorial import QAOA

        # QAOA only supports power-of-2 sizes due to hypercube mixer
        # Use simple_oracle which has power-of-2 size
        for _ in range(3):
            alg = QAOA(simple_oracle.system_size, mpi_comm)
            alg.set_qualities(simple_oracle.qualities_function())
            alg.set_depth(1)
            alg.execute()

            # All ranks must call get_probabilities (collective)
            probs = alg.get_probabilities()
            if mpi_comm.Get_rank() == 0:
                assert len(probs) == simple_oracle.system_size

            alg.destroy()

    def test_sequential_algorithms_mixed_sizes_qwoa(self, mpi_comm, mpi_sizing):
        """Test creating QWOA algorithms with mixed sizes sequentially."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_sizes = [
            mpi_sizing.power_of_two(base=8, min_per_rank=1),
            mpi_sizing.power_of_two(base=16, min_per_rank=2, min_per_node=16),
            mpi_sizing.prime(base=7, min_per_rank=1),
            mpi_sizing.prime(base=13, min_per_rank=2, min_per_node=8),
        ]

        # QWOA supports both power-of-two and prime-sized state spaces.
        for system_size in system_sizes:

            def qualities(local_i, local_i_offset):
                return np.random.RandomState(42 + local_i_offset).random(local_i)

            alg = QWOA(system_size, mpi_comm)
            alg.set_qualities(qualities)
            alg.set_depth(1)
            alg.execute()

            # All ranks must call get_probabilities (collective)
            probs = alg.get_probabilities()
            if mpi_comm.Get_rank() == 0:
                assert len(probs) == system_size
                np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-10)
                # State should have evolved from uniform - not all probabilities equal
                uniform_prob = 1.0 / system_size
                assert not np.allclose(probs, uniform_prob), "State did not evolve from uniform"

            alg.destroy()

    def test_in_subcomm_consistency(self, mpi_comm, simple_oracle):
        """Verify in_subcomm() returns consistent results."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()

        in_subcomm = alg.subcomms.in_subcomm()

        # Gather from all ranks
        all_in_subcomm = mpi_comm.gather(in_subcomm, root=0)

        if mpi_comm.Get_rank() == 0:
            # At least one rank should be in subcomm
            assert any(all_in_subcomm), "No ranks in subcomm"

        alg.destroy()
