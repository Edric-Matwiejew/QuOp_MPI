"""
Tests for MPI scaling behavior across different process counts.

These tests help identify and isolate MPI-related issues that may occur
when running with different numbers of MPI processes.

Run with: mpiexec -n <N> python -m pytest tests/mpi/test_mpi_scaling.py -v --with-mpi

Known issue: Deadlocks have been observed with N > 2 processes.
These tests are designed to help isolate the root cause.
"""
import pytest
import numpy as np
from mpi4py import MPI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import TestOracle


def get_mpi_info():
    """Return MPI rank and size."""
    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()


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

    def test_ansatz_creation(self, mpi_comm):
        """Verify Ansatz can be created."""
        from quop_mpi import Ansatz
        
        rank, size = mpi_comm.Get_rank(), mpi_comm.Get_size()
        system_size = 64
        
        # All ranks should be able to create an Ansatz
        alg = Ansatz(system_size, mpi_comm)
        
        assert alg.system_size == system_size
        mpi_comm.Barrier()  # Ensure all ranks complete

    def test_qaoa_creation(self, mpi_comm):
        """Verify QAOA can be created."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        system_size = 64
        alg = qaoa(system_size, mpi_comm)
        
        assert alg.system_size == system_size
        mpi_comm.Barrier()

    def test_qwoa_creation(self, mpi_comm):
        """Verify QWOA can be created."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        system_size = 64
        alg = qwoa(system_size, mpi_comm)
        
        assert alg.system_size == system_size
        mpi_comm.Barrier()


@pytest.mark.mpi
class TestQualitiesSetting:
    """Test setting qualities/observables with various process counts."""

    def test_set_qualities_qaoa(self, mpi_comm, simple_oracle):
        """Verify set_qualities completes for QAOA."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        
        mpi_comm.Barrier()

    def test_set_qualities_qwoa(self, mpi_comm, simple_oracle):
        """Verify set_qualities completes for QWOA."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        alg = qwoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        
        mpi_comm.Barrier()

    def test_set_depth(self, mpi_comm, simple_oracle):
        """Verify set_depth completes."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(2)
        
        mpi_comm.Barrier()


@pytest.mark.mpi
class TestSetupPhase:
    """Test the setup phase which initializes MPI subcommunicators."""

    def test_setup_completes_qaoa(self, mpi_comm, simple_oracle):
        """Verify QAOA setup completes without deadlock."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        # This is often where subcommunicator issues arise
        alg.setup()
        
        assert alg.setup_called == True
        mpi_comm.Barrier()

    def test_setup_completes_qwoa(self, mpi_comm, simple_oracle):
        """Verify QWOA setup completes without deadlock."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        alg = qwoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        alg.setup()
        
        assert alg.setup_called == True
        mpi_comm.Barrier()

    def test_subcomms_after_setup(self, mpi_comm, simple_oracle):
        """Examine subcommunicator state after setup."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        
        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()
        
        # Gather diagnostic info about subcomms
        in_subcomm = alg.subcomms.in_subcomm()
        subcomm_index = alg.subcomms.get_subcomm_index() if in_subcomm else -1
        
        # Collect info from all ranks
        all_in_subcomm = mpi_comm.gather(in_subcomm, root=0)
        all_subcomm_index = mpi_comm.gather(subcomm_index, root=0)
        
        if rank == 0:
            # All ranks should be in a subcomm for simple cases
            assert all(all_in_subcomm), \
                f"Not all ranks in subcomm: {all_in_subcomm}"
            # All should be in subcomm index 0
            assert all(idx == 0 for idx in all_subcomm_index), \
                f"Ranks in different subcomm indices: {all_subcomm_index}"


@pytest.mark.mpi
class TestEvolveStatePhase:
    """Test evolve_state which may involve MPI communication."""

    def test_evolve_state_completes(self, mpi_comm, simple_oracle):
        """Verify evolve_state completes without deadlock."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        params = simple_oracle.optimal_params(depth=1)
        
        # evolve_state calls setup internally
        alg.evolve_state(params)
        
        mpi_comm.Barrier()


@pytest.mark.mpi
class TestExecutePhase:
    """Test execute which runs the full optimization."""

    def test_execute_completes(self, mpi_comm, simple_oracle):
        """Verify execute completes without deadlock."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        # This is where deadlock is observed with > 2 processes
        alg.execute()
        
        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
        
        mpi_comm.Barrier()


@pytest.mark.mpi
class TestGatherPhase:
    """Test state gathering operations."""

    def test_get_probabilities_after_evolve(self, mpi_comm, simple_oracle):
        """Verify get_probabilities works after evolve_state."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)
        
        # This involves MPI gather
        probs = alg.get_probabilities()
        
        if mpi_comm.Get_rank() == 0:
            assert probs is not None
            assert len(probs) == simple_oracle.system_size
        
        mpi_comm.Barrier()

    def test_get_final_state_after_evolve(self, mpi_comm, simple_oracle):
        """Verify get_final_state works after evolve_state."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)
        
        # This involves MPI gather
        state = alg.get_final_state()
        
        if mpi_comm.Get_rank() == 0:
            assert state is not None
            assert len(state) == simple_oracle.system_size
        
        mpi_comm.Barrier()


@pytest.mark.mpi
class TestSystemSizeScaling:
    """Test with various system sizes relative to process count."""

    def test_system_size_equal_to_nprocs(self, mpi_comm):
        """Test when system_size equals number of processes."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        size = mpi_comm.Get_size()
        # System size must be power of 2
        system_size = 2 ** max(1, (size - 1).bit_length())
        
        oracle = TestOracle(system_size=system_size, n_marked=1, seed=42)
        
        alg = qaoa(system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()
        
        mpi_comm.Barrier()

    def test_system_size_less_than_nprocs(self, mpi_comm):
        """Test when system_size is less than number of processes."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        size = mpi_comm.Get_size()
        if size <= 2:
            pytest.skip("Need more than 2 processes for this test")
        
        # Use system size smaller than process count
        system_size = 2  # Minimum power of 2
        
        oracle = TestOracle(system_size=system_size, n_marked=1, seed=42)
        
        alg = qaoa(system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()
        
        mpi_comm.Barrier()

    def test_system_size_much_larger_than_nprocs(self, mpi_comm):
        """Test when system_size is much larger than number of processes."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        system_size = 256  # Should be divisible by any reasonable nprocs
        
        oracle = TestOracle(system_size=system_size, n_marked=4, seed=42)
        
        alg = qaoa(system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()
        
        mpi_comm.Barrier()


@pytest.mark.mpi
class TestPartitionTable:
    """Test partition table correctness across processes."""

    def test_partition_table_covers_all_elements(self, mpi_comm, simple_oracle):
        """Verify partition table correctly distributes all elements."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        
        # Get partition info from Ansatz (not unitaries)
        partition_table = alg.partition_table
        
        # Gather partition tables from all ranks
        all_tables = mpi_comm.gather(list(partition_table) if partition_table is not None else None, root=0)
        
        if mpi_comm.Get_rank() == 0:
            # Filter out None values (ranks not in subcomm)
            valid_tables = [t for t in all_tables if t is not None]
            
            if len(valid_tables) > 0:
                # All ranks in subcomm should have the same partition table
                for table in valid_tables:
                    assert table == valid_tables[0], \
                        "Partition tables differ across ranks"
                
                # Partition table should account for all elements
                # partition_table uses 1-based Fortran indexing with cumsum + 1
                # So partition_table[-1] = system_size + 1
                assert valid_tables[0][-1] == simple_oracle.system_size + 1, \
                    f"Partition table end {valid_tables[0][-1]} != system size + 1 ({simple_oracle.system_size + 1})"

    def test_local_i_and_offset_consistency(self, mpi_comm, simple_oracle):
        """Verify local_i and local_i_offset are consistent across ranks."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
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
            valid_pairs = [(li, off) for li, off in zip(all_local_i, all_offsets) 
                          if li is not None and off is not None]
            
            if len(valid_pairs) > 0:
                # Sort by offset to verify continuity
                valid_pairs.sort(key=lambda x: x[1])
                
                # Verify offsets are cumulative sums of local_i
                expected_offset = 0
                for li, offset in valid_pairs:
                    assert offset == expected_offset, \
                        f"Offset {offset} != expected {expected_offset}"
                    expected_offset += li
                
                # Total should equal system size
                assert expected_offset == simple_oracle.system_size, \
                    f"Total elements {expected_offset} != system size"
