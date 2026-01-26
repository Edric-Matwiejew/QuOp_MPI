"""
Tests for core MPI scatter/gather operations.

These tests verify the foundational MPI data distribution functions that
all higher-level QuOp_MPI components depend on:
- scatter_1D_array: Distributes 1D arrays across MPI ranks
- scatter_sparse: Distributes CSR sparse matrices across MPI ranks
- gather_array: Gathers distributed arrays back to rank 0

Run with: mpiexec -n <N> python -m pytest tests/mpi/test_scatter_operations.py -v --with-mpi
"""
import pytest
import numpy as np
from mpi4py import MPI
from scipy import sparse as sp

# Import the internal functions at module level to avoid Python name mangling
# issues when importing dunder-prefixed names inside class methods
import quop_mpi.__utils.__mpi as mpi_utils

# Create aliases for the dunder-prefixed functions
scatter_1D_array = mpi_utils.__scatter_1D_array
scatter_sparse = mpi_utils.__scatter_sparse
gather_array = mpi_utils.gather_array


# =============================================================================
# Helper Functions
# =============================================================================

def create_partition_table(system_size, comm_size):
    """
    Create a 1-based partition table matching QuOp_MPI conventions.
    
    The partition table has length (comm_size + 1) where:
    - partition_table[0] = 1 (1-based indexing)
    - partition_table[rank+1] - partition_table[rank] = local_i for that rank
    - partition_table[-1] = system_size + 1
    """
    base_size = system_size // comm_size
    remainder = system_size % comm_size
    
    partition_table = np.zeros(comm_size + 1, dtype=np.int64)
    partition_table[0] = 1  # 1-based indexing
    
    for i in range(comm_size):
        # Distribute remainder across first 'remainder' ranks
        local_size = base_size + (1 if i < remainder else 0)
        partition_table[i + 1] = partition_table[i] + local_size
    
    return partition_table


def create_test_csr_matrix(n_rows, n_cols, density=0.3, seed=42):
    """Create a reproducible sparse CSR matrix for testing."""
    rng = np.random.default_rng(seed)
    matrix = sp.random(n_rows, n_cols, density=density, format='csr', 
                       dtype=np.complex128, random_state=rng)
    return matrix


# =============================================================================
# Tests for scatter_1D_array
# =============================================================================

@pytest.mark.mpi
class TestScatter1DArray:
    """Tests for the scatter_1D_array function."""

    def test_scatter_float64_even_distribution(self, mpi_comm):
        """Test scattering float64 array with even distribution across ranks."""
        # using module-level scatter_1D_array
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        # Create array that divides evenly
        system_size = size * 4
        partition_table = create_partition_table(system_size, size)
        
        # Create test data on rank 0
        if rank == 0:
            full_array = np.arange(system_size, dtype=np.float64)
        else:
            full_array = None
        
        # Scatter
        local_array = scatter_1D_array(full_array, partition_table, mpi_comm, np.float64)
        
        # Verify local partition
        local_i = partition_table[rank + 1] - partition_table[rank]
        local_i_offset = partition_table[rank] - 1  # Convert to 0-based
        
        assert len(local_array) == local_i, f"Rank {rank}: expected {local_i} elements, got {len(local_array)}"
        
        expected = np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)
        np.testing.assert_array_equal(local_array, expected,
            err_msg=f"Rank {rank}: local array values don't match expected")

    def test_scatter_float64_uneven_distribution(self, mpi_comm):
        """Test scattering float64 array with uneven distribution (remainder)."""
        # using module-level scatter_1D_array
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        # Create array that doesn't divide evenly
        system_size = size * 4 + size // 2 + 1
        partition_table = create_partition_table(system_size, size)
        
        if rank == 0:
            full_array = np.arange(system_size, dtype=np.float64) * 2.5
        else:
            full_array = None
        
        local_array = scatter_1D_array(full_array, partition_table, mpi_comm, np.float64)
        
        local_i = partition_table[rank + 1] - partition_table[rank]
        local_i_offset = partition_table[rank] - 1
        
        assert len(local_array) == local_i
        
        expected = np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64) * 2.5
        np.testing.assert_allclose(local_array, expected, rtol=1e-14)

    def test_scatter_complex128(self, mpi_comm):
        """Test scattering complex128 array."""
        # using module-level scatter_1D_array
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = size * 8
        partition_table = create_partition_table(system_size, size)
        
        if rank == 0:
            real_part = np.arange(system_size, dtype=np.float64)
            imag_part = np.arange(system_size, dtype=np.float64) * 0.5
            full_array = real_part + 1j * imag_part
        else:
            full_array = None
        
        local_array = scatter_1D_array(full_array, partition_table, mpi_comm, np.complex128)
        
        local_i = partition_table[rank + 1] - partition_table[rank]
        local_i_offset = partition_table[rank] - 1
        
        assert len(local_array) == local_i
        assert local_array.dtype == np.complex128
        
        expected_real = np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)
        expected_imag = expected_real * 0.5
        expected = expected_real + 1j * expected_imag
        
        np.testing.assert_allclose(local_array, expected, rtol=1e-14)

    def test_scatter_small_system(self, mpi_comm):
        """Test scattering when system_size is smaller than or equal to rank count."""
        # using module-level scatter_1D_array
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        # System size equal to number of ranks (each rank gets 1 element)
        system_size = size
        partition_table = create_partition_table(system_size, size)
        
        if rank == 0:
            full_array = np.arange(system_size, dtype=np.float64) + 100.0
        else:
            full_array = None
        
        local_array = scatter_1D_array(full_array, partition_table, mpi_comm, np.float64)
        
        local_i = partition_table[rank + 1] - partition_table[rank]
        assert len(local_array) == local_i
        
        if local_i > 0:
            local_i_offset = partition_table[rank] - 1
            expected = np.array([local_i_offset + 100.0], dtype=np.float64)
            np.testing.assert_array_equal(local_array, expected)

    def test_scatter_preserves_precision(self, mpi_comm):
        """Test that scatter preserves floating point precision."""
        # using module-level scatter_1D_array
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = size * 10
        partition_table = create_partition_table(system_size, size)
        
        if rank == 0:
            # Use values that require full precision
            full_array = np.array([np.pi, np.e, np.sqrt(2), 1e-15, 1e15] * (system_size // 5),
                                  dtype=np.float64)[:system_size]
        else:
            full_array = None
        
        local_array = scatter_1D_array(full_array, partition_table, mpi_comm, np.float64)
        
        # Gather back and compare
        gathered = mpi_comm.gather(local_array.tolist(), root=0)
        
        if rank == 0:
            reconstructed = np.concatenate([np.array(g) for g in gathered])
            np.testing.assert_array_equal(reconstructed, full_array)


# =============================================================================
# Tests for gather_array
# =============================================================================

@pytest.mark.mpi
class TestGatherArray:
    """Tests for the gather_array function."""

    def test_gather_float64(self, mpi_comm):
        """Test gathering float64 arrays from all ranks."""
        # using module-level gather_array
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = size * 5
        partition_table = create_partition_table(system_size, size)
        
        local_i = partition_table[rank + 1] - partition_table[rank]
        local_i_offset = partition_table[rank] - 1
        
        # Each rank creates its local portion
        local_array = np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)
        
        gathered = gather_array(local_array, partition_table, mpi_comm)
        
        if rank == 0:
            assert gathered is not None
            assert len(gathered) == system_size
            expected = np.arange(system_size, dtype=np.float64)
            np.testing.assert_array_equal(gathered, expected)
        else:
            assert gathered is None

    def test_gather_complex128(self, mpi_comm):
        """Test gathering complex128 arrays from all ranks."""
        # using module-level gather_array
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = size * 4
        partition_table = create_partition_table(system_size, size)
        
        local_i = partition_table[rank + 1] - partition_table[rank]
        local_i_offset = partition_table[rank] - 1
        
        # Create complex local array
        real_part = np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)
        imag_part = real_part * 2.0
        local_array = (real_part + 1j * imag_part).astype(np.complex128)
        
        gathered = gather_array(local_array, partition_table, mpi_comm)
        
        if rank == 0:
            assert gathered is not None
            expected_real = np.arange(system_size, dtype=np.float64)
            expected = expected_real + 1j * (expected_real * 2.0)
            np.testing.assert_allclose(gathered, expected, rtol=1e-14)

    def test_scatter_gather_roundtrip(self, mpi_comm):
        """Test that scatter followed by gather reconstructs the original array."""
        # using module-level imports
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = size * 7 + 3  # Uneven distribution
        partition_table = create_partition_table(system_size, size)
        
        if rank == 0:
            original = np.random.default_rng(42).random(system_size).astype(np.float64)
        else:
            original = None
        
        # Broadcast original for later comparison
        original = mpi_comm.bcast(original, root=0)
        
        # Scatter
        if rank == 0:
            local_array = scatter_1D_array(original, partition_table, mpi_comm, np.float64)
        else:
            local_array = scatter_1D_array(None, partition_table, mpi_comm, np.float64)
        
        # Gather
        gathered = gather_array(local_array, partition_table, mpi_comm)
        
        if rank == 0:
            np.testing.assert_array_equal(gathered, original)


# =============================================================================
# Tests for scatter_sparse
# =============================================================================

@pytest.mark.mpi
class TestScatterSparse:
    """Tests for the scatter_sparse function."""

    def test_scatter_single_sparse_matrix(self, mpi_comm):
        """Test scattering a single sparse CSR matrix."""
        # using module-level scatter_sparse
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = size * 4
        partition_table = create_partition_table(system_size, size)
        
        if rank == 0:
            matrix = create_test_csr_matrix(system_size, system_size, density=0.3, seed=42)
            # Convert to 1-based indexing for QuOp_MPI convention
            row_starts = [matrix.indptr + 1]
            col_indexes = [matrix.indices + 1]
            values = [matrix.data]
        else:
            row_starts = None
            col_indexes = None
            values = None
        
        W_row_starts, W_col_indexes, W_values = scatter_sparse(
            row_starts, col_indexes, values, partition_table, mpi_comm
        )
        
        # Verify we got one term back
        assert len(W_row_starts) == 1
        assert len(W_col_indexes) == 1
        assert len(W_values) == 1
        
        local_i = partition_table[rank + 1] - partition_table[rank]
        
        # Row starts should have local_i + 1 entries
        assert len(W_row_starts[0]) == local_i + 1
        
        # Number of non-zeros should match
        n_local_nnz = W_row_starts[0][-1] - W_row_starts[0][0]
        assert len(W_col_indexes[0]) == n_local_nnz
        assert len(W_values[0]) == n_local_nnz

    def test_scatter_sparse_values_correct(self, mpi_comm):
        """Test that scattered sparse matrix values are correct."""
        # using module-level scatter_sparse
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = size * 3
        partition_table = create_partition_table(system_size, size)
        
        # Create a simple diagonal matrix for easy verification
        if rank == 0:
            matrix = sp.diags(np.arange(1, system_size + 1, dtype=np.complex128), 
                             format='csr')
            row_starts = [matrix.indptr + 1]
            col_indexes = [matrix.indices + 1]
            values = [matrix.data]
            original_data = matrix.data.copy()
        else:
            row_starts = None
            col_indexes = None
            values = None
            original_data = None
        
        original_data = mpi_comm.bcast(original_data, root=0)
        
        W_row_starts, W_col_indexes, W_values = scatter_sparse(
            row_starts, col_indexes, values, partition_table, mpi_comm
        )
        
        local_i_offset = partition_table[rank] - 1
        local_i = partition_table[rank + 1] - partition_table[rank]
        
        # For a diagonal matrix, each local row should have one entry
        # with value equal to (row_index + 1)
        expected_values = np.arange(local_i_offset + 1, local_i_offset + local_i + 1, 
                                    dtype=np.complex128)
        
        np.testing.assert_allclose(W_values[0], expected_values, rtol=1e-14)

    def test_scatter_multiple_sparse_matrices(self, mpi_comm):
        """Test scattering multiple sparse matrices (list of CSR)."""
        # using module-level scatter_sparse
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = size * 4
        partition_table = create_partition_table(system_size, size)
        n_matrices = 3
        
        if rank == 0:
            row_starts = []
            col_indexes = []
            values = []
            for i in range(n_matrices):
                matrix = create_test_csr_matrix(system_size, system_size, 
                                               density=0.2, seed=42 + i)
                row_starts.append(matrix.indptr + 1)
                col_indexes.append(matrix.indices + 1)
                values.append(matrix.data)
        else:
            row_starts = None
            col_indexes = None
            values = None
        
        W_row_starts, W_col_indexes, W_values = scatter_sparse(
            row_starts, col_indexes, values, partition_table, mpi_comm
        )
        
        # Verify we got all matrices
        assert len(W_row_starts) == n_matrices
        assert len(W_col_indexes) == n_matrices
        assert len(W_values) == n_matrices
        
        local_i = partition_table[rank + 1] - partition_table[rank]
        
        for i in range(n_matrices):
            assert len(W_row_starts[i]) == local_i + 1
            n_local_nnz = W_row_starts[i][-1] - W_row_starts[i][0]
            assert len(W_col_indexes[i]) == n_local_nnz
            assert len(W_values[i]) == n_local_nnz

    def test_scatter_sparse_empty_rows(self, mpi_comm):
        """Test scattering sparse matrix where some rows may be empty."""
        # using module-level scatter_sparse
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = size * 4
        partition_table = create_partition_table(system_size, size)
        
        if rank == 0:
            # Create a very sparse matrix (some rows will be empty)
            matrix = create_test_csr_matrix(system_size, system_size, 
                                           density=0.1, seed=123)
            row_starts = [matrix.indptr + 1]
            col_indexes = [matrix.indices + 1]
            values = [matrix.data]
        else:
            row_starts = None
            col_indexes = None
            values = None
        
        # Should not raise even with empty rows
        W_row_starts, W_col_indexes, W_values = scatter_sparse(
            row_starts, col_indexes, values, partition_table, mpi_comm
        )
        
        local_i = partition_table[rank + 1] - partition_table[rank]
        assert len(W_row_starts[0]) == local_i + 1


# =============================================================================
# Tests for partition_table consistency
# =============================================================================

@pytest.mark.mpi
class TestPartitionTableConsistency:
    """Tests verifying partition table handling is consistent across operations."""

    def test_partition_covers_all_elements(self, mpi_comm):
        """Verify partition table covers all elements exactly once."""
        size = mpi_comm.Get_size()
        
        for system_size in [size, size * 2, size * 3 + 1, size * 7 - 2]:
            partition_table = create_partition_table(system_size, size)
            
            # First element should be 1 (1-based indexing)
            assert partition_table[0] == 1
            
            # Last element should be system_size + 1
            assert partition_table[-1] == system_size + 1
            
            # Total elements across all partitions
            total = sum(partition_table[i + 1] - partition_table[i] for i in range(size))
            assert total == system_size

    def test_all_ranks_agree_on_partition(self, mpi_comm):
        """Verify all ranks compute the same partition table."""
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = size * 5 + 3
        local_partition = create_partition_table(system_size, size)
        
        # Gather all partition tables
        all_partitions = mpi_comm.gather(local_partition.tolist(), root=0)
        
        if rank == 0:
            for i, partition in enumerate(all_partitions):
                np.testing.assert_array_equal(partition, local_partition,
                    err_msg=f"Rank {i} has different partition table")


# =============================================================================
# Edge case tests
# =============================================================================

@pytest.mark.mpi
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_element_per_rank(self, mpi_comm):
        """Test when each rank gets exactly one element."""
        # using module-level imports
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = size
        partition_table = create_partition_table(system_size, size)
        
        if rank == 0:
            full_array = np.arange(system_size, dtype=np.float64) * 10.0
        else:
            full_array = None
        
        local_array = scatter_1D_array(full_array, partition_table, mpi_comm, np.float64)
        
        assert len(local_array) == 1
        assert local_array[0] == rank * 10.0

    def test_large_array(self, mpi_comm):
        """Test scattering a larger array."""
        # using module-level imports
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        system_size = 1000
        partition_table = create_partition_table(system_size, size)
        
        if rank == 0:
            full_array = np.random.default_rng(42).random(system_size).astype(np.float64)
        else:
            full_array = None
        
        full_array = mpi_comm.bcast(full_array, root=0)
        
        if rank == 0:
            local_array = scatter_1D_array(full_array, partition_table, mpi_comm, np.float64)
        else:
            local_array = scatter_1D_array(None, partition_table, mpi_comm, np.float64)
        
        gathered = gather_array(local_array, partition_table, mpi_comm)
        
        if rank == 0:
            np.testing.assert_allclose(gathered, full_array, rtol=1e-14)

    def test_prime_system_size(self, mpi_comm):
        """Test with prime system sizes that don't divide evenly."""
        # using module-level imports
        
        size = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        
        # Test several prime sizes
        for system_size in [7, 11, 13, 17, 23]:
            if system_size < size:
                continue  # Skip if smaller than rank count
                
            partition_table = create_partition_table(system_size, size)
            
            if rank == 0:
                full_array = np.arange(system_size, dtype=np.float64)
            else:
                full_array = None
            
            full_array = mpi_comm.bcast(full_array, root=0)
            
            if rank == 0:
                local_array = scatter_1D_array(full_array, partition_table, mpi_comm, np.float64)
            else:
                local_array = scatter_1D_array(None, partition_table, mpi_comm, np.float64)
            
            gathered = gather_array(local_array, partition_table, mpi_comm)
            
            if rank == 0:
                np.testing.assert_array_equal(gathered, full_array,
                    err_msg=f"Roundtrip failed for prime size {system_size}")
