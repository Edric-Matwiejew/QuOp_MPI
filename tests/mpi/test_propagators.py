"""
Tests for propagator unitaries.

These tests verify that each propagator type correctly:
1. Computes partition sizes via plan()
2. Distributes operator data via gen_operator()
3. Performs state evolution via propagate()
4. Produces consistent results across MPI ranks

Propagator types tested:
- diagonal: Phase-shift unitaries (e^{-i * gamma * H_diagonal})
- circulant: Complete graph mixing (used by QWOA)
- sparse: Sparse matrix exponential (used by QAOA hypercube mixer)
- composite: Multi-dimensional grid operators (used by qmoa)

Run with: mpiexec -n <N> python -m pytest tests/mpi/test_propagators.py -v --with-mpi
"""
import pytest
import numpy as np
from mpi4py import MPI


# =============================================================================
# Helper Functions
# =============================================================================

def create_partition_table(system_size, comm_size):
    """
    Create a 1-based partition table matching QuOp_MPI conventions.
    """
    base_size = system_size // comm_size
    remainder = system_size % comm_size
    
    partition_table = np.zeros(comm_size + 1, dtype=np.int64)
    partition_table[0] = 1
    
    for i in range(comm_size):
        local_size = base_size + (1 if i < remainder else 0)
        partition_table[i + 1] = partition_table[i] + local_size
    
    return partition_table


def gather_state(local_state, partition_table, mpi_comm):
    """Gather distributed state to rank 0."""
    from quop_mpi.__utils.__mpi import gather_array
    return gather_array(local_state, partition_table, mpi_comm)


# =============================================================================
# Tests for Diagonal Propagator
# =============================================================================

@pytest.mark.mpi
class TestDiagonalPropagator:
    """Tests for the diagonal (phase-shift) propagator."""

    def test_diagonal_plan_returns_valid_partition(self, mpi_comm):
        """Test that plan() returns valid local_i and alloc_local."""
        from quop_mpi.propagator.diagonal import unitary
        from quop_mpi.propagator.diagonal import operator
        
        system_size = 64
        
        u = unitary(operator.observables)
        local_i, alloc_local = u.plan(system_size, mpi_comm)
        
        # Gather all local_i values
        all_local_i = mpi_comm.gather(local_i, root=0)
        
        if mpi_comm.Get_rank() == 0:
            total = sum(all_local_i)
            assert total == system_size, f"Sum of local_i ({total}) != system_size ({system_size})"
        
        assert local_i >= 0
        assert alloc_local >= local_i

    def test_diagonal_plan_consistent_across_ranks(self, mpi_comm):
        """Test that all ranks compute consistent partitioning."""
        from quop_mpi.propagator.diagonal import unitary
        from quop_mpi.propagator.diagonal import operator
        
        for system_size in [16, 64, 100]:
            u = unitary(operator.observables)
            local_i, _ = u.plan(system_size, mpi_comm)
            
            all_local_i = mpi_comm.allgather(local_i)
            
            # Each rank should have the same view of the full partitioning
            total = sum(all_local_i)
            assert total == system_size

    def test_diagonal_propagator_via_qaoa(self, mpi_comm):
        """Test diagonal propagator through QAOA (indirect test)."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        system_size = 16
        
        def qualities(local_i, local_i_offset):
            # Simple linear qualities
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)
        
        alg = qaoa(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        
        # Evolve with zero parameters (no evolution)
        params = np.array([0.0, 0.0])
        alg.evolve_state(params)
        
        # State should be uniform superposition
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            expected_prob = 1.0 / system_size
            np.testing.assert_allclose(probs, expected_prob, rtol=1e-10)
        
        del alg

    def test_diagonal_phase_shift_correctness(self, mpi_comm):
        """Test that phase-shift applies correct phases to state."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        system_size = 8
        
        # Uniform qualities = all same phase shift, state unchanged (up to global phase)
        def uniform_qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)
        
        alg = qaoa(system_size, mpi_comm)
        alg.set_qualities(uniform_qualities)
        alg.set_depth(1)
        
        # Apply phase shift only (gamma=pi, t=0 to skip mixer)
        params = np.array([np.pi, 0.0])
        alg.evolve_state(params)
        
        # With uniform qualities, all states get same phase
        # Probabilities should remain uniform
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            expected_prob = 1.0 / system_size
            np.testing.assert_allclose(probs, expected_prob, rtol=1e-10)
        
        del alg


# =============================================================================
# Tests for Circulant Propagator
# =============================================================================

@pytest.mark.mpi
class TestCirculantPropagator:
    """Tests for the circulant (complete graph) propagator."""

    def test_circulant_plan_returns_valid_partition(self, mpi_comm):
        """Test that plan() returns valid local_i and alloc_local."""
        from quop_mpi.propagator.circulant import unitary
        from quop_mpi.propagator.circulant import operator
        
        system_size = 64
        
        u = unitary(operator.complete)
        local_i, alloc_local = u.plan(system_size, mpi_comm)
        
        all_local_i = mpi_comm.gather(local_i, root=0)
        
        if mpi_comm.Get_rank() == 0:
            total = sum(all_local_i)
            assert total == system_size
        
        assert local_i >= 0
        assert alloc_local >= local_i

    def test_circulant_propagator_via_qwoa(self, mpi_comm):
        """Test circulant propagator through QWOA (indirect test)."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        system_size = 16
        
        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)
        
        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        
        # Zero parameters = no evolution
        params = np.array([0.0, 0.0])
        alg.evolve_state(params)
        
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            expected_prob = 1.0 / system_size
            np.testing.assert_allclose(probs, expected_prob, rtol=1e-10)
        
        del alg

    def test_circulant_complete_graph_mixing(self, mpi_comm):
        """Test complete graph mixing preserves total probability."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        system_size = 16
        
        def qualities(local_i, local_i_offset):
            return np.zeros(local_i, dtype=np.float64)
        
        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        
        # Apply mixing only (gamma=0, t=pi/N)
        t = np.pi / system_size
        params = np.array([0.0, t])
        alg.evolve_state(params)
        
        # Probability should still sum to 1
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            total_prob = np.sum(probs)
            assert abs(total_prob - 1.0) < 1e-10
        
        del alg


# =============================================================================
# Tests for Sparse Propagator
# =============================================================================

@pytest.mark.mpi
class TestSparsePropagator:
    """Tests for the sparse (hypercube) propagator."""

    def test_sparse_plan_returns_valid_partition(self, mpi_comm):
        """Test that plan() returns valid local_i and alloc_local."""
        from quop_mpi.propagator.sparse import unitary
        from quop_mpi.propagator.sparse import operator
        
        system_size = 64  # 2^6
        
        u = unitary(operator.hypercube)
        local_i, alloc_local = u.plan(system_size, mpi_comm)
        
        all_local_i = mpi_comm.gather(local_i, root=0)
        
        if mpi_comm.Get_rank() == 0:
            total = sum(all_local_i)
            assert total == system_size
        
        assert local_i >= 0
        assert alloc_local >= local_i

    def test_sparse_requires_even_system_size(self, mpi_comm):
        """Test that hypercube operator requires even system size."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        # Odd system_size should raise RuntimeError
        system_size = 11
        
        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)
        
        alg = qaoa(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        
        # Should raise RuntimeError about system_size % 2 != 0
        with pytest.raises(RuntimeError):
            alg.evolve_state(np.array([0.1, 0.1]))
        
        del alg

    def test_sparse_hypercube_mixing(self, mpi_comm):
        """Test hypercube mixing preserves total probability."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        system_size = 16  # 2^4
        
        def qualities(local_i, local_i_offset):
            return np.zeros(local_i, dtype=np.float64)
        
        alg = qaoa(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        
        # Apply mixing (gamma=0, t=pi/4)
        params = np.array([0.0, np.pi/4])
        alg.evolve_state(params)
        
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            total_prob = np.sum(probs)
            assert abs(total_prob - 1.0) < 1e-10
        
        del alg


# =============================================================================
# Tests for Composite Propagator
# =============================================================================

@pytest.mark.mpi
class TestCompositePropagator:
    """Tests for the composite (multi-dimensional) propagator."""

    def test_composite_plan_returns_valid_partition(self, mpi_comm):
        """Test that plan() returns valid local_i and alloc_local."""
        from quop_mpi.propagator.composite import unitary
        from quop_mpi.propagator.composite import operator
        
        Ns = [4, 4]  # 2D grid, 4x4 = 16 total
        system_size = np.prod([2**n for n in Ns])
        
        u = unitary(Ns, operator.ith)
        local_i, alloc_local = u.plan(system_size, mpi_comm)
        
        all_local_i = mpi_comm.gather(local_i, root=0)
        
        if mpi_comm.Get_rank() == 0:
            total = sum(all_local_i)
            assert total == system_size
        
        assert local_i >= 0
        assert alloc_local >= local_i

    def test_composite_is_planner(self, mpi_comm):
        """Test that composite unitary has planner=True."""
        from quop_mpi.propagator.composite import unitary
        from quop_mpi.propagator.composite import operator
        
        Ns = [3, 3]
        u = unitary(Ns, operator.ith)
        
        assert u.planner == True


# =============================================================================
# Cross-propagator tests
# =============================================================================

@pytest.mark.mpi
class TestPropagatorConsistency:
    """Tests ensuring propagators behave consistently."""

    def test_all_propagators_partition_consistently(self, mpi_comm):
        """Test that different propagators agree on partitioning for same system."""
        from quop_mpi.propagator.diagonal import unitary as diagonal_unitary
        from quop_mpi.propagator.diagonal import operator as diagonal_operator
        from quop_mpi.propagator.circulant import unitary as circulant_unitary
        from quop_mpi.propagator.circulant import operator as circulant_operator
        from quop_mpi.propagator.sparse import unitary as sparse_unitary
        from quop_mpi.propagator.sparse import operator as sparse_operator
        
        system_size = 64
        
        u_diag = diagonal_unitary(diagonal_operator.observables)
        u_circ = circulant_unitary(circulant_operator.complete)
        u_sparse = sparse_unitary(sparse_operator.hypercube)
        
        local_i_diag, _ = u_diag.plan(system_size, mpi_comm)
        local_i_circ, _ = u_circ.plan(system_size, mpi_comm)
        local_i_sparse, _ = u_sparse.plan(system_size, mpi_comm)
        
        # All should return same local_i for same system_size
        assert local_i_diag == local_i_circ == local_i_sparse

    def test_qaoa_qwoa_same_system_size_partitioning(self, mpi_comm):
        """Test QAOA and QWOA produce consistent partitioning."""
        from quop_mpi.algorithm.combinatorial import qaoa, qwoa
        
        system_size = 16
        
        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)
        
        alg_qaoa = qaoa(system_size, mpi_comm)
        alg_qaoa.set_qualities(qualities)
        alg_qaoa.set_depth(1)
        alg_qaoa.setup()
        
        alg_qwoa = qwoa(system_size, mpi_comm)
        alg_qwoa.set_qualities(qualities)
        alg_qwoa.set_depth(1)
        alg_qwoa.setup()
        
        # Both should have same local_i
        assert alg_qaoa.local_i == alg_qwoa.local_i
        assert alg_qaoa.local_i_offset == alg_qwoa.local_i_offset
        
        del alg_qaoa
        del alg_qwoa


# =============================================================================
# State Evolution Correctness Tests
# =============================================================================

@pytest.mark.mpi  
class TestStateEvolutionCorrectness:
    """Tests verifying state evolution produces correct results."""

    def test_identity_evolution(self, mpi_comm):
        """Test that zero parameters give identity evolution."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        system_size = 16
        
        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)
        
        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        
        # Zero parameters = no evolution
        params = np.zeros(2)
        alg.evolve_state(params)
        
        state = alg.get_final_state()
        if mpi_comm.Get_rank() == 0:
            # Should be uniform superposition
            expected = np.ones(system_size, dtype=np.complex128) / np.sqrt(system_size)
            np.testing.assert_allclose(state, expected, rtol=1e-10)
        
        del alg

    def test_evolution_is_unitary(self, mpi_comm):
        """Test that evolution preserves state norm."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        system_size = 16
        
        def qualities(local_i, local_i_offset):
            return np.random.default_rng(42).random(local_i)
        
        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(2)
        
        # Random parameters
        params = np.random.default_rng(123).random(4) * 2 * np.pi
        alg.evolve_state(params)
        
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            total = np.sum(probs)
            assert abs(total - 1.0) < 1e-10, f"State not normalized: {total}"
        
        del alg

    def test_deterministic_evolution(self, mpi_comm):
        """Test that same parameters give same result."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        system_size = 16
        
        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)
        
        params = np.array([0.5, 0.3])
        
        # First evolution
        alg1 = qwoa(system_size, mpi_comm)
        alg1.set_qualities(qualities)
        alg1.set_depth(1)
        alg1.evolve_state(params)
        state1 = alg1.get_final_state()
        del alg1
        
        # Second evolution with same params
        alg2 = qwoa(system_size, mpi_comm)
        alg2.set_qualities(qualities)
        alg2.set_depth(1)
        alg2.evolve_state(params)
        state2 = alg2.get_final_state()
        del alg2
        
        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(state1, state2, rtol=1e-14)

    def test_different_params_different_results(self, mpi_comm):
        """Test that different parameters give different states."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        system_size = 16
        
        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)
        
        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        
        params1 = np.array([0.1, 0.1])
        alg.evolve_state(params1)
        state1 = alg.get_final_state()
        
        params2 = np.array([0.5, 0.5])
        alg.evolve_state(params2)
        state2 = alg.get_final_state()
        
        if mpi_comm.Get_rank() == 0:
            # States should be different
            diff = np.linalg.norm(state1 - state2)
            assert diff > 1e-6, "Different parameters should produce different states"
        
        del alg
