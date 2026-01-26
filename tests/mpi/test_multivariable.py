"""
Tests for multivariable optimization algorithms (QMOA and QOWE).

These tests verify that qmoa and qowe correctly:
1. Initialize with multi-dimensional grids
2. Distribute state and operators across MPI ranks
3. Perform state evolution with composite/momentum mixers
4. Produce consistent results across different rank counts

Algorithms tested:
- qmoa: Quantum Multivariable Optimization Algorithm (composite mixer)
- qowe: Quantum Optimization with Wavepacket Evolution (momentum mixer)

Run with: mpiexec -n <N> python -m pytest tests/mpi/test_multivariable.py -v --with-mpi
"""
import pytest
import numpy as np
from mpi4py import MPI


# =============================================================================
# Test Functions (Objective Functions)
# =============================================================================

def sphere(x):
    """Sphere function: f(x) = sum(x_i^2). Minimum at origin."""
    return np.sum(x**2, axis=1)


def rastrigin(x):
    """Rastrigin function: highly multimodal."""
    n = x.shape[1] if len(x.shape) > 1 else len(x)
    return n * 10 + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def simple_linear(x):
    """Simple linear function for testing: f(x) = sum(x_i)."""
    return np.sum(x, axis=1)


# =============================================================================
# Tests for QMOA (Quantum Multivariable Optimization Algorithm)
# =============================================================================

@pytest.mark.mpi
class TestQMOA:
    """Tests for the QMOA algorithm with composite mixer."""

    def test_qmoa_initialization(self, mpi_comm):
        """Test that QMOA initializes correctly with multi-dimensional grid."""
        from quop_mpi.algorithm.multivariable import qmoa, setup_cartesian, cartesian
        
        # 2D grid: 2^2 x 2^2 = 16 points
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        
        deltas, mins = setup_cartesian(Ns, bounds)
        alg = qmoa(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(1)
        
        # System size should be product of 2^N for each dimension
        expected_size = (2**2) * (2**2)  # 16
        assert alg.system_size == expected_size
        
        del alg

    def test_qmoa_partitioning_consistency(self, mpi_comm):
        """Test that QMOA partitions state consistently across ranks."""
        from quop_mpi.algorithm.multivariable import qmoa, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        
        deltas, mins = setup_cartesian(Ns, bounds)
        alg = qmoa(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(1)
        alg.setup()
        
        # Gather all local_i values
        all_local_i = mpi_comm.allgather(alg.local_i)
        total = sum(all_local_i)
        
        assert total == alg.system_size, f"Partition sum {total} != system_size {alg.system_size}"
        
        del alg

    def test_qmoa_identity_evolution(self, mpi_comm):
        """Test that zero parameters give identity evolution (uniform superposition)."""
        from quop_mpi.algorithm.multivariable import qmoa, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        
        deltas, mins = setup_cartesian(Ns, bounds)
        alg = qmoa(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(1)
        alg.setup()
        
        # Zero parameters = no evolution
        # total_params is available after set_unitaries (called in __init__)
        # For depth=1: n_variational_parameters = total_params * depth
        n_params = alg.total_params * alg.ansatz_depth
        params = np.zeros(n_params)
        alg.evolve_state(params)
        
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            expected_prob = 1.0 / alg.system_size
            np.testing.assert_allclose(probs, expected_prob, rtol=1e-10)
        
        del alg

    def test_qmoa_preserves_normalization(self, mpi_comm):
        """Test that QMOA evolution preserves state normalization."""
        from quop_mpi.algorithm.multivariable import qmoa, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]
        
        deltas, mins = setup_cartesian(Ns, bounds)
        alg = qmoa(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(2)
        
        # Random parameters - use total_params * ansatz_depth
        rng = np.random.default_rng(42)
        n_params = alg.total_params * alg.ansatz_depth
        params = rng.random(n_params) * 2 * np.pi
        alg.evolve_state(params)
        
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            total_prob = np.sum(probs)
            assert abs(total_prob - 1.0) < 1e-10
        
        del alg

    def test_qmoa_deterministic_evolution(self, mpi_comm):
        """Test that same parameters produce same results."""
        from quop_mpi.algorithm.multivariable import qmoa, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)
        
        # QMOA with 2D grid, depth=1: 1 gamma + 2 t's = 3 params
        params = np.array([0.3, 0.5, 0.2])
        
        # First run
        alg1 = qmoa(Ns, mpi_comm)
        alg1.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg1.set_depth(1)
        alg1.evolve_state(params)
        state1 = alg1.get_final_state()
        del alg1
        
        # Second run with same params
        alg2 = qmoa(Ns, mpi_comm)
        alg2.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg2.set_depth(1)
        alg2.evolve_state(params)
        state2 = alg2.get_final_state()
        del alg2
        
        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(state1, state2, rtol=1e-12)

    def test_qmoa_3d_grid(self, mpi_comm):
        """Test QMOA with 3-dimensional grid."""
        from quop_mpi.algorithm.multivariable import qmoa, setup_cartesian, cartesian
        
        # 3D grid: 2^2 x 2^2 x 2^2 = 64 points
        Ns = [2, 2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
        
        deltas, mins = setup_cartesian(Ns, bounds)
        alg = qmoa(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(1)
        
        expected_size = (2**2) ** 3  # 64
        assert alg.system_size == expected_size
        
        # Should evolve without error
        # 3D QMOA: 1 gamma + 3 t's = 4 params
        n_params = alg.total_params * alg.ansatz_depth
        params = np.zeros(n_params)
        alg.evolve_state(params)
        
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert abs(np.sum(probs) - 1.0) < 1e-10
        
        del alg

    def test_qmoa_set_mixer(self, mpi_comm):
        """Test that set_mixer changes the circulant graph structure."""
        from quop_mpi.algorithm.multivariable import qmoa, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)
        
        alg = qmoa(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(1)
        
        # Set custom mixer (cycle graphs instead of complete)
        Cs = [1, 1]
        alg.set_mixer(Cs)
        
        # 2D QMOA: 1 gamma + 2 t's = 3 params
        params = np.array([0.5, 0.3, 0.2])
        alg.evolve_state(params)
        
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert abs(np.sum(probs) - 1.0) < 1e-10
        
        del alg

    def test_qmoa_independent_t(self, mpi_comm):
        """Test setting independent vs shared walk times."""
        from quop_mpi.algorithm.multivariable import qmoa, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)
        
        # Independent walk times (default): 1 gamma + 2 t's = 3 params
        alg_ind = qmoa(Ns, mpi_comm)
        alg_ind.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg_ind.set_depth(1)
        n_params_ind = alg_ind.total_params
        
        # Shared walk time: 1 gamma + 1 t = 2 params
        # Note: set_independent_t must be called BEFORE set_depth for param count update
        alg_shared = qmoa(Ns, mpi_comm)
        alg_shared.set_independent_t(False)
        alg_shared.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg_shared.set_depth(1)
        n_params_shared = alg_shared.total_params
        
        # Independent should have more parameters (1 gamma + n t's vs 1 gamma + 1 t)
        assert n_params_ind == 3, f"Expected 3 params for independent, got {n_params_ind}"
        assert n_params_shared == 2, f"Expected 2 params for shared, got {n_params_shared}"
        
        del alg_ind
        del alg_shared


# =============================================================================
# Tests for QOWE (Quantum Optimization with Wavepacket Evolution)
# =============================================================================

@pytest.mark.mpi
class TestQOWE:
    """Tests for the QOWE algorithm with momentum mixer."""

    def test_qowe_initialization(self, mpi_comm):
        """Test that QOWE initializes correctly."""
        from quop_mpi.algorithm.multivariable import qowe, setup_cartesian
        
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        
        deltas, mins = setup_cartesian(Ns, bounds)
        alg = qowe(Ns, deltas, mins, mpi_comm)
        
        expected_size = (2**2) * (2**2)
        assert alg.system_size == expected_size
        
        # Check momentum-space parameters are computed
        assert hasattr(alg, 'deltask')
        assert hasattr(alg, 'minsk')
        assert len(alg.deltask) == len(Ns)
        
        del alg

    def test_qowe_partitioning_consistency(self, mpi_comm):
        """Test that QOWE partitions state consistently across ranks."""
        from quop_mpi.algorithm.multivariable import qowe, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        
        deltas, mins = setup_cartesian(Ns, bounds)
        alg = qowe(Ns, deltas, mins, mpi_comm)
        # For QOWE, only pass function - Ns, deltas, mins are auto-bound from attributes
        alg.set_qualities(cartesian, {"args": [sphere]})
        alg.set_depth(1)
        alg.setup()
        
        all_local_i = mpi_comm.allgather(alg.local_i)
        total = sum(all_local_i)
        
        assert total == alg.system_size
        
        del alg

    def test_qowe_preserves_normalization(self, mpi_comm):
        """Test that QOWE evolution preserves state normalization."""
        from quop_mpi.algorithm.multivariable import qowe, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]
        
        deltas, mins = setup_cartesian(Ns, bounds)
        alg = qowe(Ns, deltas, mins, mpi_comm)
        # For QOWE, only pass function - Ns, deltas, mins are auto-bound from attributes
        alg.set_qualities(cartesian, {"args": [sphere]})
        alg.set_depth(1)
        
        rng = np.random.default_rng(42)
        n_params = alg.total_params * alg.ansatz_depth
        params = rng.random(n_params) * 0.5  # smaller params for stability
        alg.evolve_state(params)
        
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            total_prob = np.sum(probs)
            assert abs(total_prob - 1.0) < 1e-10
        
        del alg

    def test_qowe_identity_evolution(self, mpi_comm):
        """Test that zero parameters give identity evolution."""
        from quop_mpi.algorithm.multivariable import qowe, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        
        deltas, mins = setup_cartesian(Ns, bounds)
        alg = qowe(Ns, deltas, mins, mpi_comm)
        # For QOWE, only pass function - Ns, deltas, mins are auto-bound from attributes
        alg.set_qualities(cartesian, {"args": [sphere]})
        alg.set_depth(1)
        
        n_params = alg.total_params * alg.ansatz_depth
        params = np.zeros(n_params)
        alg.evolve_state(params)
        
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            # QOWE uses position_grid initial state, not uniform superposition
            # Just check normalization
            total_prob = np.sum(probs)
            assert abs(total_prob - 1.0) < 1e-10
        
        del alg

    def test_qowe_deterministic_evolution(self, mpi_comm):
        """Test that same parameters produce same results with fixed seed."""
        from quop_mpi.algorithm.multivariable import qowe, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)
        
        # QOWE with 2D grid, depth=1: 1 gamma + 2 t's = 3 params
        params = np.array([0.3, 0.2, 0.1])
        
        # First run - seed np.random for position_grid's random mean
        np.random.seed(42)
        alg1 = qowe(Ns, deltas, mins, mpi_comm)
        # For QOWE, only pass function - Ns, deltas, mins are auto-bound from attributes
        alg1.set_qualities(cartesian, {"args": [sphere]})
        alg1.set_depth(1)
        alg1.evolve_state(params)
        state1 = alg1.get_final_state()
        del alg1
        
        # Second run - same seed should give same initial state
        np.random.seed(42)
        alg2 = qowe(Ns, deltas, mins, mpi_comm)
        alg2.set_qualities(cartesian, {"args": [sphere]})
        alg2.set_depth(1)
        alg2.evolve_state(params)
        state2 = alg2.get_final_state()
        del alg2
        
        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(state1, state2, rtol=1e-12)


# =============================================================================
# Cross-Algorithm Tests
# =============================================================================

@pytest.mark.mpi
class TestMultivariableConsistency:
    """Tests ensuring consistency across multivariable algorithms."""

    def test_qmoa_qowe_same_partitioning(self, mpi_comm):
        """Test that QMOA and QOWE produce same partitioning for same grid."""
        from quop_mpi.algorithm.multivariable import qmoa, qowe, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)
        
        alg_qmoa = qmoa(Ns, mpi_comm)
        alg_qmoa.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg_qmoa.set_depth(1)
        alg_qmoa.setup()
        
        alg_qowe = qowe(Ns, deltas, mins, mpi_comm)
        # For QOWE, only pass function - Ns, deltas, mins are auto-bound from attributes
        alg_qowe.set_qualities(cartesian, {"args": [sphere]})
        alg_qowe.set_depth(1)
        alg_qowe.setup()
        
        assert alg_qmoa.local_i == alg_qowe.local_i
        assert alg_qmoa.local_i_offset == alg_qowe.local_i_offset
        
        del alg_qmoa
        del alg_qowe

    def test_different_dimensions_different_system_sizes(self, mpi_comm):
        """Test that different grid dimensions produce different system sizes."""
        from quop_mpi.algorithm.multivariable import qmoa, setup_cartesian, cartesian
        
        # 1D
        Ns_1d = [3]
        bounds_1d = [[-1.0, 1.0]]
        deltas_1d, mins_1d = setup_cartesian(Ns_1d, bounds_1d)
        alg_1d = qmoa(Ns_1d, mpi_comm)
        
        # 2D
        Ns_2d = [2, 2]
        bounds_2d = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas_2d, mins_2d = setup_cartesian(Ns_2d, bounds_2d)
        alg_2d = qmoa(Ns_2d, mpi_comm)
        
        # 3D
        Ns_3d = [2, 2, 2]
        bounds_3d = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
        deltas_3d, mins_3d = setup_cartesian(Ns_3d, bounds_3d)
        alg_3d = qmoa(Ns_3d, mpi_comm)
        
        assert alg_1d.system_size == 2**3  # 8
        assert alg_2d.system_size == (2**2) * (2**2)  # 16
        assert alg_3d.system_size == (2**2) ** 3  # 64
        
        del alg_1d
        del alg_2d
        del alg_3d


# =============================================================================
# Tests for setup_cartesian and cartesian operator function
# =============================================================================

@pytest.mark.mpi
class TestCartesianSetup:
    """Tests for the cartesian grid setup utilities."""

    def test_setup_cartesian_returns_correct_structure(self, mpi_comm):
        """Test that setup_cartesian returns deltas and mins arrays."""
        from quop_mpi.algorithm.multivariable import setup_cartesian
        
        Ns = [2, 3]
        bounds = [[-1.0, 1.0], [-2.0, 2.0]]
        
        deltas, mins = setup_cartesian(Ns, bounds)
        
        # Should return arrays matching dimensions
        assert len(deltas) == len(Ns)
        assert len(mins) == len(Ns)
        
        # Mins should match lower bounds
        for i, bound in enumerate(bounds):
            assert mins[i] == pytest.approx(bound[0])

    def test_cartesian_operator_distributes_correctly(self, mpi_comm):
        """Test that cartesian operator function produces correct observables."""
        from quop_mpi.algorithm.multivariable import qmoa, setup_cartesian, cartesian
        
        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)
        
        alg = qmoa(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(1)
        
        # Need to trigger observable generation by evolving
        n_params = alg.total_params * alg.ansatz_depth
        params = np.zeros(n_params)
        alg.evolve_state(params)
        
        # Observables should be non-negative for sphere function
        local_obs = alg.observables
        assert local_obs is not None, "Observables should be set after evolve_state"
        assert np.all(local_obs >= 0), "Sphere function should be non-negative"
        
        del alg
