"""
Tests for propagator unitaries.

These tests verify that each propagator type correctly:
1. Distributes operator data via gen_operator()
2. Performs state evolution via propagate()
3. Produces consistent results across MPI ranks

Propagator types tested:
- diagonal: Phase-shift unitaries (e^{-i * gamma * H_diagonal})
- circulant: Complete graph mixing (used by QWOA)
- sparse: Sparse matrix exponential (used by QAOA hypercube mixer)
- composite: Multi-dimensional grid operators (used by qmoa)

Run with: mpiexec -n <N> python -m pytest tests/mpi/test_propagators.py -v --with-mpi
"""

import numpy as np
import pytest

# =============================================================================
# Helper Functions
# =============================================================================


def assert_probabilities_normalized(probs, *, atol=1e-8, context=""):
    """Assert that a probability vector remains normalized within backend noise."""
    total_prob = float(np.sum(probs, dtype=np.float64))
    np.testing.assert_allclose(
        total_prob,
        1.0,
        rtol=0.0,
        atol=atol,
        err_msg=f"{context} total probability {total_prob} exceeds tolerance {atol}",
    )


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
    from quop_mpi._utils._mpi import gather_array

    return gather_array(local_state, partition_table, mpi_comm)


def _scaled_grid_exponents(mpi_sizing, base_exponents):
    """Scale multivariable grid resolution while preserving dimensions."""
    exponents = [int(exponent) for exponent in base_exponents]
    extra_bits = max(0, (mpi_sizing.topology.world_size - 1).bit_length() - 1)
    for _ in range(extra_bits):
        smallest = min(exponents)
        index = exponents.index(smallest)
        exponents[index] += 1
    return exponents


@pytest.fixture
def propagator_small_system_size(small_system_size):
    """Small power-of-two size for lightweight propagator checks."""
    return small_system_size


@pytest.fixture
def propagator_medium_system_size(mpi_sizing):
    """Moderate power-of-two size for deeper sparse-propagator checks."""
    return mpi_sizing.power_of_two(base=32, min_per_rank=1)


@pytest.fixture
def propagator_grid_ns_2d(mpi_sizing):
    """2D multivariable grid that scales with MPI size."""
    return _scaled_grid_exponents(mpi_sizing, [2, 2])


@pytest.fixture
def propagator_grid_ns_3d(mpi_sizing):
    """3D multivariable grid that scales with MPI size."""
    return _scaled_grid_exponents(mpi_sizing, [2, 2, 2])


# =============================================================================
# Tests for Diagonal Propagator
# =============================================================================


@pytest.mark.mpi
class TestDiagonalPropagator:
    """Tests for the diagonal (phase-shift) propagator."""

    def test_diagonal_propagator_via_qaoa(self, mpi_comm, propagator_small_system_size):
        """Test diagonal propagator through QAOA (indirect test)."""
        from quop_mpi.algorithm.combinatorial import QAOA

        system_size = propagator_small_system_size

        def qualities(local_i, local_i_offset):
            # Simple linear qualities
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        alg = QAOA(system_size, mpi_comm)
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

        alg.destroy()

    def test_diagonal_phase_shift_correctness(self, mpi_comm, propagator_small_system_size):
        """Test that phase-shift applies correct phases to state."""
        from quop_mpi.algorithm.combinatorial import QAOA

        system_size = propagator_small_system_size

        # Uniform qualities = all same phase shift, state unchanged (up to global phase)
        def uniform_qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        alg = QAOA(system_size, mpi_comm)
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

        alg.destroy()


# =============================================================================
# Tests for Circulant Propagator
# =============================================================================


@pytest.mark.mpi
class TestCirculantPropagator:
    """Tests for the circulant (complete graph) propagator."""

    def test_circulant_propagator_via_qwoa(self, mpi_comm, propagator_small_system_size):
        """Test circulant propagator through QWOA (indirect test)."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = propagator_small_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        # Zero parameters = no evolution
        params = np.array([0.0, 0.0])
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            expected_prob = 1.0 / system_size
            np.testing.assert_allclose(probs, expected_prob, rtol=1e-10)

        alg.destroy()

    def test_circulant_complete_graph_mixing(self, mpi_comm, propagator_small_system_size):
        """Test complete graph mixing preserves total probability."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = propagator_small_system_size

        def qualities(local_i, local_i_offset):
            return np.zeros(local_i, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        # Apply mixing only (gamma=0, t=pi/N)
        t = np.pi / system_size
        params = np.array([0.0, t])
        alg.evolve_state(params)

        # Probability should still sum to 1
        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert_probabilities_normalized(
                probs, context="Complete-graph mixing should preserve normalization"
            )

        alg.destroy()


# =============================================================================
# Tests for Sparse Propagator
# =============================================================================


@pytest.mark.mpi
class TestSparsePropagator:
    """Tests for the sparse (hypercube) propagator."""

    def test_sparse_hypercube_preserves_normalization(self, mpi_comm, propagator_small_system_size):
        """Test hypercube mixing preserves total probability."""
        from quop_mpi.algorithm.combinatorial import QAOA

        system_size = propagator_small_system_size

        def qualities(local_i, local_i_offset):
            return np.zeros(local_i, dtype=np.float64)

        alg = QAOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        # Apply mixing (gamma=0, t=pi/4)
        params = np.array([0.0, np.pi / 4])
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert_probabilities_normalized(
                probs, context="Sparse hypercube mixing should preserve normalization"
            )

        alg.destroy()

    def test_sparse_multi_depth_preserves_normalization(
        self, mpi_comm, propagator_medium_system_size
    ):
        """Test that sparse propagator works over multiple evolution steps."""
        from quop_mpi.algorithm.combinatorial import QAOA

        system_size = propagator_medium_system_size

        def qualities(local_i, local_i_offset):
            return np.sin(np.arange(local_i) + local_i_offset).astype(np.float64)

        alg = QAOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(3)  # Multiple layers

        # 3 layers = 6 parameters
        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        alg.evolve_state(params)

        probs = alg.get_probabilities()

        if mpi_comm.Get_rank() == 0:
            assert_probabilities_normalized(
                probs,
                context="Sparse multi-depth evolution should preserve normalization",
            )
            # All probabilities should be non-negative
            assert np.all(probs >= -1e-15), "Negative probabilities found"

        alg.destroy()


# =============================================================================
# Tests for Composite Propagator
# =============================================================================


@pytest.mark.mpi
class TestCompositePropagator:
    """Tests for the composite (multi-dimensional) propagator."""

    def test_composite_is_planner(self, mpi_comm):
        """Test that composite unitary has planner=True."""
        from quop_mpi.propagator.composite import operator, unitary

        Ns = [3, 3]  # noqa: N806
        u = unitary(Ns, operator.ith)

        assert u.planner


# =============================================================================
# State Evolution Correctness Tests
# =============================================================================


@pytest.mark.mpi
class TestStateEvolutionCorrectness:
    """Tests verifying state evolution produces correct results."""

    def test_identity_evolution(self, mpi_comm, propagator_small_system_size):
        """Test that zero parameters give identity evolution."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = propagator_small_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
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

        alg.destroy()

    def test_evolution_is_unitary(self, mpi_comm, propagator_small_system_size):
        """Test that evolution preserves state norm."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = propagator_small_system_size

        def qualities(local_i, local_i_offset):
            return np.random.default_rng(42).random(local_i)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(2)

        # Random parameters
        params = np.random.default_rng(123).random(4) * 2 * np.pi
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert_probabilities_normalized(
                probs, context="State evolution should preserve normalization"
            )

        alg.destroy()

    def test_deterministic_evolution(self, mpi_comm, propagator_small_system_size):
        """Test that same parameters give same result."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = propagator_small_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        params = np.array([0.5, 0.3])

        # First evolution
        alg1 = QWOA(system_size, mpi_comm)
        alg1.set_qualities(qualities)
        alg1.set_depth(1)
        alg1.evolve_state(params)
        state1 = alg1.get_final_state()
        alg1.destroy()

        # Second evolution with same params
        alg2 = QWOA(system_size, mpi_comm)
        alg2.set_qualities(qualities)
        alg2.set_depth(1)
        alg2.evolve_state(params)
        state2 = alg2.get_final_state()
        alg2.destroy()

        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(state1, state2, rtol=1e-14)

    def test_different_params_different_results(self, mpi_comm, propagator_small_system_size):
        """Test that different parameters give different states."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = propagator_small_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
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

        alg.destroy()


# =============================================================================
# Tests for Momentum Propagator (used by QOWE)
# =============================================================================


@pytest.mark.mpi
class TestMomentumPropagator:
    """Tests for the momentum (FFT-based kinetic) propagator used by QOWE."""

    def test_momentum_propagator_via_qowe(self, mpi_comm, propagator_grid_ns_2d):
        """Test momentum propagator through QOWE algorithm."""
        from quop_mpi.algorithm.multivariable import QOWE, cartesian, setup_cartesian

        def sphere(x):
            """Sphere function: x is 2D array, return sum of squares per row."""
            return np.sum(x**2, axis=1)

        Ns = propagator_grid_ns_2d  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)

        alg = QOWE(Ns, deltas, mins, mpi_comm)
        alg.set_qualities(cartesian, {"args": [sphere]})
        alg.set_depth(1)

        # Zero params = no evolution, uniform state
        n_params = alg.total_params * alg.ansatz_depth
        params = np.zeros(n_params)
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert_probabilities_normalized(
                probs, context="Momentum propagator should preserve normalization"
            )

        alg.destroy()

    def test_momentum_propagator_preserves_normalization(self, mpi_comm, propagator_grid_ns_2d):
        """Test that momentum propagation preserves state normalization."""
        from quop_mpi.algorithm.multivariable import QOWE, cartesian, setup_cartesian

        def sphere(x):
            """Sphere function: x is 2D array, return sum of squares per row."""
            return np.sum(x**2, axis=1)

        Ns = propagator_grid_ns_2d  # noqa: N806
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]
        deltas, mins = setup_cartesian(Ns, bounds)

        np.random.seed(42)  # Seed for reproducible position_grid initial state
        alg = QOWE(Ns, deltas, mins, mpi_comm)
        alg.set_qualities(cartesian, {"args": [sphere]})
        alg.set_depth(2)

        # Non-zero params for actual evolution
        rng = np.random.default_rng(123)
        n_params = alg.total_params * alg.ansatz_depth
        params = rng.random(n_params) * 0.5
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert_probabilities_normalized(
                probs, context="Momentum propagator should remain normalized after evolution"
            )

        alg.destroy()

    def test_momentum_propagator_different_times_different_states(
        self, mpi_comm, propagator_grid_ns_2d
    ):
        """Test that different evolution times produce different states."""
        from quop_mpi.algorithm.multivariable import QOWE, cartesian, setup_cartesian

        def quadratic(x):
            """Quadratic function: x is 2D array with columns [x0, x1]."""
            return x[:, 0] ** 2 + 2 * x[:, 1] ** 2

        Ns = propagator_grid_ns_2d  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)

        # First evolution with small times
        np.random.seed(42)
        alg1 = QOWE(Ns, deltas, mins, mpi_comm)
        alg1.set_qualities(cartesian, {"args": [quadratic]})
        alg1.set_depth(1)
        params1 = np.array([0.1, 0.1, 0.1])
        alg1.evolve_state(params1)
        state1 = alg1.get_final_state()
        alg1.destroy()

        # Second evolution with larger times
        np.random.seed(42)  # Same initial state
        alg2 = QOWE(Ns, deltas, mins, mpi_comm)
        alg2.set_qualities(cartesian, {"args": [quadratic]})
        alg2.set_depth(1)
        params2 = np.array([0.5, 0.5, 0.5])
        alg2.evolve_state(params2)
        state2 = alg2.get_final_state()
        alg2.destroy()

        if mpi_comm.Get_rank() == 0:
            diff = np.linalg.norm(state1 - state2)
            assert diff > 1e-6, "Different evolution times should produce different states"

    def test_momentum_propagator_deterministic(self, mpi_comm, propagator_grid_ns_2d):
        """Test that momentum propagation is deterministic."""
        from quop_mpi.algorithm.multivariable import QOWE, cartesian, setup_cartesian

        def sphere(x):
            """Sphere function: x is 2D array, return sum of squares per row."""
            return np.sum(x**2, axis=1)

        Ns = propagator_grid_ns_2d  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)

        params = np.array([0.3, 0.2, 0.1])

        # First run
        np.random.seed(42)
        alg1 = QOWE(Ns, deltas, mins, mpi_comm)
        alg1.set_qualities(cartesian, {"args": [sphere]})
        alg1.set_depth(1)
        alg1.evolve_state(params)
        state1 = alg1.get_final_state()
        alg1.destroy()

        # Second run with same seed
        np.random.seed(42)
        alg2 = QOWE(Ns, deltas, mins, mpi_comm)
        alg2.set_qualities(cartesian, {"args": [sphere]})
        alg2.set_depth(1)
        alg2.evolve_state(params)
        state2 = alg2.get_final_state()
        alg2.destroy()

        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(state1, state2, rtol=1e-12)

    def test_momentum_propagator_multi_depth(self, mpi_comm, propagator_grid_ns_2d):
        """Test momentum propagation with multiple ansatz depths."""
        from quop_mpi.algorithm.multivariable import QOWE, cartesian, setup_cartesian

        def rosenbrock(x):
            """Rosenbrock function: x is 2D array with columns [x0, x1]."""
            return (1 - x[:, 0]) ** 2 + 100 * (x[:, 1] - x[:, 0] ** 2) ** 2

        Ns = propagator_grid_ns_2d  # noqa: N806
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]
        deltas, mins = setup_cartesian(Ns, bounds)

        np.random.seed(42)
        alg = QOWE(Ns, deltas, mins, mpi_comm)
        alg.set_qualities(cartesian, {"args": [rosenbrock]})
        alg.set_depth(3)  # Three iterations

        # 3 params per layer * 3 layers = 9 params
        rng = np.random.default_rng(99)
        n_params = alg.total_params * alg.ansatz_depth
        params = rng.random(n_params) * 0.3

        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert_probabilities_normalized(
                probs, context="Momentum multi-depth evolution should preserve normalization"
            )

        alg.destroy()

    def test_momentum_propagator_3d_grid(self, mpi_comm, propagator_grid_ns_3d):
        """Test momentum propagator with 3D grid."""
        from quop_mpi.algorithm.multivariable import QOWE, cartesian, setup_cartesian

        def sphere_3d(x):
            """3D sphere function: x is 2D array with columns [x0, x1, x2]."""
            return np.sum(x**2, axis=1)

        Ns = propagator_grid_ns_3d  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)

        np.random.seed(42)
        alg = QOWE(Ns, deltas, mins, mpi_comm)
        alg.set_qualities(cartesian, {"args": [sphere_3d]})
        alg.set_depth(1)

        # 3D QOWE: 1 gamma + 3 t's = 4 params
        assert alg.total_params == 4

        n_params = alg.total_params * alg.ansatz_depth
        params = np.zeros(n_params)
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert_probabilities_normalized(
                probs, context="3D momentum evolution should preserve normalization"
            )

        alg.destroy()

    def test_momentum_space_parameters_correct(self, mpi_comm, propagator_grid_ns_2d):
        """Test that momentum-space grid parameters are correctly computed."""
        from quop_mpi.algorithm.multivariable import QOWE, setup_cartesian

        Ns = propagator_grid_ns_2d  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)

        alg = QOWE(Ns, deltas, mins, mpi_comm)

        # Check deltask = 2*pi / (N * delta)
        for i, (n, delta) in enumerate(zip(alg.Ns, deltas, strict=True)):
            expected_deltask = 2 * np.pi / (n * delta)
            assert abs(alg.deltask[i] - expected_deltask) < 1e-12

        # Check minsk = -(N/2) * deltask
        for i, (n, dk) in enumerate(zip(alg.Ns, alg.deltask, strict=True)):
            expected_minsk = -(n / 2) * dk
            assert abs(alg.minsk[i] - expected_minsk) < 1e-12

        alg.destroy()
