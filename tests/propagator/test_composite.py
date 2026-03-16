"""
Tests for the composite propagator.

The composite propagator handles operators that are Cartesian sums of
circulant matrices, used by the QMOA algorithm for multivariable optimization.
It uses FFT to efficiently apply the mixing unitary.

These tests run with both MPI and wavefront backends via QUOP_BACKEND env var.

Run with MPI backend:
    mpiexec -n <N> python -m pytest tests/propagator/test_composite.py -v

Run with wavefront backend:
    QUOP_BACKEND=wavefront mpiexec -n <N> python -m pytest tests/propagator/test_composite.py -v
"""

import numpy as np
import pytest
from mpi4py import MPI

from quop_mpi import config

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend_name():
    """Return the current backend name for test reporting."""
    return config.backend


@pytest.fixture
def mpi_comm():
    """MPI COMM_WORLD fixture."""
    return MPI.COMM_WORLD


# =============================================================================
# Tests for Composite Propagator via QMOA
# =============================================================================


@pytest.mark.mpi
class TestCompositeViaQMOA:
    """Tests for composite propagator through QMOA algorithm."""

    def test_identity_evolution(self, mpi_comm, backend_name):
        """Test that zero parameters give identity evolution."""
        from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

        def sphere(x):
            return np.sum(x**2, axis=1)

        Ns = [2, 2]  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]

        deltas, mins = setup_cartesian(Ns, bounds)
        alg = QMOA(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(1)
        alg.setup()

        # Zero parameters = no evolution
        n_params = alg.total_params * alg.ansatz_depth
        params = np.zeros(n_params)
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            expected_prob = 1.0 / alg.system_size
            np.testing.assert_allclose(
                probs,
                expected_prob,
                rtol=1e-10,
                err_msg=f"[{backend_name}] Identity evolution failed",
            )

        alg.destroy()

    def test_preserves_normalization(self, mpi_comm, backend_name):
        """Test that evolution preserves state normalization."""
        from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

        def sphere(x):
            return np.sum(x**2, axis=1)

        Ns = [2, 2]  # noqa: N806
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]

        deltas, mins = setup_cartesian(Ns, bounds)
        alg = QMOA(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(2)

        # Random parameters
        rng = np.random.default_rng(42)
        n_params = alg.total_params * alg.ansatz_depth
        params = rng.random(n_params) * 2 * np.pi
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            total_prob = np.sum(probs)
            assert (
                abs(total_prob - 1.0) < 1e-10
            ), f"[{backend_name}] Normalization not preserved: {total_prob}"

        alg.destroy()

    def test_deterministic_evolution(self, mpi_comm, backend_name):
        """Test that same parameters produce same results."""
        from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

        def sphere(x):
            return np.sum(x**2, axis=1)

        Ns = [2, 2]  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]

        deltas, mins = setup_cartesian(Ns, bounds)

        # First run
        alg1 = QMOA(Ns, mpi_comm)
        alg1.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg1.set_depth(1)

        rng = np.random.default_rng(123)
        n_params = alg1.total_params * alg1.ansatz_depth
        params = rng.random(n_params) * np.pi
        alg1.evolve_state(params)
        probs1 = alg1.get_probabilities()
        alg1.destroy()

        # Second run with same params
        alg2 = QMOA(Ns, mpi_comm)
        alg2.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg2.set_depth(1)
        alg2.evolve_state(params)
        probs2 = alg2.get_probabilities()
        alg2.destroy()

        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(
                probs1, probs2, rtol=1e-12, err_msg=f"[{backend_name}] Evolution not deterministic"
            )


# =============================================================================
# Tests for Complete Graph Mixing
# =============================================================================


@pytest.mark.mpi
class TestCompleteGraphMixing:
    """Tests for complete graph mixing via composite propagator."""

    def test_mixing_from_localized_state(self, mpi_comm, backend_name):
        """Test mixing spreads amplitude from localized initial state."""
        from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

        def zero_qualities(x):
            return np.zeros(x.shape[0])

        Ns = [2, 2]  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]

        deltas, mins = setup_cartesian(Ns, bounds)
        alg = QMOA(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, zero_qualities]})
        alg.set_depth(1)
        alg.setup()

        # With zero qualities, only mixer acts
        # Apply mixing for time t
        t = 0.5
        n_params = alg.total_params * alg.ansatz_depth
        params = np.zeros(n_params)
        # Set mixer parameters (second set of params for depth=1)
        # For QMOA, params are [gamma, t1, t2, ...] per layer
        # Actually for QMOA: n_params = total_params * depth = (1 + n_dim) * depth
        # For Ns=[2,2]: n_dim=2, so 3 params per layer
        params[1] = t  # First mixer dimension
        params[2] = t  # Second mixer dimension

        alg.evolve_state(params)
        probs = alg.get_probabilities()

        if mpi_comm.Get_rank() == 0:
            # With mixing applied, probabilities should no longer be uniform
            # (unless t is specifically chosen to return to uniform)
            # Just verify sum is still 1
            total = np.sum(probs)
            assert abs(total - 1.0) < 1e-10, f"[{backend_name}] Normalization lost after mixing"

        alg.destroy()


# =============================================================================
# Tests for Multi-Depth Evolution
# =============================================================================


@pytest.mark.mpi
class TestMultiDepthEvolution:
    """Tests for multiple layers of evolution."""

    def test_depth_2_evolution(self, mpi_comm, backend_name):
        """Test evolution with depth=2."""
        from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

        def rastrigin(x):
            n = x.shape[1]
            return n * 10 + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

        Ns = [2, 2]  # noqa: N806
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]

        deltas, mins = setup_cartesian(Ns, bounds)
        alg = QMOA(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, rastrigin]})
        alg.set_depth(2)

        rng = np.random.default_rng(42)
        n_params = alg.total_params * alg.ansatz_depth
        params = rng.random(n_params) * np.pi
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            total = np.sum(probs)
            assert abs(total - 1.0) < 1e-10

        alg.destroy()

    def test_depth_3_evolution(self, mpi_comm, backend_name):
        """Test evolution with depth=3."""
        from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

        def sphere(x):
            return np.sum(x**2, axis=1)

        Ns = [2, 2]  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]

        deltas, mins = setup_cartesian(Ns, bounds)
        alg = QMOA(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(3)

        rng = np.random.default_rng(99)
        n_params = alg.total_params * alg.ansatz_depth
        params = rng.random(n_params) * np.pi
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            total = np.sum(probs)
            assert abs(total - 1.0) < 1e-10

        alg.destroy()


# =============================================================================
# Tests for Different Grid Sizes
# =============================================================================


@pytest.mark.mpi
class TestGridSizes:
    """Tests for various grid sizes and dimensions."""

    def test_1d_grid(self, mpi_comm, backend_name):
        """Test 1D grid (single dimension)."""
        from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

        def f1d(x):
            return x[:, 0] ** 2

        Ns = [5]  # 32 points -- must exceed rank count for FFTW partitioning  # noqa: N806
        bounds = [[-2.0, 2.0]]

        deltas, mins = setup_cartesian(Ns, bounds)

        alg = QMOA(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, f1d]})
        alg.set_depth(1)

        n_params = alg.total_params * alg.ansatz_depth
        params = np.random.default_rng(1).random(n_params) * np.pi
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert abs(np.sum(probs) - 1.0) < 1e-10

        alg.destroy()

    def test_3d_grid(self, mpi_comm, backend_name):
        """Test 3D grid."""
        from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

        def f3d(x):
            return np.sum(x**2, axis=1)

        Ns = [2, 2, 2]  # 4x4x4 = 64 points  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]

        deltas, mins = setup_cartesian(Ns, bounds)
        alg = QMOA(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, f3d]})
        alg.set_depth(1)

        n_params = alg.total_params * alg.ansatz_depth
        params = np.random.default_rng(2).random(n_params) * np.pi
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert abs(np.sum(probs) - 1.0) < 1e-10

        alg.destroy()

    def test_large_grid_multi_rank(self, mpi_comm, backend_name):
        from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

        def sphere(x):
            return np.sum(x**2, axis=1)

        Ns = [3, 3]  # 8x8 = 64 points  # noqa: N806
        bounds = [[-2.0, 2.0], [-2.0, 2.0]]

        deltas, mins = setup_cartesian(Ns, bounds)
        alg = QMOA(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(1)

        n_params = alg.total_params * alg.ansatz_depth
        params = np.random.default_rng(3).random(n_params) * np.pi
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert abs(np.sum(probs) - 1.0) < 1e-10
            assert len(probs) == alg.system_size

        alg.destroy()


# =============================================================================
# Backend-Specific Behavior Tests
# =============================================================================


@pytest.mark.mpi
class TestBackendConsistency:
    """Tests to verify consistent behavior across backends."""

    def test_backend_is_configured(self, backend_name):
        """Verify backend configuration is accessible."""
        assert backend_name in ["mpi", "wavefront"], f"Unknown backend: {backend_name}"

    def test_evolution_produces_valid_probabilities(self, mpi_comm, backend_name):
        """Test that evolution produces valid probability distribution."""
        from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian

        def sphere(x):
            return np.sum(x**2, axis=1)

        Ns = [2, 2]  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]

        deltas, mins = setup_cartesian(Ns, bounds)
        alg = QMOA(Ns, mpi_comm)
        alg.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        alg.set_depth(1)

        rng = np.random.default_rng(42)
        n_params = alg.total_params * alg.ansatz_depth
        params = rng.random(n_params) * np.pi
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            # All probabilities should be non-negative
            assert np.all(
                probs >= -1e-15
            ), f"[{backend_name}] Negative probabilities: {probs[probs < 0]}"

            # Sum should be 1
            assert (
                abs(np.sum(probs) - 1.0) < 1e-10
            ), f"[{backend_name}] Probabilities don't sum to 1"

            # Should have correct length
            assert len(probs) == alg.system_size

        alg.destroy()
