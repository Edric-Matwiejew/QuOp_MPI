"""
Tests for the circulant propagator.

The circulant propagator handles operators that are circulant matrices,
used for implementing quantum walk mixing unitaries in QWOA and related algorithms.
It uses FFT to efficiently apply the unitary U = F^{-1} diag(exp(-i*t*eigenvalues)) F.

These tests run with both MPI and wavefront backends via QUOP_BACKEND env var.

Run with MPI backend:
    mpiexec -n <N> python -m pytest tests/propagator/test_circulant.py -v

Run with wavefront backend:
    QUOP_BACKEND=wavefront mpiexec -n <N> python -m pytest tests/propagator/test_circulant.py -v
"""

import numpy as np
import pytest


# =============================================================================
# Tests for Circulant Propagator via QWOA
# =============================================================================


@pytest.mark.mpi
class TestCirculantViaQWOA:
    """Tests for circulant propagator through QWOA algorithm."""

    def test_identity_evolution(self, mpi_comm, backend_name, circulant_small_system_size):
        """Test that zero parameters give identity evolution."""
        from quop_mpi.algorithm.combinatorial import QWOA, serial

        system_size = circulant_small_system_size

        def qualities_func():
            return np.ones(system_size, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [qualities_func]})
        alg.set_depth(1)
        alg.prepare()

        # Zero parameters = no evolution
        n_params = alg.total_params * alg.ansatz_depth
        params = np.zeros(n_params)
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            expected_prob = 1.0 / system_size
            np.testing.assert_allclose(
                probs,
                expected_prob,
                rtol=1e-10,
                err_msg=f"[{backend_name}] Identity evolution failed",
            )

        alg.destroy()

    def test_preserves_normalization(self, mpi_comm, backend_name, circulant_medium_system_size):
        """Test that evolution preserves state normalization."""
        from quop_mpi.algorithm.combinatorial import QWOA, serial

        system_size = circulant_medium_system_size

        def random_qualities():
            rng = np.random.default_rng(42)
            return rng.random(system_size).astype(np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [random_qualities]})
        alg.set_depth(2)
        alg.prepare()

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

    def test_deterministic_evolution(self, mpi_comm, backend_name, circulant_small_system_size):
        """Test that same parameters produce same results."""
        from quop_mpi.algorithm.combinatorial import QWOA, serial

        system_size = circulant_small_system_size

        def qualities_func():
            return np.arange(system_size, dtype=np.float64)

        # First run
        alg1 = QWOA(system_size, mpi_comm)
        alg1.set_qualities(serial, {"args": [qualities_func]})
        alg1.set_depth(1)
        alg1.prepare()

        rng = np.random.default_rng(123)
        n_params = alg1.total_params * alg1.ansatz_depth
        params = rng.random(n_params) * np.pi
        alg1.evolve_state(params)
        probs1 = alg1.get_probabilities()
        alg1.destroy()

        # Second run with same params
        alg2 = QWOA(system_size, mpi_comm)
        alg2.set_qualities(serial, {"args": [qualities_func]})
        alg2.set_depth(1)
        alg2.evolve_state(params)
        probs2 = alg2.get_probabilities()
        alg2.destroy()

        if mpi_comm.Get_rank() == 0:
            np.testing.assert_allclose(
                probs1, probs2, rtol=1e-12, err_msg=f"[{backend_name}] Evolution not deterministic"
            )

    def test_destroy_is_idempotent_after_prepare(
        self, mpi_comm, backend_name, circulant_small_system_size
    ):
        """Destroying a planned QWOA multiple times should stay safe."""
        from quop_mpi.algorithm.combinatorial import QWOA, serial

        system_size = circulant_small_system_size

        def qualities_func():
            return np.ones(system_size, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [qualities_func]})
        alg.set_depth(1)
        alg.prepare()

        alg.destroy()
        alg.destroy()
        alg.destroy()


# =============================================================================
# Tests for Circulant Operator Types
# =============================================================================


@pytest.mark.mpi
class TestCirculantOperators:
    """Tests for different circulant operator types."""

    def test_complete_graph_operator(self, mpi_comm, backend_name, circulant_small_system_size):
        """Test evolution with complete graph operator."""
        from quop_mpi.algorithm.combinatorial import QWOA, serial

        system_size = circulant_small_system_size

        def qualities_func():
            return np.ones(system_size, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [qualities_func]})
        alg.set_depth(1)

        params = np.array([0.1, 0.2])  # gamma, beta
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert probs is not None
            assert len(probs) == system_size
            assert np.sum(probs) == pytest.approx(1.0, rel=1e-10)

        alg.destroy()

    def test_cycle_graph_operator(self, mpi_comm, backend_name, circulant_small_system_size):
        """Test evolution with cycle graph operator (i=1)."""
        from quop_mpi.ansatz import Ansatz
        from quop_mpi.propagator import diagonal
        from quop_mpi.propagator.circulant import operator, unitary

        system_size = circulant_small_system_size

        def qualities_func(local_i, local_i_offset, system_size):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        # Create ansatz with cycle graph mixing
        phase_unitary = diagonal.Unitary(diagonal.operator.observables)
        mixer_unitary = unitary(operator.graph, operator_dict={"args": [1]})  # Cycle graph (i=1)

        ansatz = Ansatz(system_size, mpi_comm)
        ansatz.set_unitaries([phase_unitary, mixer_unitary])
        ansatz.set_observables(qualities_func)
        ansatz.set_depth(1)
        ansatz.prepare()

        params = np.array([0.1, 0.2])
        ansatz.evolve_state(params)

        probs = ansatz.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert probs is not None
            assert np.sum(probs) == pytest.approx(1.0, rel=1e-10)

        ansatz.destroy()

    def test_general_circulant_operator(self, mpi_comm, backend_name, circulant_small_system_size):
        """Test evolution with general circulant graph operator."""
        from quop_mpi.ansatz import Ansatz
        from quop_mpi.propagator import diagonal
        from quop_mpi.propagator.circulant import operator, unitary

        system_size = circulant_small_system_size

        def qualities_func(local_i, local_i_offset, system_size):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        # Create ansatz with i=2 circulant graph
        phase_unitary = diagonal.Unitary(diagonal.operator.observables)
        mixer_unitary = unitary(operator.graph, operator_dict={"args": [2]})  # i=2 circulant

        ansatz = Ansatz(system_size, mpi_comm)
        ansatz.set_unitaries([phase_unitary, mixer_unitary])
        ansatz.set_observables(qualities_func)
        ansatz.set_depth(1)
        ansatz.prepare()

        params = np.array([0.1, 0.2])
        ansatz.evolve_state(params)

        probs = ansatz.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert probs is not None
            assert np.sum(probs) == pytest.approx(1.0, rel=1e-10)

        ansatz.destroy()

    def test_replanning_same_circulant_unitary_is_safe(
        self, mpi_comm, backend_name, circulant_small_system_size
    ):
        """Repeated gen_operator() calls should replace old FFTW plans safely."""
        from quop_mpi.ansatz import Ansatz
        from quop_mpi.propagator import diagonal
        from quop_mpi.propagator.circulant import operator, unitary

        system_size = circulant_small_system_size

        def qualities_func(local_i, local_i_offset, system_size):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        phase_unitary = diagonal.Unitary(diagonal.operator.observables)
        mixer_unitary = unitary(operator.graph, operator_dict={"args": [1]})

        ansatz = Ansatz(system_size, mpi_comm)
        ansatz.set_unitaries([phase_unitary, mixer_unitary])
        ansatz.set_observables(qualities_func)
        ansatz.set_depth(1)
        ansatz.prepare()

        if ansatz.subcomms.in_subcomm():
            ansatz.unitaries[1].gen_operator()
            ansatz.unitaries[1].gen_operator()

        params = np.array([0.1, 0.2])
        ansatz.evolve_state(params)

        probs = ansatz.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert probs is not None
            assert np.sum(probs) == pytest.approx(1.0, rel=1e-10)

        ansatz.destroy()


# =============================================================================
# Tests for Multi-rank Consistency
# =============================================================================


@pytest.mark.mpi
class TestMultiRankConsistency:
    """Tests that verify consistent results across MPI ranks."""

    @pytest.mark.requires_nprocs(2)
    def test_all_ranks_same_probabilities(
        self, mpi_comm, backend_name, circulant_medium_system_size
    ):
        """Test that all ranks compute the same probabilities."""
        from quop_mpi.algorithm.combinatorial import QWOA, serial

        system_size = circulant_medium_system_size

        def qualities_func():
            return np.arange(system_size, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [qualities_func]})
        alg.set_depth(1)

        params = np.array([0.5, 0.3])
        alg.evolve_state(params)

        probs = alg.get_probabilities()

        # Gather all probabilities from all ranks
        all_probs = mpi_comm.gather(probs, root=0)

        if mpi_comm.Get_rank() == 0:
            # Only rank 0 has valid probs in gather result
            for i, p in enumerate(all_probs):
                if p is not None:
                    np.testing.assert_allclose(
                        probs,
                        p,
                        rtol=1e-12,
                        err_msg=f"[{backend_name}] Rank {i} has different probabilities",
                    )

        alg.destroy()


# =============================================================================
# Tests for MaxCut Problem
# =============================================================================


@pytest.mark.mpi
class TestMaxCutProblem:
    """Tests using MaxCut as a realistic use case for circulant propagator."""

    def test_maxcut_small_graph(self, mpi_comm, backend_name):
        """Test QWOA on a small MaxCut instance."""
        import networkx as nx

        from quop_mpi.algorithm.combinatorial import QWOA, serial
        from quop_mpi.toolkit import I, Z

        # Create a simple cycle graph
        n_vertices = 4
        G = nx.cycle_graph(n_vertices)  # noqa: N806
        system_size = 2**n_vertices

        A = nx.to_scipy_sparse_array(G)  # noqa: N806

        def maxcut_qualities(A):  # noqa: N803
            C = 0  # noqa: N806
            n = A.shape[0]
            for i in range(n):
                for j in range(n):
                    if A[i, j] != 0:
                        C += 0.5 * (I(n) - (Z(i, n) @ Z(j, n)))  # noqa: N806
            return -C.diagonal()

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [maxcut_qualities, A]})
        alg.set_depth(2)

        params = np.array([0.1, 0.2, 0.3, 0.4])  # depth 2 = 4 params
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            assert probs is not None
            assert len(probs) == system_size
            assert np.sum(probs) == pytest.approx(1.0, rel=1e-10)

            # Verify probabilities are valid (non-negative)
            assert np.all(probs >= -1e-15)

        alg.destroy()

    def test_maxcut_optimization(self, mpi_comm, backend_name):
        """Test that QWOA can find good MaxCut solutions."""
        import networkx as nx

        from quop_mpi.algorithm.combinatorial import QWOA, serial
        from quop_mpi.toolkit import I, Z

        # Simple 3-node triangle graph
        n_vertices = 3
        G = nx.complete_graph(n_vertices)  # noqa: N806
        system_size = 2**n_vertices

        A = nx.to_scipy_sparse_array(G)  # noqa: N806

        def maxcut_qualities(A):  # noqa: N803
            C = 0  # noqa: N806
            n = A.shape[0]
            for i in range(n):
                for j in range(n):
                    if A[i, j] != 0:
                        C += 0.5 * (I(n) - (Z(i, n) @ Z(j, n)))  # noqa: N806
            return -C.diagonal()

        uniform_expectation = float(np.mean(maxcut_qualities(A)))

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [maxcut_qualities, A]})
        alg.set_depth(3)
        alg.set_seed(123)
        alg.execute()

        # Better-than-random here means lower cost than the uniform superposition
        # baseline for this fixed triangle instance.
        if mpi_comm.Get_rank() == 0:
            expectation = float(alg.result["fun"])
            assert np.isfinite(expectation)
            assert expectation < uniform_expectation

        alg.destroy()


# =============================================================================
# Backend Comparison Tests
# =============================================================================


@pytest.mark.mpi
class TestBackendConsistency:
    """Tests that can be used to compare MPI and wavefront backends."""

    def test_known_evolution(self, mpi_comm, backend_name, circulant_small_system_size):
        """Test evolution against known analytical results.

        For a complete graph with uniform initial state and uniform qualities,
        the evolution should produce predictable probability distributions.
        """
        from quop_mpi.algorithm.combinatorial import QWOA, serial

        system_size = circulant_small_system_size

        # Uniform qualities - all same
        def uniform_qualities():
            return np.ones(system_size, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [uniform_qualities]})
        alg.set_depth(1)

        # With uniform qualities, the phase unitary is just global phase
        # The mixing unitary with t=pi should flip amplitudes
        params = np.array([0.0, np.pi / 4])
        alg.evolve_state(params)

        probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0:
            expected = np.full(system_size, 1.0 / system_size, dtype=np.float64)
            np.testing.assert_allclose(
                probs,
                expected,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"[{backend_name}] Uniform state should remain invariant",
            )

        alg.destroy()

    def test_gradient_consistency(self, mpi_comm, backend_name, circulant_small_system_size):
        """Test that numerical gradients are consistent."""
        from quop_mpi.algorithm.combinatorial import QWOA, serial

        system_size = circulant_small_system_size

        def qualities_func():
            return np.arange(system_size, dtype=np.float64)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [qualities_func]})
        alg.set_depth(1)

        # First evolve to initialize state, then compute objective
        params = np.array([0.3, 0.4])
        alg.evolve_state(params)
        obj1 = alg.objective(params)

        eps = 1e-6
        forward_obj = alg.objective(params + np.array([eps, 0.0]))
        backward_obj = alg.objective(params - np.array([eps, 0.0]))
        half_eps = eps / 2
        forward_half = alg.objective(params + np.array([half_eps, 0.0]))
        backward_half = alg.objective(params - np.array([half_eps, 0.0]))

        if mpi_comm.Get_rank() == 0:
            grad_eps = (forward_obj - backward_obj) / (2 * eps)
            grad_half_eps = (forward_half - backward_half) / (2 * half_eps)
            assert np.isfinite(grad_eps)
            assert np.isfinite(grad_half_eps)
            assert np.isclose(
                grad_eps,
                grad_half_eps,
                rtol=1e-3,
                atol=1e-5,
            ), f"[{backend_name}] Gradient estimates drift: {grad_eps} vs {grad_half_eps}"

        alg.destroy()
