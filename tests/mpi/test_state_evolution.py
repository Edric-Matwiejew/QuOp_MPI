"""
Tests for quantum state evolution (evolve_state method).

These tests verify that the Ansatz.evolve_state() method correctly
evolves the quantum state according to the variational parameters.

Uses test oracles with analytically known outcomes to verify correctness.
For Grover-like tests we use ``create_qaoa_complete_graph`` to build a base
Ansatz with a diagonal phase separator and a sparse complete-graph mixer,
which is the QWOA-equivalent composition required for direct comparison
with QWOA at the same parameters.

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_state_evolution.py -v --with-mpi
"""

import numpy as np
import pytest

from tests.conftest import (
    TestOracle,
    create_qaoa_complete_graph,
    gather_state_probabilities,
)


def _scaled_power_of_two_system_size(mpi_sizing, base):
    """Choose a power-of-two state size large enough for the active MPI world."""
    return mpi_sizing.power_of_two(base=base, min_per_rank=1)


def _marked_count_from_ratio(system_size, denominator, minimum):
    """Preserve the original marked-state density while allowing larger systems."""
    return max(minimum, system_size // denominator)


@pytest.fixture
def simple_oracle(mpi_sizing):
    """Scale the multi-solution Grover oracle while preserving M/N = 1/16."""
    system_size = _scaled_power_of_two_system_size(mpi_sizing, base=64)
    return TestOracle(
        system_size=system_size,
        n_marked=_marked_count_from_ratio(system_size, denominator=16, minimum=4),
        seed=42,
    )


@pytest.fixture
def single_solution_oracle(mpi_sizing):
    """Scale the single-solution oracle with the active MPI world size."""
    system_size = _scaled_power_of_two_system_size(mpi_sizing, base=64)
    return TestOracle(system_size=system_size, n_marked=1, seed=123)


@pytest.mark.mpi
class TestEvolveStateBasic:
    """Test basic evolve_state functionality."""

    def test_evolve_state_runs_without_error(self, mpi_comm, simple_oracle):
        """Verify evolve_state completes without raising exceptions."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)

        # Should complete without error (evolve_state handles setup internally)
        alg.evolve_state(params)

        # State should be populated (use get_probabilities to verify)
        probs = alg.get_probabilities()
        if probs is not None:
            assert len(probs) == simple_oracle.system_size

        alg.destroy()

    def test_evolve_state_produces_normalized_state(self, mpi_comm, simple_oracle):
        """Verify the final state is properly normalized (probabilities sum to 1)."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        full_probs = gather_state_probabilities(alg, mpi_comm)

        if mpi_comm.Get_rank() == 0:
            total_prob = np.sum(full_probs)
            assert (
                abs(total_prob - 1.0) < 1e-10
            ), f"State not normalized: total probability = {total_prob}"

        alg.destroy()

    def test_evolve_state_is_deterministic(self, mpi_comm, simple_oracle):
        """Verify same parameters give same results."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)

        # First evolution - all ranks must call get_final_state (contains MPI gather)
        alg.evolve_state(params)
        state1 = alg.get_final_state()  # Returns state on rank 0, None on others

        # Second evolution with same params
        alg.evolve_state(params)
        state2 = alg.get_final_state()

        # Only rank 0 has the gathered state, so only check there
        if mpi_comm.Get_rank() == 0:
            assert state1 is not None, "get_final_state returned None on rank 0"
            assert state2 is not None, "get_final_state returned None on rank 0"
            assert np.allclose(state1, state2), "Same parameters should produce identical states"

        alg.destroy()


@pytest.mark.mpi
class TestEvolveStateParameterSensitivity:
    """Test that state evolution responds correctly to parameter changes."""

    def test_different_params_produce_different_states(self, mpi_comm, simple_oracle):
        """Verify different parameters lead to different final states."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Optimal parameters
        params1 = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params1)
        state1 = alg.get_final_state()  # All ranks must call (contains MPI gather)

        # Modified parameters
        params2 = params1 * 0.5
        alg.evolve_state(params2)
        state2 = alg.get_final_state()

        # Only rank 0 has the gathered state
        if mpi_comm.Get_rank() == 0:
            assert state1 is not None and state2 is not None
            assert not np.allclose(
                state1, state2
            ), "Different parameters should produce different states"

        alg.destroy()

    def test_zero_params_gives_uniform_state(self, mpi_comm, simple_oracle):
        """Verify zero parameters leave the state unchanged (uniform superposition)."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Zero parameters: no phase shift, no mixing
        params = np.zeros(2, dtype=np.float64)
        alg.evolve_state(params)

        full_probs = gather_state_probabilities(alg, mpi_comm)

        if mpi_comm.Get_rank() == 0:
            # Should be uniform: each state has probability 1/N
            expected_prob = 1.0 / simple_oracle.system_size
            assert np.allclose(
                full_probs, expected_prob, rtol=1e-6
            ), "Zero parameters should give uniform superposition"

        alg.destroy()


@pytest.mark.mpi
class TestEvolveStateCorrectness:
    """Test that state evolution produces theoretically correct results."""

    def test_qaoa_complete_graph_matches_grover(self, mpi_comm, simple_oracle):
        """
        Verify QAOA with complete graph mixer achieves Grover-like probability.

        When QAOA uses a complete graph as its mixing operator (instead of
        the default hypercube), it becomes equivalent to QWOA and should
        achieve the theoretical Grover success probability.
        """
        oracle = simple_oracle

        alg = create_qaoa_complete_graph(oracle.system_size, mpi_comm, oracle)
        alg.set_depth(1)
        alg.setup()

        params = oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        full_probs = gather_state_probabilities(alg, mpi_comm)

        if mpi_comm.Get_rank() == 0:
            marked_prob = oracle.compute_marked_probability(full_probs)
            theoretical = oracle.theoretical_success_probability(1)

            assert abs(marked_prob - theoretical) < 0.01, (
                f"QAOA with complete graph should match Grover theory: "
                f"got {marked_prob:.4f}, expected {theoretical:.4f}"
            )

        alg.destroy()

    def test_qwoa_matches_grover(self, mpi_comm, simple_oracle):
        """
        Verify QWOA achieves the theoretical Grover success probability.

        QWOA with a complete graph circulant mixer implements Grover's
        algorithm and should match theoretical predictions.
        """
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        full_probs = gather_state_probabilities(alg, mpi_comm)

        if mpi_comm.Get_rank() == 0:
            marked_prob = oracle.compute_marked_probability(full_probs)
            theoretical = oracle.theoretical_success_probability(1)

            assert abs(marked_prob - theoretical) < 0.01, (
                f"QWOA should match Grover theory: "
                f"got {marked_prob:.4f}, expected {theoretical:.4f}"
            )

        alg.destroy()

    def test_qaoa_qwoa_equivalence_with_complete_graph(self, mpi_comm, simple_oracle):
        """
        Verify QAOA (with complete graph) and QWOA produce equivalent results.

        When QAOA uses a complete graph mixer, both algorithms implement
        the same quantum walk and should achieve similar probability
        concentration on marked states.

        Note: Minor numerical differences exist between sparse and circulant
        implementations, so we compare marked probability rather than
        element-wise equality.
        """
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle
        params = oracle.optimal_params(depth=1)

        # Run QAOA with complete graph (built via create_qaoa_complete_graph)
        alg_qaoa = create_qaoa_complete_graph(oracle.system_size, mpi_comm, oracle)
        alg_qaoa.set_depth(1)
        alg_qaoa.setup()
        alg_qaoa.evolve_state(params)
        probs_qaoa = gather_state_probabilities(alg_qaoa, mpi_comm)

        # Run QWOA
        alg_qwoa = QWOA(oracle.system_size, mpi_comm)
        alg_qwoa.set_qualities(oracle.qualities_function())
        alg_qwoa.set_depth(1)
        alg_qwoa.setup()
        alg_qwoa.evolve_state(params)
        probs_qwoa = gather_state_probabilities(alg_qwoa, mpi_comm)

        if mpi_comm.Get_rank() == 0:
            # Compare marked state probabilities (the meaningful quantity)
            qaoa_marked = oracle.compute_marked_probability(probs_qaoa)
            qwoa_marked = oracle.compute_marked_probability(probs_qwoa)

            assert abs(qaoa_marked - qwoa_marked) < 0.001, (
                f"QAOA and QWOA should achieve same marked probability: "
                f"QAOA={qaoa_marked:.6f}, QWOA={qwoa_marked:.6f}"
            )

            # Also verify both match theoretical prediction
            theoretical = oracle.theoretical_success_probability(1)
            assert (
                abs(qaoa_marked - theoretical) < 0.01
            ), f"Both should match theory ({theoretical:.4f})"

        alg_qaoa.destroy()
        alg_qwoa.destroy()

    def test_optimal_params_concentrate_probability(self, mpi_comm, single_solution_oracle):
        """
        Verify optimal parameters concentrate probability on solution states.

        With analytically derived optimal parameters, the probability should
        concentrate on the marked (solution) states.
        """
        oracle = single_solution_oracle
        depth = oracle.optimal_iterations

        alg = create_qaoa_complete_graph(oracle.system_size, mpi_comm, oracle)
        alg.set_depth(depth)
        alg.setup()

        params = oracle.optimal_params(depth)
        alg.evolve_state(params)

        full_probs = gather_state_probabilities(alg, mpi_comm)

        if mpi_comm.Get_rank() == 0:
            marked_prob = oracle.compute_marked_probability(full_probs)
            theoretical = oracle.theoretical_success_probability(depth)

            # Should match theoretical Grover probability
            assert abs(marked_prob - theoretical) < 0.02, (
                f"Optimal params should match theory: "
                f"got {marked_prob:.4f}, expected {theoretical:.4f}"
            )

        alg.destroy()

    def test_multiple_solutions_share_probability(self, mpi_comm, simple_oracle):
        """
        Verify probability is distributed among multiple solution states.

        When there are multiple marked states, probability should be
        roughly equal among them (by symmetry of Grover's algorithm).
        """
        oracle = simple_oracle  # Has 4 marked states
        depth = oracle.optimal_iterations

        alg = create_qaoa_complete_graph(oracle.system_size, mpi_comm, oracle)
        alg.set_depth(depth)
        alg.setup()

        params = oracle.optimal_params(depth)
        alg.evolve_state(params)

        full_probs = gather_state_probabilities(alg, mpi_comm)

        if mpi_comm.Get_rank() == 0:
            marked_probs = [full_probs[i] for i in oracle.marked_states]

            # All marked states should have similar probability
            mean_marked = np.mean(marked_probs)
            max_deviation = max(abs(p - mean_marked) for p in marked_probs)

            assert (
                max_deviation < 0.1 * mean_marked
            ), f"Marked states should have similar probabilities (max dev: {max_deviation:.4f})"

        alg.destroy()

    def test_increasing_depth_improves_concentration(self, mpi_comm, mpi_sizing):
        """
        Verify deeper circuits can achieve better probability concentration.

        For problems with unique solutions, more layers generally allow
        better optimization (up to the optimal depth).
        """
        oracle = TestOracle(
            system_size=_scaled_power_of_two_system_size(mpi_sizing, base=64),
            n_marked=1,
            seed=456,
        )

        concentrations = []

        for depth in [1, 2, 3]:
            alg = create_qaoa_complete_graph(oracle.system_size, mpi_comm, oracle)
            alg.set_depth(depth)
            alg.setup()

            params = oracle.optimal_params(depth)
            alg.evolve_state(params)

            full_probs = gather_state_probabilities(alg, mpi_comm)

            if mpi_comm.Get_rank() == 0:
                marked_prob = oracle.compute_marked_probability(full_probs)
                concentrations.append(marked_prob)

            alg.destroy()

        if mpi_comm.Get_rank() == 0:
            # Each depth should match theoretical Grover probability
            for i, depth in enumerate([1, 2, 3]):
                theoretical = oracle.theoretical_success_probability(depth)
                assert (
                    abs(concentrations[i] - theoretical) < 0.02
                ), f"Depth {depth}: got {concentrations[i]:.4f}, expected {theoretical:.4f}"


@pytest.mark.mpi
class TestPostExecuteInspection:
    """Verify state inspection after execute() completes."""

    def test_post_execute_state_inspection(self, mpi_comm, simple_oracle):
        """After execute(), the final state, probabilities, and expectation
        value should all be accessible and consistent."""
        from quop_mpi.algorithm.combinatorial import QAOA

        with QAOA(simple_oracle.system_size, mpi_comm) as alg:
            alg.set_qualities(simple_oracle.qualities_function())
            alg.set_depth(1)

            alg.execute()

            # Final state should be populated
            assert alg.final_state is not None
            assert len(alg.final_state) > 0

            # Probabilities should be valid
            local_probs = np.abs(alg.final_state) ** 2
            assert np.all(local_probs >= 0)

            # Expectation value should be finite
            exp_val = alg.get_expectation_value()
            assert isinstance(exp_val, float)
            assert np.isfinite(exp_val)

            # Result should be available on rank 0
            if mpi_comm.Get_rank() == 0:
                assert alg.result is not None
                assert np.isfinite(alg.result["fun"])
