"""
Tests for the optimization process (execute method).

These tests verify that the Ansatz.execute() method correctly performs
variational optimization to find parameters that minimize the objective.

Uses test oracles with analytically known solutions to verify correctness.

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_optimization.py -v --with-mpi
"""

import numpy as np
import pytest

from tests.conftest import TestOracle


@pytest.mark.mpi
class TestExecuteBasic:
    """Test basic execute() functionality."""

    def test_execute_completes_without_error(self, mpi_comm, simple_oracle):
        """Verify execute() runs to completion."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Should complete without error
        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None, "Result should be populated after execute()"

        alg.destroy()

    def test_execute_returns_result_dictionary(self, mpi_comm, simple_oracle):
        """Verify execute() returns properly structured result."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            # Result should have standard optimization fields
            assert "fun" in alg.result, "Result should contain objective value 'fun'"
            assert "x" in alg.result, "Result should contain parameters 'x'"

        alg.destroy()

    def test_execute_respects_depth(self, mpi_comm, simple_oracle):
        """Verify execute() uses the configured depth."""
        from quop_mpi.algorithm.combinatorial import QAOA

        for depth in [1, 2]:
            alg = QAOA(simple_oracle.system_size, mpi_comm)
            alg.set_qualities(simple_oracle.qualities_function())
            alg.set_depth(depth)

            alg.execute()

            if mpi_comm.Get_rank() == 0:
                # Number of parameters should match depth
                # 2 params per layer (phase, mixing)
                expected_n_params = 2 * depth
                assert (
                    len(alg.result["x"]) == expected_n_params
                ), f"Depth {depth} should have {expected_n_params} params"

            alg.destroy()


@pytest.mark.mpi
class TestOptimizationQuality:
    """Test that optimization finds good solutions."""

    def test_optimization_beats_random(self, mpi_comm, simple_oracle):
        """Verify optimization achieves better expectation than random/uniform."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(2)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            optimized_expectation = alg.result["fun"]
            uniform_expectation = simple_oracle.uniform_expectation()

            assert (
                optimized_expectation < uniform_expectation
            ), (
                f"Optimized ({optimized_expectation:.4f}) "
                f"should beat uniform ({uniform_expectation:.4f})"
            )

        alg.destroy()

    def test_optimization_finds_low_expectation(self, mpi_comm):
        """Verify optimization can find solutions with low objective value."""
        from quop_mpi.algorithm.combinatorial import QAOA

        # Use oracle with multiple solutions (easier to optimize)
        oracle = TestOracle(system_size=64, n_marked=8, seed=789)

        alg = QAOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(3)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            final_expectation = alg.result["fun"]

            # Should achieve expectation well below uniform
            uniform = oracle.uniform_expectation()
            improvement = (uniform - final_expectation) / uniform

            assert (
                improvement > 0.1
            ), f"Should improve at least 10% over uniform (got {improvement*100:.1f}%)"

        alg.destroy()

    def test_deeper_circuit_can_achieve_better_result(self, mpi_comm):
        """Verify increasing depth allows better optimization."""
        from quop_mpi.algorithm.combinatorial import QAOA

        oracle = TestOracle(system_size=64, n_marked=4, seed=999)

        results = {}

        for depth in [1, 3]:
            alg = QAOA(oracle.system_size, mpi_comm)
            alg.set_qualities(oracle.qualities_function())
            alg.set_depth(depth)

            alg.execute()

            if mpi_comm.Get_rank() == 0:
                results[depth] = alg.result["fun"]

            alg.destroy()

        if mpi_comm.Get_rank() == 0:
            # Deeper circuit should achieve at least as good (usually better) result
            assert (
                results[3] <= results[1] + 0.05
            ), f"Depth 3 ({results[3]:.4f}) should be no worse than depth 1 ({results[1]:.4f})"


@pytest.mark.mpi
class TestOptimizationConvergence:
    """Test optimizer convergence behavior."""

    def test_near_optimal_start_converges_quickly(self, mpi_comm, single_solution_oracle):
        """Verify starting near optimum doesn't diverge.

        Uses QWOA since oracle.optimal_params() are calculated for complete
        graph mixing (Grover-like), which matches QWOA's default mixer.
        """
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = single_solution_oracle
        depth = 2

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(depth)

        # Start from near-optimal parameters (small perturbation)
        optimal = oracle.optimal_params(depth)
        rng = np.random.default_rng(42)
        perturbed = optimal + 0.1 * rng.uniform(-1, 1, size=len(optimal))

        # Pass initial parameters to execute()
        alg.execute(perturbed)

        if mpi_comm.Get_rank() == 0:
            final_expectation = alg.result["fun"]
            uniform = oracle.uniform_expectation()

            # Starting near optimum should achieve much better than uniform
            assert (
                final_expectation < uniform * 0.8
            ), (
                f"Near-optimal start should beat uniform ({uniform:.4f}) "
                f"significantly (got {final_expectation:.4f})"
            )

        alg.destroy()

    def test_multiple_executions_give_consistent_results(self, mpi_comm):
        """Verify repeated optimization gives similar results."""
        from quop_mpi.algorithm.combinatorial import QAOA

        oracle = TestOracle(system_size=32, n_marked=4, seed=111)

        expectations = []

        for _trial in range(3):
            alg = QAOA(oracle.system_size, mpi_comm)
            alg.set_qualities(oracle.qualities_function())
            alg.set_depth(2)

            alg.execute()

            if mpi_comm.Get_rank() == 0:
                expectations.append(alg.result["fun"])

            alg.destroy()

        if mpi_comm.Get_rank() == 0:
            # Results should be reasonably consistent
            # (may vary due to random initialization)
            assert (
                max(expectations) - min(expectations) < 0.3
            ), f"Results should be consistent: {expectations}"


@pytest.mark.mpi
class TestOptimizationWithDifferentAlgorithms:
    """Test optimization works for different algorithm classes."""

    def test_qaoa_optimization(self, mpi_comm, simple_oracle):
        """Verify QAOA optimization works correctly."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(2)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result["fun"] < simple_oracle.uniform_expectation()

        alg.destroy()

    def test_qwoa_optimization(self, mpi_comm, simple_oracle):
        """Verify QWOA optimization works correctly."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(2)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result["fun"] < simple_oracle.uniform_expectation()

        alg.destroy()
