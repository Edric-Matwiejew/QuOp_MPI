"""
Tests for Ansatz evaluation methods: evaluate(), objective(), and get_expectation_value().

These methods are core to how the optimization works:
- evaluate(): Lazily computes objective function with caching
- objective(): Direct objective function computation (used by optimizers)
- get_expectation_value(): Computes expectation value of current state

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_evaluation.py -v --with-mpi
"""

import numpy as np
import pytest

from tests.conftest import TestOracle


def _scaled_power_of_two_system_size(mpi_sizing, base):
    """Choose a power-of-two size that keeps evaluation tests multi-rank aware."""
    return mpi_sizing.power_of_two(base=base, min_per_rank=1, min_per_node=16)


def _marked_count_from_ratio(system_size, denominator, minimum):
    """Preserve the original marked-state density while allowing larger systems."""
    return max(minimum, system_size // denominator)


@pytest.fixture
def simple_oracle(mpi_sizing):
    """Scale the shared evaluation oracle while preserving M/N = 1/16."""
    system_size = _scaled_power_of_two_system_size(mpi_sizing, base=64)
    return TestOracle(
        system_size=system_size,
        n_marked=_marked_count_from_ratio(system_size, denominator=16, minimum=4),
        seed=42,
    )


@pytest.mark.mpi
class TestEvaluateMethod:
    """Tests for the evaluate() method - lazy objective function computation."""

    def test_evaluate_returns_float(self, mpi_comm, simple_oracle):
        """Verify evaluate() returns a float value."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        params = simple_oracle.optimal_params(depth=1)

        result = alg.evaluate(params)

        if alg.subcomms.in_subcomm():
            assert isinstance(
                result, (float, np.floating)
            ), f"evaluate() should return float, got {type(result)}"
        else:
            assert result is None

        alg.destroy()

    def test_evaluate_produces_consistent_results(self, mpi_comm, simple_oracle):
        """Verify evaluate() produces same result for same parameters."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        params = simple_oracle.optimal_params(depth=1)

        result1 = alg.evaluate(params)
        result2 = alg.evaluate(params)

        if alg.subcomms.in_subcomm():
            assert np.isclose(
                result1, result2
            ), f"evaluate() should be deterministic: {result1} vs {result2}"

        alg.destroy()

    def test_evaluate_caches_result(self, mpi_comm, simple_oracle):
        """Verify evaluate() caches results and doesn't recompute for same params."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        params = simple_oracle.optimal_params(depth=1)

        # First call - should compute
        alg.evaluate(params)

        if alg.subcomms.in_subcomm():
            # Check that last_evaluated is set
            assert alg.last_evaluated is not None, "last_evaluated should be set after evaluate()"
            assert np.array_equal(
                alg.last_evaluated, params
            ), "last_evaluated should match input parameters"

        alg.destroy()

    def test_evaluate_different_params_give_different_results(self, mpi_comm, simple_oracle):
        """Verify evaluate() gives different results for different parameters."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        params1 = simple_oracle.optimal_params(depth=1)
        params2 = np.zeros_like(params1)  # Different params

        result1 = alg.evaluate(params1)
        result2 = alg.evaluate(params2)

        if alg.subcomms.in_subcomm():
            # Different parameters should give different results
            assert not np.isclose(
                result1, result2
            ), "Different parameters should give different results"

        alg.destroy()

    def test_evaluate_works_with_list_input(self, mpi_comm, simple_oracle):
        """Verify evaluate() accepts list input."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        params = simple_oracle.optimal_params(depth=1).tolist()  # Convert to list

        result = alg.evaluate(params)

        if alg.subcomms.in_subcomm():
            assert isinstance(result, (float, np.floating))

        alg.destroy()

    def test_evaluate_value_in_expected_range(self, mpi_comm, simple_oracle):
        """Verify evaluate() returns values in a reasonable range for our test oracle."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        params = simple_oracle.optimal_params(depth=1)

        result = alg.evaluate(params)

        if alg.subcomms.in_subcomm():
            # For Grover oracle: marked states = 0, unmarked = 1
            # Expectation value should be between 0 and 1
            min_quality = 0.0
            max_quality = 1.0

            assert (
                min_quality <= result <= max_quality
            ), f"Result {result} outside quality range [{min_quality}, {max_quality}]"

        alg.destroy()


@pytest.mark.mpi
class TestGetExpectationValue:
    """Tests for get_expectation_value() method."""

    def test_get_expectation_value_after_evolve(self, mpi_comm, simple_oracle):
        """Verify get_expectation_value() works after evolve_state()."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        # get_expectation_value only returns on subcomm index 0
        result = alg.get_expectation_value()

        if alg.subcomms.get_subcomm_index() == 0:
            assert result is not None, "get_expectation_value() should return a value"
            assert isinstance(result, (float, np.floating))

        alg.destroy()

    def test_get_expectation_value_consistent_with_state_norm(self, mpi_comm, simple_oracle):
        """Verify expectation value is consistent with a normalized state."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        # get_state_norm() is collective over SUBCOMM and returns the norm
        # on all active ranks.  Excluded ranks receive None.
        alg.get_expectation_value()
        computed_norm = alg.get_state_norm()

        if alg.subcomms.in_subcomm():
            # Active ranks: state_norm should be set and equal to 1.0 (normalized)
            assert (
                computed_norm is not None
            ), "get_state_norm() should return a value on active subcomm ranks"
            assert np.isclose(
                float(computed_norm), 1.0, atol=1e-10
            ), f"State should be normalized, got norm {computed_norm}"
        else:
            # Excluded ranks: get_state_norm() should return None
            assert (
                computed_norm is None
            ), f"get_state_norm() should return None on ranks outside of subcomm, got {computed_norm}"

        alg.destroy()

    def test_get_expectation_value_in_quality_range(self, mpi_comm, simple_oracle):
        """Verify expectation value is within the range of qualities."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        result = alg.get_expectation_value()

        if alg.subcomms.get_subcomm_index() == 0:
            # For Grover oracle: marked states = 0, unmarked = 1
            min_quality = 0.0
            max_quality = 1.0

            assert (
                min_quality <= result <= max_quality
            ), f"Expectation {result} outside quality range [{min_quality}, {max_quality}]"

        alg.destroy()


@pytest.mark.mpi
class TestEvaluationCorrectness:
    """Tests for correctness of evaluation methods."""

    def test_optimal_params_beat_random_baseline(self, mpi_comm, simple_oracle):
        """Verify known-good parameters beat a deterministic random baseline."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(2)  # More depth for better optimization
        alg.prepare()

        optimal_params = simple_oracle.optimal_params(depth=2)

        # Use a fixed batch of random candidates so the comparison is reproducible
        # across ranks and across test runs.
        if mpi_comm.Get_rank() == 0:
            rng = np.random.default_rng(12345)
            random_params = rng.uniform(0, np.pi, size=(5, optimal_params.size))
        else:
            random_params = np.empty((5, optimal_params.size))
        mpi_comm.Bcast(random_params, root=0)

        optimal_result = alg.evaluate(optimal_params)
        random_results = [alg.evaluate(candidate) for candidate in random_params]

        if alg.subcomms.in_subcomm():
            assert np.isfinite(optimal_result)
            assert np.all(np.isfinite(random_results))
            assert optimal_result < simple_oracle.uniform_expectation()
            assert optimal_result < float(np.mean(random_results))

        alg.destroy()

    def test_zero_params_give_uniform_expectation(self, mpi_comm, simple_oracle):
        """Verify zero parameters give expectation equal to mean quality."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        zero_params = np.zeros(2)  # gamma=0, beta=0

        result = alg.evaluate(zero_params)

        if alg.subcomms.in_subcomm():
            # With zero params, state should be uniform
            # Use oracle's uniform_expectation method: E = (N-M)/N
            mean_quality = simple_oracle.uniform_expectation()

            assert np.isclose(
                result, mean_quality, rtol=1e-5
            ), f"Zero params should give mean quality {mean_quality}, got {result}"

        alg.destroy()

    def test_evaluate_matches_execute_result(self, mpi_comm, simple_oracle):
        """Verify evaluate() on optimized params matches execute() result."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            optimized_params = alg.result["x"]
            execute_result = alg.result["fun"]
        else:
            optimized_params = None
            execute_result = None

        # Broadcast to all ranks
        optimized_params = mpi_comm.bcast(optimized_params, root=0)
        execute_result = mpi_comm.bcast(execute_result, root=0)

        # Now evaluate with those params
        eval_result = alg.evaluate(optimized_params)

        if alg.subcomms.in_subcomm():
            # Results should match
            assert np.isclose(
                eval_result, execute_result, rtol=1e-5
            ), f"evaluate() {eval_result} should match execute() result {execute_result}"

        alg.destroy()


@pytest.mark.mpi
class TestEvaluationWithQWOA:
    """Tests for evaluation methods with QWOA algorithm."""

    def test_qwoa_evaluate_returns_float(self, mpi_comm, simple_oracle):
        """Verify evaluate() works with QWOA."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        params = simple_oracle.optimal_params(depth=1)

        result = alg.evaluate(params)

        if alg.subcomms.in_subcomm():
            assert isinstance(result, (float, np.floating))
        else:
            assert result is None

        alg.destroy()

    def test_qwoa_get_expectation_value_after_evolve(self, mpi_comm, simple_oracle):
        """Verify get_expectation_value() works with QWOA after evolution."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        result = alg.get_expectation_value()

        if alg.subcomms.get_subcomm_index() == 0:
            assert result is not None

            # For Grover oracle: marked = 0, unmarked = 1
            min_quality = 0.0
            max_quality = 1.0
            assert min_quality <= result <= max_quality

        alg.destroy()


@pytest.mark.mpi
class TestEvaluationEdgeCases:
    """Tests for edge cases in evaluation methods."""

    def test_evaluate_with_very_large_params(self, mpi_comm, simple_oracle):
        """Test evaluate() with very large parameter values."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        large_params = np.array([1e10, 1e10])

        result = alg.evaluate(large_params)

        if alg.subcomms.in_subcomm():
            # Due to periodicity of exp(i*theta), should still be in range
            min_quality = 0.0
            max_quality = 1.0
            assert min_quality <= result <= max_quality

        alg.destroy()

    def test_repeated_evaluate_calls_efficient(self, mpi_comm, simple_oracle):
        """Test that repeated evaluate() calls with same params use caching."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.prepare()

        params = simple_oracle.optimal_params(depth=1)

        # First call - computes
        result1 = alg.evaluate(params)

        # Second call with same params - should use cache
        result2 = alg.evaluate(params)

        if alg.subcomms.in_subcomm():
            # Results should be identical
            assert result1 == result2

        alg.destroy()
