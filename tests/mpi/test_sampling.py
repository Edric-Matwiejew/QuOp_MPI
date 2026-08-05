"""
Tests for Ansatz sampling functionality: set_sampling() and unset_sampling().

Sampling allows simulating measurement-based objective function evaluation,
which is more realistic for actual quantum hardware where we can't directly
access the full wavefunction.

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_sampling.py -v --with-mpi
"""

import numpy as np
import pytest
from mpi4py import MPI

from tests.conftest import TestOracle


def _scaled_power_of_two_system_size(mpi_sizing, base):
    """Choose a power-of-two size that keeps sampling tests multi-rank aware."""
    return mpi_sizing.power_of_two(base=base, min_per_rank=1, min_per_node=16)


def _marked_count_from_ratio(system_size, denominator, minimum):
    """Preserve the original marked-state density while allowing larger systems."""
    return max(minimum, system_size // denominator)


@pytest.fixture
def simple_oracle(mpi_sizing):
    """Scale the sampling oracle while preserving M/N = 1/16."""
    system_size = _scaled_power_of_two_system_size(mpi_sizing, base=64)
    return TestOracle(
        system_size=system_size,
        n_marked=_marked_count_from_ratio(system_size, denominator=16, minimum=4),
        seed=42,
    )


@pytest.mark.mpi
class TestSetSamplingBasic:
    """Basic tests for set_sampling() method."""

    def test_set_sampling_sets_flag(self, mpi_comm, simple_oracle):
        """Verify set_sampling() sets the setup_sampling flag."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Before set_sampling
        assert not alg.setup_sampling

        alg.set_sampling(sample_block_size=10)

        # After set_sampling
        assert alg.setup_sampling

        alg.destroy()

    def test_set_sampling_stores_block_size(self, mpi_comm, simple_oracle):
        """Verify set_sampling() stores the sample_block_size."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        block_size = 50
        alg.set_sampling(sample_block_size=block_size)

        assert alg.sample_block_size == block_size

        alg.destroy()

    def test_set_sampling_stores_max_iterations(self, mpi_comm, simple_oracle):
        """Verify set_sampling() stores max_sample_iterations."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        max_iters = 50
        alg.set_sampling(sample_block_size=10, max_sample_iterations=max_iters)

        assert alg.max_sample_iterations == max_iters

        alg.destroy()

    def test_set_sampling_default_max_iterations(self, mpi_comm, simple_oracle):
        """Verify default max_sample_iterations is 100."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        alg.set_sampling(sample_block_size=10)

        assert alg.max_sample_iterations == 100

        alg.destroy()

    def test_set_sampling_stores_function(self, mpi_comm, simple_oracle):
        """Verify set_sampling() stores the sampling function."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        def custom_sampler(samples):
            return np.mean(samples), True

        alg.set_sampling(sample_block_size=10, function=custom_sampler)

        assert alg.sampling_function_input is custom_sampler

        alg.destroy()

    def test_set_sampling_default_function(self, mpi_comm, simple_oracle):
        """Verify default sampling function uses mean."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        alg.set_sampling(sample_block_size=10)

        # Default function should exist
        assert alg.sampling_function_input is not None

        # Test that default function computes mean and returns True
        test_samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        result, accept = alg.sampling_function_input(test_samples)
        assert result == 3.0  # mean of 1-5
        assert accept

        alg.destroy()


@pytest.mark.mpi
class TestUnsetSampling:
    """Tests for unset_sampling() method."""

    def test_unset_sampling_clears_flag(self, mpi_comm, simple_oracle):
        """Verify unset_sampling() clears the sampling flags."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        alg.set_sampling(sample_block_size=10)
        assert alg.setup_sampling

        alg.unset_sampling()

        assert not alg.setup_sampling
        assert not alg.sampling

        alg.destroy()

    def test_unset_sampling_removes_pre_execution_method(self, mpi_comm, simple_oracle):
        """Verify unset_sampling() removes pre_execution method."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        initial_pre_methods = len(alg.pre_execution_methods)

        alg.set_sampling(sample_block_size=10)
        assert len(alg.pre_execution_methods) == initial_pre_methods + 1

        alg.unset_sampling()
        assert len(alg.pre_execution_methods) == initial_pre_methods

        alg.destroy()


@pytest.mark.mpi
class TestSamplingExecution:
    """Tests for sampling during execute()."""

    def test_execute_with_sampling_completes(self, mpi_comm, simple_oracle):
        """Verify execute() completes with sampling enabled."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=10, max_sample_iterations=5)

        # Should complete without error
        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_execute_with_sampling_produces_result(self, mpi_comm, simple_oracle):
        """Verify execute() with sampling produces valid optimization result."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=20, max_sample_iterations=10)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert "x" in alg.result
            assert "fun" in alg.result
            assert np.isfinite(alg.result["fun"])

        alg.destroy()

    def test_sampling_tracks_total_shots(self, mpi_comm, simple_oracle):
        """Verify sampling tracks total shots taken."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=10, max_sample_iterations=5)

        alg.execute()

        # total_shots should be > 0 after execution
        # (on rank 0 of the subcomm)
        if alg.subcomms.in_rootcomm():
            assert alg.total_shots > 0

        alg.destroy()

    def test_sampling_finds_minimum(self, mpi_comm, simple_oracle):
        """Verify sampling tracks minimum sampled value."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=50, max_sample_iterations=10)

        alg.execute()

        # minimum_sampled should be updated
        if alg.subcomms.in_rootcomm():
            assert alg.minimum_sampled != np.inf
            # For Grover oracle, minimum should be 0 or close to it
            assert alg.minimum_sampled >= 0.0

        alg.destroy()


@pytest.mark.mpi
class TestSamplingResults:
    """Tests for sampling result attributes."""

    def test_quop_result_contains_sampling_info(self, mpi_comm, simple_oracle):
        """Verify quop_result contains sampling statistics after execute."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=20, max_sample_iterations=5)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert "sampling total shots" in alg.quop_result
            assert "sampling minimum measured" in alg.quop_result
            assert "sampling shots to minimum measured" in alg.quop_result

        alg.destroy()

    def test_global_minimum_tracked(self, mpi_comm, simple_oracle):
        """Verify global_minimum is computed correctly."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=10, max_sample_iterations=5)

        alg.execute()

        # global_minimum should be the minimum observable value
        # For Grover oracle, this is 0 (marked states)
        if alg.subcomms.in_rootcomm():
            assert alg.global_minimum is not None
            assert alg.global_minimum == 0.0  # Grover oracle minimum

        alg.destroy()


@pytest.mark.mpi
class TestSamplingWithQWOA:
    """Tests for sampling with QWOA algorithm."""

    def test_qwoa_with_sampling_completes(self, mpi_comm, simple_oracle):
        """Verify QWOA with sampling completes."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=10, max_sample_iterations=5)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_qwoa_sampling_tracks_shots(self, mpi_comm, simple_oracle):
        """Verify QWOA sampling tracks total shots."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=20, max_sample_iterations=5)

        alg.execute()

        if alg.subcomms.in_rootcomm():
            assert alg.total_shots > 0

        alg.destroy()


@pytest.mark.mpi
class TestCustomSamplingFunction:
    """Tests for custom sampling functions."""

    def test_custom_sampling_function_called(self, mpi_comm, simple_oracle):
        """Verify custom sampling function is called during execution."""
        from quop_mpi.algorithm.combinatorial import QAOA

        # Track if function was called
        call_count = [0]

        def counting_sampler(samples):
            call_count[0] += 1
            return np.mean(samples), True

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=10, function=counting_sampler, max_sample_iterations=3)

        alg.execute()

        # Synchronize call count across ranks
        total_calls = mpi_comm.reduce(call_count[0], op=MPI.SUM, root=0)

        if mpi_comm.Get_rank() == 0:
            # Function should have been called at least once
            assert total_calls > 0

        alg.destroy()

    def test_custom_sampling_function_can_reject(self, mpi_comm, simple_oracle):
        """Verify custom function returning False continues sampling."""
        from quop_mpi.algorithm.combinatorial import QAOA

        iteration_count = [0]

        def always_reject(samples):
            iteration_count[0] += 1
            # Return False to continue sampling up to max_iterations
            return np.mean(samples), False

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        max_iters = 5
        alg.set_sampling(
            sample_block_size=10,
            function=always_reject,
            max_sample_iterations=max_iters,
        )

        alg.prepare()
        alg.evaluate(np.array([0.1, 0.2]))

        total_iterations = mpi_comm.reduce(iteration_count[0], op=MPI.SUM, root=0)

        # The function should be called max_iterations times for the single
        # sampled objective evaluation, since it always returns False.
        if mpi_comm.Get_rank() == 0:
            assert total_iterations == max_iters
            assert alg.total_shots == alg.sample_block_size * max_iters

        alg.destroy()

    def test_custom_sampling_function_early_accept(self, mpi_comm, simple_oracle):
        """Verify custom function returning True stops sampling early."""
        from quop_mpi.algorithm.combinatorial import QAOA

        iteration_count = [0]

        def always_accept(samples):
            iteration_count[0] += 1
            return np.mean(samples), True

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=10, function=always_accept, max_sample_iterations=100)

        alg.prepare()
        alg.evaluate(np.array([0.1, 0.2]))

        total_iterations = mpi_comm.reduce(iteration_count[0], op=MPI.SUM, root=0)

        if mpi_comm.Get_rank() == 0:
            assert total_iterations == 1
            assert alg.total_shots == alg.sample_block_size

        alg.destroy()


@pytest.mark.mpi
class TestSamplingVsExact:
    """Tests comparing sampling to exact computation."""

    def test_sampling_expectation_in_quality_range(self, mpi_comm, simple_oracle):
        """Verify sampled expectation is within valid quality range."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=100, max_sample_iterations=10)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            result_value = alg.result["fun"]
            # For Grover oracle, quality range is [0, 1]
            assert 0.0 <= result_value <= 1.0

        alg.destroy()

    def test_sampling_vs_exact_converge(self, mpi_comm, simple_oracle):
        """Verify sampling with many shots approaches exact computation."""
        from quop_mpi.algorithm.combinatorial import QAOA

        # First, get exact result
        alg_exact = QAOA(simple_oracle.system_size, mpi_comm)
        alg_exact.set_qualities(simple_oracle.qualities_function())
        alg_exact.set_depth(1)
        alg_exact.setup()
        alg_exact._Ansatz__pre()

        params = simple_oracle.optimal_params(depth=1)
        exact_result = alg_exact.evaluate(params)
        alg_exact.destroy()

        # Now with sampling (large block size for accuracy)
        alg_sampled = QAOA(simple_oracle.system_size, mpi_comm)
        alg_sampled.set_qualities(simple_oracle.qualities_function())
        alg_sampled.set_depth(1)
        alg_sampled.set_sampling(sample_block_size=500, max_sample_iterations=1)
        alg_sampled.setup()
        alg_sampled._Ansatz__pre()

        # evolve_state and get sampled expectation
        alg_sampled.evolve_state(params)
        sampled_result = alg_sampled.get_expectation_value()

        # With large sample size, results should be reasonably close
        if alg_sampled.subcomms.in_rootcomm():
            # Allow for sampling variance (within 0.2 of exact)
            assert (
                abs(sampled_result - exact_result) < 0.2
            ), f"Sampled {sampled_result} too far from exact {exact_result}"

        alg_sampled.destroy()


@pytest.mark.mpi
class TestSamplingEdgeCases:
    """Tests for edge cases in sampling."""

    def test_small_block_size(self, mpi_comm, simple_oracle):
        """Test sampling with very small block size."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=1, max_sample_iterations=10)

        # Should complete without error
        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_large_block_size(self, mpi_comm, simple_oracle):
        """Test sampling with block size larger than system size."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        # Block size > system_size
        alg.set_sampling(sample_block_size=simple_oracle.system_size * 2, max_sample_iterations=3)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_single_iteration(self, mpi_comm, simple_oracle):
        """Test sampling with max_sample_iterations=1."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.set_sampling(sample_block_size=50, max_sample_iterations=1)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_toggle_sampling_on_off(self, mpi_comm, simple_oracle):
        """Test toggling sampling on and off."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Enable sampling
        alg.set_sampling(sample_block_size=10)
        assert alg.setup_sampling

        # Disable sampling
        alg.unset_sampling()
        assert not alg.setup_sampling

        # Re-enable sampling
        alg.set_sampling(sample_block_size=20)
        assert alg.setup_sampling
        assert alg.sample_block_size == 20

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_execute_after_unset_sampling(self, mpi_comm, simple_oracle):
        """Test execute() uses exact computation after unset_sampling()."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Enable then disable sampling
        alg.set_sampling(sample_block_size=10)
        alg.unset_sampling()

        # Execute should use exact computation
        alg.execute()

        # Should not have sampling statistics
        assert not alg.sampling

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()


@pytest.mark.mpi
class TestSamplingCombinations:
    """Tests for sampling combined with manual evolution and parameter map."""

    def test_sampling_after_evolve_state(self, mpi_comm, simple_oracle):
        """Verify evolve_state + get_expectation_value works with sampling enabled."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_sampling(sample_block_size=50)
        alg.set_depth(1)

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        result = alg.get_expectation_value()

        if alg.subcomms.get_subcomm_index() == 0:
            assert result is not None
            assert isinstance(result, (float, np.floating))
            assert np.isfinite(result)

        alg.destroy()

    def test_sampling_with_param_map_execute(self, mpi_comm, simple_oracle):
        """Verify execute() works with both sampling and parameter map."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_sampling(sample_block_size=50)

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        alg.set_parameter_map(2, parameter_map)
        alg.set_depth(1)

        initial_params = np.array([np.pi, oracle.optimal_walk_time])
        alg.execute(initial_params)

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert len(alg.result["x"]) == 2
            assert np.isfinite(alg.result["fun"])

        alg.destroy()
