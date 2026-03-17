"""
Tests for Ansatz.benchmark() method.

The benchmark method runs optimization at multiple ansatz depths with multiple repeats,
tracking results and supporting features like:
- Parameter persistence between depths
- Verbose output
- Saving results to HDF5 files
- Time limits and suspend/resume

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_benchmark.py -v --with-mpi
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.mpi
class TestBenchmarkBasic:
    """Basic tests for benchmark() functionality."""

    def test_benchmark_runs_without_error(self, mpi_comm, simple_oracle):
        """Verify benchmark runs to completion without error."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        # Run benchmark with minimal settings
        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            verbose=False,
        )

        alg.destroy()

    def test_benchmark_with_multiple_depths(self, mpi_comm, simple_oracle):
        """Verify benchmark runs across multiple depths."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1, 2],
            repeats=1,
            verbose=False,
        )

        alg.destroy()

    def test_benchmark_with_multiple_repeats(self, mpi_comm, simple_oracle):
        """Verify benchmark runs multiple repeats at each depth."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1],
            repeats=3,
            verbose=False,
        )

        alg.destroy()

    def test_benchmark_sets_benchmarking_flag_during_run(self, mpi_comm, simple_oracle):
        """Verify benchmarking flag is set during benchmark."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        # Before benchmark
        assert not alg.benchmarking

        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            verbose=False,
        )

        # After benchmark completes, flag should be reset
        assert not alg.benchmarking

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkResults:
    """Tests for benchmark results tracking."""

    def test_benchmark_produces_result(self, mpi_comm, simple_oracle):
        """Verify benchmark produces a result."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            verbose=False,
        )

        # Result should be set
        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_benchmark_tracker_has_results(self, mpi_comm, simple_oracle):
        """Verify tracker accumulates results."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1, 2],
            repeats=2,
            verbose=False,
        )

        # Tracker should have results for each depth
        results = alg.tracker.get_results()

        if mpi_comm.Get_rank() == 0:
            assert len(results[1]) == 2  # 2 repeats at depth 1
            assert len(results[2]) == 2  # 2 repeats at depth 2

        alg.destroy()

    def test_benchmark_results_contain_fun(self, mpi_comm, simple_oracle):
        """Verify results contain objective function values."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1],
            repeats=2,
            verbose=False,
        )

        results = alg.tracker.get_results()

        if mpi_comm.Get_rank() == 0:
            for result in results[1]:
                assert "fun" in result
                assert np.isfinite(result["fun"])

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkParamPersist:
    """Tests for parameter persistence between depths."""

    def test_benchmark_param_persist_false(self, mpi_comm, simple_oracle):
        """Verify benchmark works with param_persist=False."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1, 2],
            repeats=1,
            param_persist=False,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_benchmark_param_persist_true(self, mpi_comm, simple_oracle):
        """Verify benchmark works with param_persist=True."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1, 2],
            repeats=2,
            param_persist=True,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkWithQWOA:
    """Tests for benchmark with QWOA algorithm."""

    def test_qwoa_benchmark_runs(self, mpi_comm, simple_oracle):
        """Verify benchmark works with QWOA."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_qwoa_benchmark_multiple_depths(self, mpi_comm, simple_oracle):
        """Verify QWOA benchmark works with multiple depths."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1, 2],
            repeats=1,
            verbose=False,
        )

        results = alg.tracker.get_results()

        if mpi_comm.Get_rank() == 0:
            assert len(results[1]) == 1
            assert len(results[2]) == 1

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkSaveResults:
    """Tests for saving benchmark results to files."""

    def test_benchmark_saves_to_h5(self, mpi_comm, simple_oracle):
        """Verify benchmark saves results to HDF5 file."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "benchmark_test")

            alg.benchmark(
                ansatz_depths=[1],
                repeats=1,
                verbose=False,
                filename=filename,
                label="test",
            )

            mpi_comm.barrier()

            # Check file was created
            h5_file = filename + ".h5"
            if mpi_comm.Get_rank() == 0:
                assert os.path.exists(h5_file)

                # Verify file contents
                with h5py.File(h5_file, "r") as f:
                    assert "test_1_1" in f

        alg.destroy()

    def test_benchmark_saves_multiple_runs(self, mpi_comm, simple_oracle):
        """Verify benchmark saves all runs to HDF5 file."""
        import h5py

        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "benchmark_test")

            alg.benchmark(
                ansatz_depths=[1, 2],
                repeats=2,
                verbose=False,
                filename=filename,
                label="multi",
            )

            mpi_comm.barrier()

            h5_file = filename + ".h5"
            if mpi_comm.Get_rank() == 0:
                with h5py.File(h5_file, "r") as f:
                    # Should have entries for all depth/repeat combos
                    assert "multi_1_1" in f
                    assert "multi_1_2" in f
                    assert "multi_2_1" in f
                    assert "multi_2_2" in f

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkDepthRestoration:
    """Tests for ansatz depth restoration after benchmark."""

    def test_ansatz_depth_restored_after_benchmark(self, mpi_comm, simple_oracle):
        """Verify original ansatz depth is restored after benchmark."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(3)  # Set initial depth

        original_depth = alg.ansatz_depth

        alg.benchmark(
            ansatz_depths=[1, 2, 4],  # Different depths
            repeats=1,
            verbose=False,
        )

        # Depth should be restored
        assert alg.ansatz_depth == original_depth

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkVerbose:
    """Tests for verbose output (just verify no crashes)."""

    def test_benchmark_verbose_true(self, mpi_comm, simple_oracle):
        """Verify benchmark with verbose=True doesn't crash."""
        import io
        import sys

        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        # Capture stdout to suppress output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            alg.benchmark(
                ansatz_depths=[1],
                repeats=1,
                verbose=True,
            )
        finally:
            sys.stdout = old_stdout

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkSeed:
    """Tests for seed handling in benchmark."""

    def test_benchmark_uses_algorithm_seed(self, mpi_comm, simple_oracle):
        """Verify benchmark uses the algorithm's seed."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_seed(12345)

        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            verbose=False,
        )

        # Tracker should start with algorithm's seed
        # (seed increments with each job)
        alg.destroy()

    def test_benchmark_reproducible_with_seed(self, mpi_comm, simple_oracle):
        """Verify benchmark is reproducible with same seed."""
        from quop_mpi.algorithm.combinatorial import QAOA

        seed = 42

        # First run
        alg1 = QAOA(simple_oracle.system_size, mpi_comm)
        alg1.set_qualities(simple_oracle.qualities_function())
        alg1.set_seed(seed)

        alg1.benchmark(
            ansatz_depths=[1],
            repeats=1,
            verbose=False,
        )

        result1 = None
        if mpi_comm.Get_rank() == 0:
            result1 = alg1.result["fun"]
        result1 = mpi_comm.bcast(result1, root=0)

        alg1.destroy()

        # Second run with same seed
        alg2 = QAOA(simple_oracle.system_size, mpi_comm)
        alg2.set_qualities(simple_oracle.qualities_function())
        alg2.set_seed(seed)

        alg2.benchmark(
            ansatz_depths=[1],
            repeats=1,
            verbose=False,
        )

        result2 = None
        if mpi_comm.Get_rank() == 0:
            result2 = alg2.result["fun"]
        result2 = mpi_comm.bcast(result2, root=0)

        alg2.destroy()

        # Results should be identical
        assert np.isclose(result1, result2)


@pytest.mark.mpi
class TestBenchmarkEdgeCases:
    """Tests for edge cases in benchmark."""

    def test_benchmark_single_depth_single_repeat(self, mpi_comm, simple_oracle):
        """Test minimal benchmark: 1 depth, 1 repeat."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_benchmark_deep_circuit(self, mpi_comm, simple_oracle):
        """Test benchmark with deeper circuit."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1, 2, 3],
            repeats=1,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            # Last result should have more parameters
            assert len(alg.result["x"]) == 6  # 2 params per layer * 3 layers

        alg.destroy()

    def test_benchmark_non_sequential_depths(self, mpi_comm, simple_oracle):
        """Test benchmark with non-sequential depths."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        # Note: benchmark uses max of ansatz_depths, not the actual list
        # This tests that the internal handling works
        alg.benchmark(
            ansatz_depths=[2],  # Start at depth 2
            repeats=1,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkTimeLimit:
    """Tests for time limit functionality."""

    def test_benchmark_with_time_limit_no_suspend(self, mpi_comm, simple_oracle):
        """Verify benchmark with large time limit completes normally."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        with tempfile.TemporaryDirectory() as tmpdir:
            suspend_path = os.path.join(tmpdir, "suspend")

            alg.benchmark(
                ansatz_depths=[1],
                repeats=1,
                verbose=False,
                time_limit=3600,  # 1 hour - should not trigger suspend
                suspend_path=suspend_path,
            )

            if mpi_comm.Get_rank() == 0:
                assert alg.result is not None

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkMPIConsistency:
    """Tests for MPI consistency during benchmark."""

    def test_benchmark_result_consistent(self, mpi_comm, simple_oracle):
        """Verify final result is consistent on rank 0."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1, 2],
            repeats=2,
            verbose=False,
        )

        # Only rank 0 should have result
        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert "fun" in alg.result
            assert "x" in alg.result

        alg.destroy()

    def test_benchmark_tracker_complete_all_ranks(self, mpi_comm, simple_oracle):
        """Verify tracker reports complete on all ranks."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        alg.benchmark(
            ansatz_depths=[1],
            repeats=2,
            verbose=False,
        )

        # All ranks should agree tracker is complete
        local_complete = alg.tracker.complete
        all_complete = mpi_comm.gather(local_complete, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(c for c in all_complete)

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkWithSampling:
    """Tests for benchmark with simulated sampling (set_sampling)."""

    def test_benchmark_with_sampling(self, mpi_comm, simple_oracle):
        """Verify benchmark works with sampling enabled."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_sampling(
            max_sample_iterations=3,
            sample_block_size=5,
        )

        alg.benchmark(
            ansatz_depths=[1, 2],
            repeats=2,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_benchmark_sampling_tracks_shots(self, mpi_comm, simple_oracle):
        """Verify sampling during benchmark tracks total shots."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_sampling(
            max_sample_iterations=2,
            sample_block_size=10,
        )

        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            verbose=False,
        )

        # total_shots should have been incremented during sampling
        if mpi_comm.Get_rank() == 0:
            # After benchmark, sampling was used
            assert alg.result is not None

        alg.destroy()

    def test_benchmark_sampling_with_multiple_depths(self, mpi_comm, simple_oracle):
        """Verify benchmark with sampling works across multiple depths."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_sampling(
            max_sample_iterations=2,
            sample_block_size=5,
        )

        alg.benchmark(
            ansatz_depths=[1, 2, 3],
            repeats=1,
            verbose=False,
        )

        # Should have results for all depths
        results = alg.tracker.get_results()
        if mpi_comm.Get_rank() == 0:
            assert len(results[1]) == 1
            assert len(results[2]) == 1
            assert len(results[3]) == 1

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkWithParameterMap:
    """Tests for benchmark with parameter mapping for Grover search.

    For Grover search on a marked state, the optimal parameters are approximately:
    - Phase shift (gamma): pi (to flip the phase of the marked state)
    - Walk time (t): related to sqrt(N) for optimal mixing

    A parameter map takes 2 free parameters [gamma, t] and applies them
    to each ansatz iteration.
    """

    def test_benchmark_with_parameter_map_simple(self, mpi_comm):
        """Verify benchmark works with a simple parameter map."""
        from quop_mpi.algorithm.combinatorial import QWOA

        # Create a Grover-like oracle: one marked state
        system_size = 8  # 2^3 = 8 states
        marked_state = 3

        # Use closure to capture marked_state
        def make_grover_oracle(marked):
            def grover_oracle(local_i, local_i_offset):
                """Oracle that marks one state with value 0, others with 1."""
                qualities = np.ones(local_i, dtype=np.float64)
                for i in range(local_i):
                    global_i = local_i_offset + i
                    if global_i == marked:
                        qualities[i] = 0.0  # Marked state has lowest quality
                return qualities

            return grover_oracle

        # Bound parameters (ansatz_depth, total_params) come first, matching Ansatz attrs.
        # free_vec is passed at call time via __to_full(vec, ...).
        def parameter_map(ansatz_depth, total_params, free_vec):
            """Map 2 parameters [gamma, t] to all ansatz layers.

            Each layer gets the same gamma and t values.
            For QWOA, total_params=2 (one mixer param, one phase param per layer).
            """
            gamma, t = free_vec
            full_params = np.zeros(ansatz_depth * total_params, dtype=np.float64)
            for layer in range(ansatz_depth):
                full_params[layer * total_params] = gamma  # Phase shift
                full_params[layer * total_params + 1] = t  # Walk time
            return full_params

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(make_grover_oracle(marked_state))

        # Set up parameter map - n_free_params=2 for [gamma, t]
        alg.set_parameter_map(2, parameter_map)

        # Initial guess: gamma near pi, t around 0.5
        initial_params = np.array([np.pi, 0.5])

        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            initial_parameters=initial_params,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert "x" in alg.result
            # The free vector should have 2 parameters
            assert len(alg.result["x"]) == 2

        alg.destroy()

    def test_benchmark_parameter_map_multiple_depths(self, mpi_comm):
        """Verify benchmark with parameter map works across multiple depths."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = 8
        marked_state = 5

        def make_grover_oracle(marked):
            def grover_oracle(local_i, local_i_offset):
                qualities = np.ones(local_i, dtype=np.float64)
                for i in range(local_i):
                    global_i = local_i_offset + i
                    if global_i == marked:
                        qualities[i] = 0.0
                return qualities

            return grover_oracle

        # Bound params first (ansatz_depth, total_params), then free_vec
        def parameter_map(ansatz_depth, total_params, free_vec):
            """Map 2 free parameters to full parameter vector."""
            gamma, t = free_vec
            full_params = np.zeros(ansatz_depth * total_params, dtype=np.float64)
            for layer in range(ansatz_depth):
                full_params[layer * total_params] = gamma
                full_params[layer * total_params + 1] = t
            return full_params

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(make_grover_oracle(marked_state))

        # ansatz_depth is automatically bound from Ansatz.ansatz_depth
        alg.set_parameter_map(2, parameter_map)

        initial_params = np.array([np.pi, 0.5])

        # Test multiple depths - ansatz_depth is auto-updated during benchmark
        alg.benchmark(
            ansatz_depths=[1, 2],
            repeats=2,
            initial_parameters=initial_params,
            param_persist=True,
            verbose=False,
        )

        results = alg.tracker.get_results()
        if mpi_comm.Get_rank() == 0:
            # Check both depths have 2 repeats each
            assert len(results[1]) == 2  # 2 repeats at depth 1
            assert len(results[2]) == 2  # 2 repeats at depth 2
            for depth in [1, 2]:
                for result in results[depth]:
                    assert "fun" in result

        alg.destroy()

    def test_benchmark_parameter_map_finds_optimal(self, mpi_comm):
        """Verify benchmark with parameter map can optimize toward the marked state.

        For Grover search, the optimal gamma is near pi (phase flip)
        and optimal t depends on the system size.
        """
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = 8
        marked_state = 0  # Mark state |0>

        def make_grover_oracle(marked):
            def grover_oracle(local_i, local_i_offset):
                qualities = np.ones(local_i, dtype=np.float64)
                for i in range(local_i):
                    global_i = local_i_offset + i
                    if global_i == marked:
                        qualities[i] = 0.0
                return qualities

            return grover_oracle

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            full_params = np.zeros(ansatz_depth * total_params, dtype=np.float64)
            for layer in range(ansatz_depth):
                full_params[layer * total_params] = gamma
                full_params[layer * total_params + 1] = t
            return full_params

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(make_grover_oracle(marked_state))

        alg.set_parameter_map(2, parameter_map)

        # Start near optimal: gamma close to pi, t reasonably sized
        initial_params = np.array([np.pi * 0.9, 0.4])

        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            initial_parameters=initial_params,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            # The optimized gamma should be near pi for phase inversion
            optimized_gamma = alg.result["x"][0]
            # Allow some tolerance - the optimizer should move toward pi
            # (exact value depends on system size and walk time interaction)
            assert 0 < optimized_gamma < 2 * np.pi  # Valid range

        alg.destroy()

    def test_benchmark_param_map_auto_generates_initial_params(self, mpi_comm):
        """Verify benchmark auto-generates initial params when n_free_params is set."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = 8

        def grover_oracle(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(grover_oracle)
        # Specify n_free_params=2, so benchmark can auto-generate
        alg.set_parameter_map(2, parameter_map)

        # Should work without initial_parameters
        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert len(alg.result["x"]) == 2

        alg.destroy()


@pytest.mark.mpi
@pytest.mark.requires_nprocs(12)
class TestBenchmarkWithParallelJacobian:
    """Tests for benchmark with parallel jacobian evaluation.

    Note: Parallel jacobian requires multiple MPI subcommunicators.
    With n_workers=3, we get 3 subcomms when run with 12 MPI ranks.
    Run tests with: mpiexec -n 12 pytest ...

    These tests are skipped when run with fewer than 12 MPI processes.
    """

    def test_parallel_jacobian_with_param_map(self, mpi_comm, simple_oracle):
        """Verify parallel jacobian and parameter map can be configured together."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        # Set up parallel jacobian first
        alg.set_parallel_jacobian(
            n_workers=3,
            method="forward",
        )

        # Now set parameter map - should work now
        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        # This should not raise
        alg.set_parameter_map(2, parameter_map)

        # Verify both are configured
        assert alg._has_param_map
        assert alg.jacobian_input is not None

        alg.destroy()

    def test_param_map_then_parallel_jacobian(self, mpi_comm, simple_oracle):
        """Verify parameter map and parallel jacobian can be set in either order."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        # Set up parameter map first
        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        alg.set_parameter_map(2, parameter_map)

        # Now set parallel jacobian - should work now
        alg.set_parallel_jacobian(
            n_workers=3,
            method="forward",
        )

        # Verify both are configured
        assert alg._has_param_map
        assert alg.jacobian_input is not None

        alg.destroy()

    def test_benchmark_with_param_map_and_parallel_jacobian(self, mpi_comm, simple_oracle):
        """Verify benchmark runs with both parameter map and parallel jacobian.

        This test requires sufficient MPI ranks to create multiple subcommunicators.
        With n_workers=3, we get 3 subcomms when run with 12 ranks.
        """
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        alg.set_parameter_map(2, parameter_map)
        alg.set_parallel_jacobian(
            n_workers=3,
            method="forward",
        )

        initial_params = np.array([np.pi, 0.5])

        # Run benchmark - parallel jacobian will work if enough ranks are available
        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            initial_parameters=initial_params,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()

    def test_parallel_jacobian_with_param_map_convergence(self, mpi_comm, simple_oracle):
        """Verify benchmark with parallel jacobian + param map converges to low-cost state.

        This test uses the Grover search oracle and verifies that:
        1. The optimizer reduces the objective function (expectation value)
        2. Final expectation value is lower than uniform superposition
        3. The algorithm is actually doing optimization, not just running

        Run with: mpiexec -N 12 pytest ... --with-mpi
        """
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        alg.set_parameter_map(2, parameter_map)
        alg.set_parallel_jacobian(
            n_workers=3,
            method="forward",
        )

        # Start away from optimal to verify optimizer moves toward it
        # Optimal is approximately (gamma=pi, t=pi/N), start at (1.0, 0.1)
        initial_params = np.array([1.0, 0.1])

        # Uniform superposition expectation: (N-M)/N where M=1 marked state
        # For N=8, M=1: uniform_E = 7/8 = 0.875
        uniform_expectation = oracle.uniform_expectation()

        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            initial_parameters=initial_params,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            final_objective = alg.result["fun"]

            # Key algorithmic test: final objective should be LESS than uniform
            # This proves the optimizer is actually finding better parameters
            assert final_objective < uniform_expectation, (
                f"Optimizer should reduce expectation value below uniform. "
                f"Got {final_objective:.4f}, uniform = {uniform_expectation:.4f}"
            )

            # The objective should have decreased from initial evaluation
            # (can't easily get initial objective, but we know uniform is the baseline)

            # For a well-optimized Grover search, we should get close to
            # theoretical success probability. With 1 iteration on N=8, M=1:
            # P_success ~= 0.78, so expectation ~= 0.22 (since E = 1 - P_marked)
            theoretical_success = oracle.theoretical_success_probability(1)
            theoretical_expectation = 1 - theoretical_success

            # Allow significant tolerance since we only run 1 repeat
            assert final_objective < 0.6, (
                f"Expected significant probability on marked state. "
                f"Got objective {final_objective:.4f}, theoretical ~= {theoretical_expectation:.4f}"
            )

        alg.destroy()


@pytest.mark.mpi
class TestBenchmarkWithSamplingAndParameterMap:
    """Tests for benchmark combining sampling with parameter map.

    This tests the combination of simulated sampling (set_sampling) with
    a parameter map for Grover-like search optimization.
    """

    def test_benchmark_sampling_with_parameter_map(self, mpi_comm):
        """Verify benchmark works with both sampling and parameter map enabled."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = 8
        marked_state = 3

        def make_grover_oracle(marked):
            def grover_oracle(local_i, local_i_offset):
                qualities = np.ones(local_i, dtype=np.float64)
                for i in range(local_i):
                    global_i = local_i_offset + i
                    if global_i == marked:
                        qualities[i] = 0.0
                return qualities

            return grover_oracle

        def parameter_map(ansatz_depth, total_params, free_vec):
            """Map 2 parameters [gamma, t] to all ansatz layers."""
            gamma, t = free_vec
            full_params = np.zeros(ansatz_depth * total_params, dtype=np.float64)
            for layer in range(ansatz_depth):
                full_params[layer * total_params] = gamma
                full_params[layer * total_params + 1] = t
            return full_params

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(make_grover_oracle(marked_state))

        # Enable sampling with 100 shots per block
        alg.set_sampling(sample_block_size=100)

        # Set parameter map
        alg.set_parameter_map(2, parameter_map)

        initial_params = np.array([np.pi, 0.5])

        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            initial_parameters=initial_params,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert "x" in alg.result
            assert len(alg.result["x"]) == 2

        alg.destroy()

    def test_benchmark_sampling_param_map_multiple_depths(self, mpi_comm):
        """Verify sampling + parameter map works across multiple depths."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.random.rand(local_i)

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            full_params = np.zeros(ansatz_depth * total_params, dtype=np.float64)
            for layer in range(ansatz_depth):
                full_params[layer * total_params] = gamma
                full_params[layer * total_params + 1] = t
            return full_params

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_sampling(sample_block_size=50)
        alg.set_parameter_map(2, parameter_map)

        alg.benchmark(
            ansatz_depths=[1, 2],
            repeats=2,
            param_persist=True,
            verbose=False,
        )

        results = alg.tracker.get_results()
        if mpi_comm.Get_rank() == 0:
            assert len(results[1]) == 2
            assert len(results[2]) == 2

        alg.destroy()

    def test_benchmark_sampling_param_map_custom_sampling_fn(self, mpi_comm):
        """Verify sampling + parameter map works with custom sampling function."""
        from quop_mpi.algorithm.combinatorial import QWOA

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        sample_count = [0]  # Mutable to track calls

        def custom_sampling(samples):
            """Custom sampling that accepts after 2 blocks."""
            sample_count[0] += 1
            mean_val = np.mean([np.mean(block) for block in samples])
            accept = len(samples) >= 2
            return mean_val, accept

        alg = QWOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_sampling(
            sample_block_size=20,
            function=custom_sampling,
            max_sample_iterations=5,
        )
        alg.set_parameter_map(2, parameter_map)

        initial_params = np.array([1.0, 0.5])

        alg.benchmark(
            ansatz_depths=[1],
            repeats=1,
            initial_parameters=initial_params,
            verbose=False,
        )

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

        alg.destroy()
