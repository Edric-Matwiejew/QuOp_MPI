"""Tests for set_parameter_map() with execute().

This module tests the parameter map functionality with the execute() method,
complementing the tests in test_benchmark.py that cover parameter map with benchmark().

Tests include:
- Parameter map with execute()
- Convergence with parameter map via execute()
- Parameter map interaction with different optimizers
- Error handling for parameter map
"""

import numpy as np
import pytest


@pytest.mark.mpi
class TestParameterMapWithExecute:
    """Tests for set_parameter_map() used with execute()."""

    def test_execute_with_parameter_map(self, mpi_comm, simple_oracle):
        """Verify execute() works with a parameter map."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        # Parameter map: 2 free params -> full params
        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        alg.set_parameter_map(2, parameter_map)
        alg.set_depth(1)

        # Execute with free parameters (2 params, not 2)
        initial_params = np.array([np.pi, oracle.optimal_walk_time])
        alg.execute(initial_params)

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert "x" in alg.result
            # Result should have 2 free params
            assert len(alg.result["x"]) == 2
            assert np.isfinite(alg.result["fun"])

        alg.destroy()

    def test_execute_param_map_requires_initial_params(self, mpi_comm, simple_oracle):
        """Verify execute() raises error when param map set but no initial params."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        def parameter_map(ansatz_depth, total_params, free_vec):
            return np.tile(free_vec, ansatz_depth)

        alg.set_parameter_map(2, parameter_map)
        alg.set_depth(1)

        # Should raise error without initial_parameters
        with pytest.raises(ValueError, match="[Pp]arameter map"):
            alg.execute()  # No variational_parameters

        alg.destroy()

    def test_execute_param_map_convergence(self, mpi_comm, simple_oracle):
        """Verify execute() with param map converges to low-cost state."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        alg.set_parameter_map(2, parameter_map)
        alg.set_depth(1)

        # Start away from optimal
        initial_params = np.array([1.0, 0.1])
        alg.execute(initial_params)

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            final_fun = alg.result["fun"]

            # Should be less than uniform superposition expectation
            uniform_exp = oracle.uniform_expectation()
            assert (
                final_fun < uniform_exp
            ), f"Should converge to lower expectation. Got {final_fun}, uniform={uniform_exp}"

        alg.destroy()

    def test_execute_param_map_multiple_depths(self, mpi_comm, simple_oracle):
        """Verify execute() with param map works at different depths."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        results = []
        for depth in [1, 2, 3]:
            alg = QWOA(oracle.system_size, mpi_comm)
            alg.set_qualities(oracle.qualities_function())
            alg.set_parameter_map(2, parameter_map)
            alg.set_depth(depth)

            initial_params = np.array([np.pi * 0.8, 0.3])
            alg.execute(initial_params)

            if mpi_comm.Get_rank() == 0:
                results.append((depth, alg.result["fun"]))

            alg.destroy()

        if mpi_comm.Get_rank() == 0:
            assert len(results) == 3
            # All should have converged to something
            for _depth, fun in results:
                assert np.isfinite(fun)


@pytest.mark.mpi
class TestParameterMapDifferentMappings:
    """Tests for various parameter mapping strategies."""

    def test_param_map_single_param(self, mpi_comm, simple_oracle):
        """Verify parameter map with single free parameter."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        # Single parameter controls both gamma and t
        def single_param_map(ansatz_depth, total_params, free_vec):
            theta = free_vec[0]
            # gamma = theta, t = theta/pi
            return np.tile([theta, theta / np.pi], ansatz_depth)

        alg.set_parameter_map(1, single_param_map)
        alg.set_depth(1)

        initial_params = np.array([2.0])
        alg.execute(initial_params)

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert len(alg.result["x"]) == 1

        alg.destroy()

    def test_param_map_layer_dependent(self, mpi_comm, simple_oracle):
        """Verify parameter map can create layer-dependent parameters."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        # 2 free params but layer-dependent scaling
        def layer_scaled_map(ansatz_depth, total_params, free_vec):
            gamma_base, t_base = free_vec
            full = np.zeros(ansatz_depth * total_params)
            for layer in range(ansatz_depth):
                scale = (layer + 1) / ansatz_depth
                full[layer * total_params] = gamma_base * scale
                full[layer * total_params + 1] = t_base * scale
            return full

        alg.set_parameter_map(2, layer_scaled_map)
        alg.set_depth(2)

        initial_params = np.array([np.pi, 0.5])
        alg.execute(initial_params)

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert len(alg.result["x"]) == 2
            # quop_result should have mapped_parameters
            assert "mapped_parameters" in alg.quop_result
            mapped = alg.quop_result["mapped_parameters"]
            assert len(mapped) == 4  # 2 layers * 2 params

        alg.destroy()

    def test_param_map_with_constants(self, mpi_comm, simple_oracle):
        """Verify parameter map can include constant parameters."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        # Only gamma is free, t is fixed
        fixed_t = 0.4

        def gamma_only_map(ansatz_depth, total_params, free_vec):
            gamma = free_vec[0]
            return np.tile([gamma, fixed_t], ansatz_depth)

        alg.set_parameter_map(1, gamma_only_map)
        alg.set_depth(1)

        initial_params = np.array([2.0])
        alg.execute(initial_params)

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert len(alg.result["x"]) == 1
            # Mapped t should be fixed
            mapped = alg.quop_result["mapped_parameters"]
            assert abs(mapped[1] - fixed_t) < 1e-10

        alg.destroy()


@pytest.mark.mpi
class TestParameterMapWithFunctionDict:
    """Tests for parameter map with FunctionDict."""

    def test_param_map_with_args(self, mpi_comm, simple_oracle):
        """Verify parameter map receives extra arguments via FunctionDict."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        # Parameter map that uses extra argument
        def scaled_map(ansatz_depth, total_params, free_vec, scale_factor):
            gamma, t = free_vec
            return np.tile([gamma * scale_factor, t], ansatz_depth)

        scale = 0.5
        alg.set_parameter_map(2, scaled_map, {"args": [scale]})
        alg.set_depth(1)

        initial_params = np.array([np.pi, 0.4])
        alg.execute(initial_params)

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            mapped = alg.quop_result["mapped_parameters"]
            assert np.isclose(mapped[0] / scale, alg.result["x"][0], atol=1e-8)

        alg.destroy()

    def test_param_map_with_kwargs(self, mpi_comm, simple_oracle):
        """Verify parameter map receives keyword arguments."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        def map_with_offset(ansatz_depth, total_params, free_vec, gamma_offset=0.0):
            gamma, t = free_vec
            return np.tile([gamma + gamma_offset, t], ansatz_depth)

        offset = 0.5
        alg.set_parameter_map(2, map_with_offset, {"kwargs": {"gamma_offset": offset}})
        alg.set_depth(1)

        initial_params = np.array([np.pi, 0.4])
        alg.execute(initial_params)

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            mapped = alg.quop_result["mapped_parameters"]
            # Mapping relation: mapped gamma = free_gamma + offset
            assert np.isclose(mapped[0] - offset, alg.result["x"][0], atol=1e-8)

        alg.destroy()


@pytest.mark.mpi
class TestParameterMapValidation:
    """Tests for parameter map error handling and validation."""

    def test_param_map_wrong_output_size_raises(self, mpi_comm, simple_oracle):
        """Verify error when parameter map returns wrong size vector."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        # Map returns wrong size
        def bad_map(ansatz_depth, total_params, free_vec):
            return np.array([1.0])  # Wrong size

        alg.set_parameter_map(2, bad_map)
        alg.set_depth(1)

        initial_params = np.array([np.pi, 0.4])

        # Setup must be called first so we can check in_subcomm().
        # The ValueError fires inside subcomm-scoped __to_full();
        # excluded ranks skip it entirely.
        alg.setup()

        if alg.subcomms.in_subcomm():
            with pytest.raises(ValueError):
                alg.execute(initial_params)
        else:
            alg.execute(initial_params)

        alg.destroy()

    def test_param_map_result_contains_free_params(self, mpi_comm, simple_oracle):
        """Verify result['x'] contains free parameters, not full."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        n_free = 2
        alg.set_parameter_map(n_free, parameter_map)
        alg.set_depth(3)  # 3 layers = 6 full params, but only 2 free

        initial_params = np.array([np.pi, 0.4])
        alg.execute(initial_params)

        if mpi_comm.Get_rank() == 0:
            # result['x'] should have n_free params
            assert len(alg.result["x"]) == n_free
            # quop_result should have both
            assert len(alg.quop_result["variational_parameters"]) == n_free
            assert len(alg.quop_result["mapped_parameters"]) == 6

        alg.destroy()


@pytest.mark.mpi
class TestParameterMapWithEvolveState:
    """Tests for parameter map with evolve_state()."""

    def test_evolve_state_with_param_map(self, mpi_comm, simple_oracle):
        """Verify evolve_state() works with parameter map."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        alg.set_parameter_map(2, parameter_map)
        alg.set_depth(1)

        # evolve_state with free parameters
        free_params = np.array([np.pi, oracle.optimal_walk_time])
        alg.evolve_state(free_params)

        # Should be able to get probabilities
        probs = alg.get_probabilities()
        if probs is not None:
            assert len(probs) == oracle.system_size
            assert abs(np.sum(probs) - 1.0) < 1e-10

        alg.destroy()

    def test_evolve_state_param_map_matches_grover(self, mpi_comm, simple_oracle):
        """Verify evolve_state with param map achieves Grover probability."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        alg.set_parameter_map(2, parameter_map)
        alg.set_depth(1)

        # Optimal free parameters
        free_params = np.array([np.pi, oracle.optimal_walk_time])
        alg.evolve_state(free_params)

        # Gather probabilities to rank 0
        local_probs = alg.get_probabilities()
        if mpi_comm.Get_rank() == 0 and local_probs is not None:
            full_probs = local_probs
        else:
            full_probs = None

        # Use get_final_state to gather properly
        state = alg.get_final_state()

        if mpi_comm.Get_rank() == 0 and state is not None:
            full_probs = np.abs(state) ** 2
            marked_prob = oracle.compute_marked_probability(full_probs)
            theoretical = oracle.theoretical_success_probability(1)

            assert abs(marked_prob - theoretical) < 0.01, (
                f"Param map evolution should match Grover: "
                f"got {marked_prob:.4f}, expected {theoretical:.4f}"
            )

        alg.destroy()
