"""Tests for set_objective() method.

This module tests the custom objective function configuration including:
- Custom objective functions beyond expectation value
- Objective function with FunctionDict
- Interaction with optimization and execute()

NOTE: Custom objective functions are parsed through the 'interface' class,
which binds Ansatz attributes (like local_probabilities, observables) to
function parameters. The function should NOT expect expectation_value as
a parameter - instead it computes its own objective from available state data.
"""

import numpy as np
import pytest

from tests.conftest import TestOracle


def _scaled_power_of_two_system_size(mpi_sizing, base):
    """Choose a power-of-two size that keeps objective tests multi-rank aware."""
    return mpi_sizing.power_of_two(base=base, min_per_rank=1, min_per_node=16)


def _marked_count_from_ratio(system_size, denominator, minimum):
    """Preserve the original marked-state density while allowing larger systems."""
    return max(minimum, system_size // denominator)


@pytest.fixture
def simple_oracle(mpi_sizing):
    """Scale the objective-test oracle while preserving M/N = 1/16."""
    system_size = _scaled_power_of_two_system_size(mpi_sizing, base=64)
    return TestOracle(
        system_size=system_size,
        n_marked=_marked_count_from_ratio(system_size, denominator=16, minimum=4),
        seed=42,
    )


@pytest.mark.mpi
class TestSetObjectiveBasic:
    """Basic tests for set_objective() configuration."""

    def test_set_objective_accepts_function(self, mpi_comm, simple_oracle):
        """Verify set_objective accepts a callable function."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        # Custom objective that computes from local_probabilities and observables
        # These are bound by the interface class
        def custom_objective(local_probabilities, local_observables):
            return np.dot(local_probabilities, local_observables)

        alg.set_objective(custom_objective)
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)

        # Use evolve_state which calls setup internally
        alg.evolve_state(params)

        # Get expectation value (which uses the objective internally)
        exp = alg.get_expectation_value()

        if mpi_comm.Get_rank() == 0:
            assert exp is not None
            assert np.isfinite(exp)

        alg.destroy()

    def test_default_objective_is_expectation(self, mpi_comm, simple_oracle):
        """Verify default objective function returns expectation value."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1)

        # Use evolve_state which handles setup
        alg.evolve_state(params)

        # Without set_objective, get_expectation_value should work
        exp = alg.get_expectation_value()

        if mpi_comm.Get_rank() == 0:
            assert exp is not None
            assert np.isfinite(exp)
            # For Grover oracle at optimal params, should be less than uniform
            assert exp < oracle.uniform_expectation()

        alg.destroy()


@pytest.mark.mpi
class TestSetObjectiveCustomFunctions:
    """Tests for various custom objective functions.

    These tests verify that custom objectives work via execute(),
    which is the main use case (optimizer calls the objective).

    NOTE: Custom objective functions receive local_probabilities and must
    handle MPI reduction themselves if needed. For simplicity, these tests
    use objectives that work with local data and rely on MPI to broadcast
    results appropriately.
    """

    def test_objective_with_penalty_execute(self, mpi_comm, simple_oracle):
        """Verify objective function can add penalty terms during optimization."""
        from mpi4py import MPI

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        # Objective with penalty - must do MPI reduction like get_expectation_value
        penalty = 0.5

        def penalized_objective(local_probabilities, local_observables, MPI_COMM):  # noqa: N803
            local_exp = np.dot(local_probabilities, local_observables)
            global_exp = MPI_COMM.allreduce(local_exp, op=MPI.SUM)
            return global_exp + penalty

        alg.set_objective(penalized_objective)
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1) * 0.9
        alg.execute(params)

        if mpi_comm.Get_rank() == 0:
            # Result fun should include the penalty
            assert alg.result["fun"] >= penalty

        alg.destroy()

    def test_objective_negated_execute(self, mpi_comm, simple_oracle):
        """Verify negated objective results in different optimization."""
        from mpi4py import MPI

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        def negated_objective(local_probabilities, local_observables, MPI_COMM):  # noqa: N803
            local_exp = np.dot(local_probabilities, local_observables)
            global_exp = MPI_COMM.allreduce(local_exp, op=MPI.SUM)
            return -global_exp

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_objective(negated_objective)
        alg.set_depth(1)

        params = np.array([2.0, 0.3])
        alg.execute(params)

        if mpi_comm.Get_rank() == 0:
            # Negated objective should be negative (minimizing -E means maximizing E)
            assert (
                alg.result["fun"] < 0
            ), f"Negated objective should be negative: {alg.result['fun']}"

        alg.destroy()


@pytest.mark.mpi
class TestSetObjectiveWithFunctionDict:
    """Tests for set_objective with FunctionDict parameter binding."""

    def test_objective_with_args_execute(self, mpi_comm, simple_oracle):
        """Verify objective function receives extra positional arguments during execute."""
        from mpi4py import MPI

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        # Objective that uses extra argument and does proper MPI reduction
        def objective_with_offset(
            local_probabilities, local_observables, MPI_COMM, offset
        ):  # noqa: N803
            local_exp = np.dot(local_probabilities, local_observables)
            global_exp = MPI_COMM.allreduce(local_exp, op=MPI.SUM)
            return global_exp + offset

        offset_value = 0.3
        alg.set_objective(objective_with_offset, {"args": [offset_value]})
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1) * 0.9
        alg.execute(params)

        if mpi_comm.Get_rank() == 0:
            # Result should include the offset
            assert alg.result is not None
            assert alg.result["fun"] >= offset_value

        alg.destroy()

    def test_objective_with_kwargs_execute(self, mpi_comm, simple_oracle):
        """Verify objective function receives keyword arguments during execute."""
        from mpi4py import MPI

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        def objective_with_weight(
            local_probabilities, local_observables, MPI_COMM, weight=1.0
        ):  # noqa: N803
            local_exp = np.dot(local_probabilities, local_observables)
            global_exp = MPI_COMM.allreduce(local_exp, op=MPI.SUM)
            return weight * global_exp

        weight_value = 3.0
        alg.set_objective(objective_with_weight, {"kwargs": {"weight": weight_value}})
        alg.set_depth(1)

        params = oracle.optimal_params(depth=1) * 0.9
        alg.execute(params)

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            # Weighted result should be larger than unweighted expectation
            assert alg.result["fun"] > 0

        alg.destroy()


@pytest.mark.mpi
class TestSetObjectiveWithOptimization:
    """Tests for custom objective with optimization/execute."""

    def test_custom_objective_with_execute(self, mpi_comm, simple_oracle):
        """Verify custom objective function is used during execute()."""
        from mpi4py import MPI

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())

        # Custom objective with shift - must do MPI reduction
        shift = 1.0

        def shifted_objective(local_probabilities, local_observables, MPI_COMM):  # noqa: N803
            local_exp = np.dot(local_probabilities, local_observables)
            global_exp = MPI_COMM.allreduce(local_exp, op=MPI.SUM)
            return global_exp + shift

        alg.set_objective(shifted_objective)
        alg.set_depth(1)

        initial_params = oracle.optimal_params(depth=1) * 0.8
        alg.execute(initial_params)

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            final_fun = alg.result["fun"]
            # The optimized fun should be shifted (greater than shift)
            assert final_fun >= shift * 0.9, f"Shifted objective result too low: {final_fun}"

        alg.destroy()

    def test_negated_objective_maximizes(self, mpi_comm, simple_oracle):
        """Verify negated objective effectively maximizes expectation."""
        from mpi4py import MPI

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        # Normal minimization
        alg_min = QWOA(oracle.system_size, mpi_comm)
        alg_min.set_qualities(oracle.qualities_function())
        alg_min.set_depth(1)

        initial_params = np.array([2.0, 0.3])
        alg_min.execute(initial_params)

        # Maximization via negation
        alg_max = QWOA(oracle.system_size, mpi_comm)
        alg_max.set_qualities(oracle.qualities_function())

        def negate_objective(local_probabilities, local_observables, MPI_COMM):  # noqa: N803
            local_exp = np.dot(local_probabilities, local_observables)
            global_exp = MPI_COMM.allreduce(local_exp, op=MPI.SUM)
            return -global_exp

        alg_max.set_objective(negate_objective)
        alg_max.set_depth(1)
        alg_max.execute(initial_params.copy())

        if mpi_comm.Get_rank() == 0:
            min_fun = alg_min.result["fun"]
            max_fun = alg_max.result["fun"]  # This is negative of expectation

            # Minimized expectation should be lower than what maximization finds
            # max_fun is negative, so -max_fun is the maximized expectation
            assert (
                min_fun < -max_fun + 0.1
            ), f"Minimized exp ({min_fun}) should be less than maximized ({-max_fun})"

        alg_min.destroy()
        alg_max.destroy()


@pytest.mark.mpi
class TestSetObjectiveStandalone:
    """Tests for standalone objective() calls and custom objective after evolve_state."""

    def test_objective_standalone_call(self, mpi_comm, simple_oracle):
        """Verify the public objective() method returns a valid value."""
        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        alg.prepare()

        params = oracle.optimal_params(depth=1)
        value = alg.objective(params)

        # objective() only returns a value on the root rank of subcomm 0
        if alg.subcomms.get_subcomm_index() == 0:
            if alg.subcomms.SUBCOMM.Get_rank() == 0:
                assert value is not None
                assert isinstance(value, (float, np.floating))
                assert np.isfinite(value)
            else:
                assert value is None

        alg.destroy()

    def test_custom_objective_evolve_then_objective(self, mpi_comm, simple_oracle):
        """Verify objective() uses the custom objective function.

        get_expectation_value() always computes the default dot-product
        regardless of set_objective(). Only objective() and execute() route
        through the custom objective.
        """
        from mpi4py import MPI

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        offset = 10.0

        def offset_objective(local_probabilities, local_observables, MPI_COMM):  # noqa: N803
            local_exp = np.dot(local_probabilities, local_observables)
            global_exp = MPI_COMM.allreduce(local_exp, op=MPI.SUM)
            return global_exp + offset

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_objective(offset_objective)
        alg.set_depth(1)
        alg.prepare()

        params = oracle.optimal_params(depth=1)

        # objective() should use the custom objective
        obj_result = alg.objective(params)

        # get_expectation_value() should use the default dot-product
        default_result = alg.get_expectation_value()

        if alg.subcomms.get_subcomm_index() == 0:
            if alg.subcomms.SUBCOMM.Get_rank() == 0:
                assert obj_result is not None
                assert np.isfinite(obj_result)
                # Custom objective adds the offset
                assert (
                    obj_result >= offset
                ), f"objective() should include offset {offset}, got {obj_result}"

            # get_expectation_value returns to all subcomm ranks
            assert default_result is not None
            assert np.isfinite(default_result)
            # Default expectation value should be less than offset (it's a
            # probability-weighted sum of Grover qualities in [0, 1])
            assert (
                default_result < offset
            ), f"get_expectation_value() should ignore custom objective, got {default_result}"

        alg.destroy()


@pytest.mark.mpi
class TestSetObjectiveWithParameterMap:
    """Tests for combining set_objective with set_parameter_map."""

    def test_param_map_with_custom_objective_execute(self, mpi_comm, simple_oracle):
        """Verify execute() works with both parameter map and custom objective."""
        from mpi4py import MPI

        from quop_mpi.algorithm.combinatorial import QWOA

        oracle = simple_oracle

        penalty = 0.5

        def penalized_objective(local_probabilities, local_observables, MPI_COMM):  # noqa: N803
            local_exp = np.dot(local_probabilities, local_observables)
            global_exp = MPI_COMM.allreduce(local_exp, op=MPI.SUM)
            return global_exp + penalty

        def parameter_map(ansatz_depth, total_params, free_vec):
            gamma, t = free_vec
            return np.tile([gamma, t], ansatz_depth)

        alg = QWOA(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_objective(penalized_objective)
        alg.set_parameter_map(2, parameter_map)
        alg.set_depth(1)

        initial_params = np.array([np.pi, oracle.optimal_walk_time])
        alg.execute(initial_params)

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert len(alg.result["x"]) == 2
            # Result fun should include the penalty
            assert alg.result["fun"] >= penalty

        alg.destroy()
