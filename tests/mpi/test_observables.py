"""Tests for set_observables() method.

This module tests the observable configuration functionality including:
- Custom observable functions
- Observable function with FunctionDict
- Observable by index reference
- Interaction with state evolution and expectation values
"""

import pytest
import numpy as np
from mpi4py import MPI


@pytest.mark.mpi
class TestSetObservablesBasic:
    """Basic tests for set_observables() configuration."""

    def test_set_observables_accepts_function(self, mpi_comm, simple_oracle):
        """Verify set_observables accepts a callable function."""
        from quop_mpi.algorithm.combinatorial import qwoa

        alg = qwoa(simple_oracle.system_size, mpi_comm)

        # Use the oracle's qualities function
        qualities_fn = simple_oracle.qualities_function()
        alg.set_qualities(qualities_fn)

        # Verify it was set (indirectly through evolution)
        alg.set_depth(1)
        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        # Should be able to get expectation value
        expectation = alg.get_expectation_value()
        if mpi_comm.Get_rank() == 0:
            assert expectation is not None
            assert np.isfinite(expectation)

        del alg

    def test_set_observables_custom_uniform(self, mpi_comm):
        """Verify custom observable function with uniform values."""
        from quop_mpi.algorithm.combinatorial import qwoa

        system_size = 8

        # Observable that returns all zeros (best case)
        def all_zero_observable(local_i, local_i_offset):
            return np.zeros(local_i, dtype=np.float64)

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(all_zero_observable)
        alg.set_depth(1)

        # With all zero observables, expectation should be 0
        params = np.array([1.0, 0.5])
        alg.evolve_state(params)

        expectation = alg.get_expectation_value()
        if mpi_comm.Get_rank() == 0:
            assert expectation is not None
            assert (
                abs(expectation) < 1e-10
            ), f"Expected 0 for all-zero observable, got {expectation}"

        del alg

    def test_set_observables_custom_ones(self, mpi_comm):
        """Verify custom observable function with all ones."""
        from quop_mpi.algorithm.combinatorial import qwoa

        system_size = 8

        # Observable that returns all ones
        def all_ones_observable(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(all_ones_observable)
        alg.set_depth(1)

        # With all ones observables and normalized state, expectation should be 1
        params = np.array([0.0, 0.0])  # No evolution = uniform superposition
        alg.evolve_state(params)

        expectation = alg.get_expectation_value()
        if mpi_comm.Get_rank() == 0:
            assert expectation is not None
            assert (
                abs(expectation - 1.0) < 1e-10
            ), f"Expected 1 for all-ones observable, got {expectation}"

        del alg


@pytest.mark.mpi
class TestSetObservablesWithFunctionDict:
    """Tests for set_observables with FunctionDict parameter binding."""

    def test_observables_with_extra_args(self, mpi_comm):
        """Verify observable function can receive extra arguments via FunctionDict."""
        from quop_mpi.algorithm.combinatorial import qwoa

        system_size = 8

        # Observable that uses an extra parameter
        def scaled_observable(local_i, local_i_offset, scale_factor):
            return np.ones(local_i, dtype=np.float64) * scale_factor

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(scaled_observable, {"args": [0.5]})
        alg.set_depth(1)

        params = np.array([0.0, 0.0])
        alg.evolve_state(params)

        expectation = alg.get_expectation_value()
        if mpi_comm.Get_rank() == 0:
            # With scale_factor=0.5, expectation should be 0.5
            assert expectation is not None
            assert (
                abs(expectation - 0.5) < 1e-10
            ), f"Expected 0.5 for scaled observable, got {expectation}"

        del alg

    def test_observables_with_kwargs(self, mpi_comm):
        """Verify observable function receives keyword arguments."""
        from quop_mpi.algorithm.combinatorial import qwoa

        system_size = 8

        def marked_observable(local_i, local_i_offset, marked_index=0):
            obs = np.ones(local_i, dtype=np.float64)
            for i in range(local_i):
                if local_i_offset + i == marked_index:
                    obs[i] = 0.0
            return obs

        alg = qwoa(system_size, mpi_comm)
        # Mark state 3 using kwargs
        alg.set_qualities(marked_observable, {"kwargs": {"marked_index": 3}})
        alg.set_depth(1)

        params = np.array([np.pi, 0.5])
        alg.evolve_state(params)

        expectation = alg.get_expectation_value()
        if mpi_comm.Get_rank() == 0:
            # Should have lower expectation than uniform (7/8)
            assert expectation is not None
            assert (
                expectation < 0.875
            ), f"Expected some probability on marked state, got {expectation}"

        del alg


@pytest.mark.mpi
class TestSetObservablesAlgorithmicBehavior:
    """Tests that verify observable functions produce correct algorithmic behavior."""

    def test_grover_oracle_concentrates_probability(self, mpi_comm, simple_oracle):
        """Verify Grover oracle leads to probability concentration on marked states."""
        from quop_mpi.algorithm.combinatorial import qwoa

        oracle = simple_oracle

        alg = qwoa(oracle.system_size, mpi_comm)
        alg.set_qualities(oracle.qualities_function())
        alg.set_depth(1)

        # Optimal parameters
        params = oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        # Get expectation value
        expectation = alg.get_expectation_value()
        uniform_expectation = oracle.uniform_expectation()

        if mpi_comm.Get_rank() == 0:
            # Expectation should be less than uniform (more probability on marked states)
            assert expectation < uniform_expectation, (
                f"Grover evolution should reduce expectation. "
                f"Got {expectation}, uniform = {uniform_expectation}"
            )

        del alg

    def test_different_observables_different_expectations(self, mpi_comm):
        """Verify different observable functions produce different expectations."""
        from quop_mpi.algorithm.combinatorial import qwoa

        system_size = 8

        def obs_half(local_i, local_i_offset):
            return np.full(local_i, 0.5, dtype=np.float64)

        def obs_quarter(local_i, local_i_offset):
            return np.full(local_i, 0.25, dtype=np.float64)

        params = np.array([0.0, 0.0])  # No evolution

        # First observable: 0.5
        alg1 = qwoa(system_size, mpi_comm)
        alg1.set_qualities(obs_half)
        alg1.set_depth(1)
        alg1.evolve_state(params)
        exp1 = alg1.get_expectation_value()

        # Second observable: 0.25
        alg2 = qwoa(system_size, mpi_comm)
        alg2.set_qualities(obs_quarter)
        alg2.set_depth(1)
        alg2.evolve_state(params)
        exp2 = alg2.get_expectation_value()

        if mpi_comm.Get_rank() == 0:
            assert abs(exp1 - 0.5) < 1e-10
            assert abs(exp2 - 0.25) < 1e-10
            assert exp1 != exp2

        del alg1
        del alg2

    def test_observables_change_triggers_recompute(self, mpi_comm):
        """Verify changing observables triggers recomputation."""
        from quop_mpi.algorithm.combinatorial import qwoa

        system_size = 8
        params = np.array([0.0, 0.0])

        def obs_a(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        def obs_b(local_i, local_i_offset):
            return np.zeros(local_i, dtype=np.float64)

        alg = qwoa(system_size, mpi_comm)

        # First evolution with ones
        alg.set_qualities(obs_a)
        alg.set_depth(1)
        alg.evolve_state(params)
        exp_a = alg.get_expectation_value()

        # Change observables and re-evolve
        alg.set_qualities(obs_b)
        alg.evolve_state(params)
        exp_b = alg.get_expectation_value()

        if mpi_comm.Get_rank() == 0:
            assert abs(exp_a - 1.0) < 1e-10
            assert abs(exp_b - 0.0) < 1e-10

        del alg
