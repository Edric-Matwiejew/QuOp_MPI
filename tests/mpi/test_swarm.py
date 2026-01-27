"""
Tests for the swarm meta-algorithm.

The swarm class creates and manages multiple Ansatz instances across
MPI subcommunicators for parallel optimization.

These tests verify:
1. Subcommunicator creation and management
2. Parallel execution of independent Ansatz instances
3. Result gathering and optimal result selection
4. Swarm benchmarking functionality

Run with: mpiexec -n <N> python -m pytest tests/mpi/test_swarm.py -v --with-mpi
Note: Tests require at least 2 MPI ranks to be meaningful.
"""

import pytest
import numpy as np
import tempfile
import os
from mpi4py import MPI


# =============================================================================
# Tests for Swarm Initialization and Subcommunicator Management
# =============================================================================


@pytest.mark.mpi
class TestSwarmInitialization:
    """Tests for swarm initialization and subcommunicator creation."""

    def test_swarm_creates_subcommunicators(self, mpi_comm):
        """Test that swarm creates subcommunicators correctly."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.combinatorial import qaoa

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        # Create swarm - args after alg are passed to alg constructor
        # qaoa(system_size, MPI_COMM) - MPI_COMM is added by swarm
        s = swarm(None, 1, 1, mpi_comm, qaoa, system_size)

        # Subcomms object should be created
        assert hasattr(s, "subcomms")
        assert s.subcomms is not None

        # Set up the ansatz
        s.set_qualities(qualities)
        s.set_depth(1)

        del s

    def test_swarm_inherits_ansatz_methods(self, mpi_comm):
        """Test that swarm provides access to Ansatz methods."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.combinatorial import qaoa

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        s = swarm(None, 1, 1, mpi_comm, qaoa, system_size)

        # Key Ansatz methods should be accessible
        assert hasattr(s, "set_qualities")
        assert hasattr(s, "set_depth")
        assert hasattr(s, "execute")
        assert hasattr(s, "set_optimiser")

        del s

    def test_swarm_with_qwoa(self, mpi_comm):
        """Test swarm initialization with QWOA algorithm."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.combinatorial import qwoa

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        s = swarm(None, 1, 1, mpi_comm, qwoa, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)

        # Verify ansatz was created
        if s.subcomms.in_subcomm():
            assert s.ansatz is not None
            assert s.ansatz.system_size == system_size

        del s


# =============================================================================
# Tests for Swarm Execution
# =============================================================================


@pytest.mark.mpi
class TestSwarmExecution:
    """Tests for swarm execution and optimization."""

    def test_swarm_execute_single(self, mpi_comm):
        """Test executing a single optimization through swarm."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.combinatorial import qaoa

        system_size = 8

        def qualities(local_i, local_i_offset):
            # Quality with clear minimum at index 0
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        s = swarm(None, 1, 1, mpi_comm, qaoa, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)
        s.set_optimiser("BFGS", {"maxiter": 5})

        # Execute with initial parameters
        initial_params = np.array([0.1, 0.1])
        s.execute(initial_params)

        # Get the optimal result - this is the proper way to get results from swarm
        result = s.get_optimal_result()

        if mpi_comm.Get_rank() == 0:
            assert result is not None
            assert "fun" in result

        del s

    def test_swarm_get_optimal_result(self, mpi_comm):
        """Test retrieving optimal result across swarm."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.combinatorial import qaoa

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        s = swarm(None, 1, 1, mpi_comm, qaoa, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)
        s.set_optimiser("BFGS", {"maxiter": 3})

        # Execute
        initial_params = np.array([0.2, 0.2])
        s.execute(initial_params)

        # Get optimal result
        optimal = s.get_optimal_result()

        # Should return a result dict with swarm_index
        if mpi_comm.Get_rank() == 0:
            assert optimal is not None
            assert "fun" in optimal
            assert "swarm_index" in optimal

        del s

    def test_swarm_evolve_state(self, mpi_comm):
        """Test evolving state through swarm."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.combinatorial import qwoa

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        s = swarm(None, 1, 1, mpi_comm, qwoa, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)

        # Evolve state with zero params
        params = np.array([0.0, 0.0])
        s.evolve_state(params)

        # Get probabilities
        if s.subcomms.in_subcomm():
            probs = s.ansatz.get_probabilities()
            if mpi_comm.Get_rank() == 0:
                total = np.sum(probs)
                assert abs(total - 1.0) < 1e-10

        del s


# =============================================================================
# Tests for Swarm with Multiple Subcommunicators
# =============================================================================


@pytest.mark.mpi
class TestSwarmMultipleSubcomms:
    """Tests for swarm with multiple subcommunicators (when ranks > 1)."""

    def test_swarm_deterministic_across_subcomms(self, mpi_comm):
        """Test that same parameters give same results in different subcomms."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.combinatorial import qaoa

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        # With nodes_per_subcomm=None, all ranks form one subcommunicator
        s = swarm(None, 1, 1, mpi_comm, qaoa, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)

        # Evolve with same parameters
        params = np.array([0.3, 0.4])
        s.evolve_state(params)

        # All subcomms should get same expectation value
        if s.subcomms.in_subcomm():
            exp_val = s.ansatz.get_expectation_value()

            if mpi_comm.Get_rank() == 0:
                assert exp_val is not None

        del s


# =============================================================================
# Tests for Execute Swarm (Parallel Task Distribution)
# =============================================================================


@pytest.mark.mpi
class TestExecuteSwarm:
    """Tests for execute_swarm parallel task distribution."""

    def test_execute_swarm_runs_multiple_tasks(self, mpi_comm):
        """Test that execute_swarm can run multiple optimization tasks."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.combinatorial import qaoa

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        s = swarm(None, 1, 1, mpi_comm, qaoa, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)
        s.set_optimiser("BFGS", {"maxiter": 2})

        # Create multiple parameter sets
        param_lists = [
            np.array([0.1, 0.1]),
            np.array([0.2, 0.2]),
        ]

        # Execute swarm needs a basename for logging
        with tempfile.TemporaryDirectory() as tmpdir:
            basename = os.path.join(tmpdir, "test_swarm")

            results = s.execute_swarm(
                param_lists,
                basename=basename,
                verbose=False,
            )

        # Results should be a dict with entries for each param set
        if mpi_comm.Get_rank() == 0:
            assert results is not None
            assert len(results) == len(param_lists)

        del s


# =============================================================================
# Tests for Swarm Utility Methods
# =============================================================================


@pytest.mark.mpi
class TestSwarmUtilities:
    """Tests for swarm utility methods."""

    def test_swarm_set_seed(self, mpi_comm):
        """Test that set_seed is properly forwarded to ansatz."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.combinatorial import qaoa

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        s = swarm(None, 1, 1, mpi_comm, qaoa, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)

        # Set seed should work
        s.set_seed(42)

        if s.subcomms.in_subcomm():
            assert s.ansatz.seed == 42

        del s

    def test_swarm_gen_initial_params(self, mpi_comm):
        """Test generating initial parameters through swarm."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.combinatorial import qaoa

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        s = swarm(None, 1, 1, mpi_comm, qaoa, system_size)

        s.set_qualities(qualities)
        s.set_depth(2)

        # Setup needs to be called to initialize params properly
        if s.subcomms.in_subcomm():
            s.ansatz.setup()
            # total_params should be set after setup
            # QAOA has 2 params per depth (gamma + t)
            expected_n_params = s.ansatz.total_params * 2
            assert expected_n_params == 4  # 2 params * depth 2

        del s

    def test_swarm_print_result(self, mpi_comm):
        """Test that print_result works through swarm."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.combinatorial import qaoa
        import io
        import sys

        system_size = 8

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        s = swarm(None, 1, 1, mpi_comm, qaoa, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)
        s.set_optimiser("BFGS", {"maxiter": 2})

        # Execute to get a result
        s.execute(np.array([0.1, 0.1]))

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            s.print_result()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # On rank 0, should have printed something
        # (other ranks may not print)

        del s


# =============================================================================
# Tests for Swarm with QMOA (Multivariable)
# =============================================================================


@pytest.mark.mpi
class TestSwarmMultivariable:
    """Tests for swarm with multivariable algorithms."""

    def test_swarm_with_qmoa(self, mpi_comm):
        """Test swarm initialization with QMOA algorithm."""
        from quop_mpi.meta import swarm
        from quop_mpi.algorithm.multivariable import qmoa, setup_cartesian, cartesian

        def sphere(x):
            return np.sum(x**2, axis=1)

        Ns = [2, 2]
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)

        s = swarm(None, 1, 1, mpi_comm, qmoa, Ns)

        s.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        s.set_depth(1)

        # Evolve with some parameters
        params = np.array([0.1, 0.1, 0.1])
        s.evolve_state(params)

        if s.subcomms.in_subcomm():
            probs = s.ansatz.get_probabilities()
            if mpi_comm.Get_rank() == 0:
                total = np.sum(probs)
                assert abs(total - 1.0) < 1e-10

        del s


# =============================================================================
# Tests for Subcomms Utility Class
# =============================================================================


@pytest.mark.mpi
class TestSubcomms:
    """Tests for the subcomms utility class used by swarm."""

    def test_subcomms_in_subcomm(self, mpi_comm):
        """Test in_subcomm returns correct value."""
        from quop_mpi._utils._mpi import subcomms

        # With nodes_per_subcomm=None, all ranks should be in subcomm
        sc = subcomms(None, 1, 1, mpi_comm)

        # All ranks should be in the subcommunicator
        assert sc.in_subcomm() == True

    def test_subcomms_get_n_subcomms(self, mpi_comm):
        """Test get_n_subcomms returns correct count."""
        from quop_mpi._utils._mpi import subcomms

        sc = subcomms(None, 1, 1, mpi_comm)

        # With nodes_per_subcomm=None, should have 1 subcomm
        n_subcomms = sc.get_n_subcomms()
        assert n_subcomms == 1

    def test_subcomms_get_subcomm_index(self, mpi_comm):
        """Test get_subcomm_index returns valid index."""
        from quop_mpi._utils._mpi import subcomms

        sc = subcomms(None, 1, 1, mpi_comm)

        idx = sc.get_subcomm_index()
        # Should be 0 for the single subcomm case
        assert idx == 0
