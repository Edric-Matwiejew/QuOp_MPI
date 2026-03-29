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

import os
import tempfile

import numpy as np
import pytest

# =============================================================================
# Tests for Swarm Initialization and Subcommunicator Management
# =============================================================================


def _scaled_grid_exponents(mpi_sizing, base_exponents):
    """Scale multivariable swarm grids while preserving their dimensions."""
    exponents = [int(exponent) for exponent in base_exponents]
    extra_bits = max(0, (mpi_sizing.topology.world_size - 1).bit_length() - 1)
    for _ in range(extra_bits):
        smallest = min(exponents)
        index = exponents.index(smallest)
        exponents[index] += 1
    return exponents


@pytest.fixture
def swarm_system_size(small_system_size):
    """Small power-of-two size for swarm algorithm tests."""
    return small_system_size


@pytest.fixture
def swarm_grid_ns_2d(mpi_sizing):
    """2D multivariable swarm grid that scales with MPI size."""
    return _scaled_grid_exponents(mpi_sizing, [2, 2])


@pytest.mark.mpi
class TestSwarmInitialization:
    """Tests for swarm initialization and subcommunicator creation."""

    def test_swarm_creates_subcommunicators(self, mpi_comm, swarm_system_size):
        """Test that swarm creates subcommunicators correctly."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.meta import Swarm

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        # Create swarm - args after alg are passed to alg constructor
        # QAOA(system_size, MPI_COMM) - MPI_COMM is added by swarm
        s = Swarm(1, mpi_comm, QAOA, system_size)

        # Subcomms object should be created
        assert hasattr(s, "swarm_comms")
        assert s.swarm_comms is not None

        # Set up the ansatz
        s.set_qualities(qualities)
        s.set_depth(1)

        s.destroy()

    def test_swarm_inherits_ansatz_methods(self, mpi_comm, swarm_system_size):
        """Test that swarm provides access to Ansatz methods."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.meta import Swarm

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        s = Swarm(1, mpi_comm, QAOA, system_size)

        # Key Ansatz methods should be accessible
        assert hasattr(s, "set_qualities")
        assert hasattr(s, "set_depth")
        assert hasattr(s, "execute")
        assert hasattr(s, "set_optimiser")

        s.destroy()

    def test_swarm_with_qwoa(self, mpi_comm, swarm_system_size):
        """Test swarm initialization with QWOA algorithm."""
        from quop_mpi.algorithm.combinatorial import QWOA
        from quop_mpi.meta import Swarm

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        s = Swarm(1, mpi_comm, QWOA, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)

        # Verify ansatz was created
        if s.swarm_comms.in_subcomm():
            assert s.ansatz is not None
            assert s.ansatz.system_size == system_size

        s.destroy()


# =============================================================================
# Tests for Swarm Execution
# =============================================================================


@pytest.mark.mpi
class TestSwarmExecution:
    """Tests for swarm execution and optimization."""

    def test_swarm_execute_single(self, mpi_comm, swarm_system_size):
        """Test executing a single optimization through swarm."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.meta import Swarm

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            # Quality with clear minimum at index 0
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        s = Swarm(1, mpi_comm, QAOA, system_size)

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

        s.destroy()

    def test_swarm_get_optimal_result(self, mpi_comm, swarm_system_size):
        """Test retrieving optimal result across swarm."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.meta import Swarm

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        s = Swarm(1, mpi_comm, QAOA, system_size)

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

        s.destroy()

    def test_swarm_evolve_state(self, mpi_comm, swarm_system_size):
        """Test evolving state through swarm."""
        from quop_mpi.algorithm.combinatorial import QWOA
        from quop_mpi.meta import Swarm

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        s = Swarm(1, mpi_comm, QWOA, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)

        # Evolve state with zero params
        params = np.array([0.0, 0.0])
        s.evolve_state(params)

        # Get probabilities
        if s.swarm_comms.in_subcomm():
            probs = s.ansatz.get_probabilities()
            if mpi_comm.Get_rank() == 0:
                total = np.sum(probs)
                assert abs(total - 1.0) < 1e-10

        s.destroy()


# =============================================================================
# Tests for Swarm with Multiple Subcommunicators
# =============================================================================


@pytest.mark.mpi
class TestSwarmMultipleSubcomms:
    """Tests for swarm with multiple subcommunicators (when ranks > 1)."""

    def test_swarm_deterministic_across_subcomms(self, mpi_comm, swarm_system_size):
        """Test that same parameters give same results in different subcomms."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.meta import Swarm

        if mpi_comm.Get_size() < 2:
            pytest.skip("Need at least 2 MPI ranks for multiple swarm subcommunicators")

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        s = Swarm(2, mpi_comm, QAOA, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)

        # Evolve with same parameters
        params = np.array([0.3, 0.4])
        s.evolve_state(params)

        local_result = None
        if s.swarm_comms.in_subcomm():
            exp_val = s.ansatz.get_expectation_value()
            if s.swarm_comms.in_rootcomm():
                local_result = (s.swarm_comms.get_subcomm_index(), float(exp_val))

        gathered = mpi_comm.gather(local_result, root=0)

        if mpi_comm.Get_rank() == 0:
            subcomm_results = {idx: value for item in gathered if item is not None for idx, value in [item]}
            assert len(subcomm_results) == 2
            reference = next(iter(subcomm_results.values()))
            for value in subcomm_results.values():
                assert np.isclose(value, reference), subcomm_results

        s.destroy()


# =============================================================================
# Tests for Execute Swarm (Parallel Task Distribution)
# =============================================================================


@pytest.mark.mpi
class TestExecuteSwarm:
    """Tests for execute_swarm parallel task distribution."""

    def test_execute_swarm_runs_multiple_tasks(self, mpi_comm, swarm_system_size):
        """Test that execute_swarm can run multiple optimization tasks."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.meta import Swarm

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64)

        s = Swarm(1, mpi_comm, QAOA, system_size)

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

        s.destroy()


# =============================================================================
# Tests for Swarm Utility Methods
# =============================================================================


@pytest.mark.mpi
class TestSwarmUtilities:
    """Tests for swarm utility methods."""

    def test_swarm_destroy_tears_down_ansatz(self, mpi_comm, swarm_system_size):
        """Destroy should release the owned Ansatz before freeing subcomms."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.meta import Swarm

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        s = Swarm(1, mpi_comm, QAOA, system_size)
        s.set_qualities(qualities)
        s.set_depth(1)

        if s.swarm_comms.in_subcomm():
            assert s.ansatz is not None

        s.destroy()

        assert s.swarm_comms is None
        if mpi_comm.Get_size() >= 1:
            assert s.ansatz is None

    def test_swarm_set_seed(self, mpi_comm, swarm_system_size):
        """Test that set_seed is properly forwarded to ansatz."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.meta import Swarm

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        s = Swarm(1, mpi_comm, QAOA, system_size)

        s.set_qualities(qualities)
        s.set_depth(1)

        # Set seed should work
        s.set_seed(42)

        if s.swarm_comms.in_subcomm():
            assert s.ansatz.seed == 42

        s.destroy()

    def test_swarm_gen_initial_params(self, mpi_comm, swarm_system_size):
        """Test generating initial parameters through swarm."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.meta import Swarm

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        s = Swarm(1, mpi_comm, QAOA, system_size)

        s.set_qualities(qualities)
        s.set_depth(2)

        # Setup needs to be called to initialize params properly
        if s.swarm_comms.in_subcomm():
            s.ansatz.setup()
            # total_params should be set after setup
            # QAOA has 2 params per depth (gamma + t)
            expected_n_params = s.ansatz.total_params * 2
            assert expected_n_params == 4  # 2 params * depth 2

        s.destroy()

    def test_swarm_print_result(self, mpi_comm, swarm_system_size):
        """Test that print_result works through swarm."""
        import io
        import sys

        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.meta import Swarm

        system_size = swarm_system_size

        def qualities(local_i, local_i_offset):
            return np.ones(local_i, dtype=np.float64)

        s = Swarm(1, mpi_comm, QAOA, system_size)

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
            sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # On rank 0, should have printed something
        # (other ranks may not print)

        s.destroy()


# =============================================================================
# Tests for Swarm with QMOA (Multivariable)
# =============================================================================


@pytest.mark.mpi
class TestSwarmMultivariable:
    """Tests for swarm with multivariable algorithms."""

    def test_swarm_with_qmoa(self, mpi_comm, swarm_grid_ns_2d):
        """Test swarm initialization with QMOA algorithm."""
        from quop_mpi.algorithm.multivariable import QMOA, cartesian, setup_cartesian
        from quop_mpi.meta import Swarm

        def sphere(x):
            return np.sum(x**2, axis=1)

        Ns = swarm_grid_ns_2d  # noqa: N806
        bounds = [[-1.0, 1.0], [-1.0, 1.0]]
        deltas, mins = setup_cartesian(Ns, bounds)

        s = Swarm(1, mpi_comm, QMOA, Ns)

        s.set_qualities(cartesian, {"args": [deltas, mins, sphere]})
        s.set_depth(1)

        # Evolve with some parameters
        params = np.array([0.1, 0.1, 0.1])
        s.evolve_state(params)

        if s.swarm_comms.in_subcomm():
            probs = s.ansatz.get_probabilities()
            if mpi_comm.Get_rank() == 0:
                total = np.sum(probs)
                assert abs(total - 1.0) < 1e-10

        s.destroy()


# =============================================================================
# Tests for Subcomms Utility Class
# =============================================================================


@pytest.mark.mpi
class TestSubcomms:
    """Tests for the QuopMpiLayout utility class used by swarm."""

    def test_subcomms_in_subcomm(self, mpi_comm):
        """Test in_subcomm returns correct value."""
        from quop_mpi._utils._comm_size import QuopMpiLayout

        # With n_workers=1, all ranks should be in subcomm
        layout = QuopMpiLayout.create_workers(1, mpi_comm)

        # All ranks should be in the subcommunicator
        assert layout.in_subcomm()

        layout.free()

    def test_subcomms_get_n_subcomms(self, mpi_comm):
        """Test get_n_subcomms returns correct count."""
        from quop_mpi._utils._comm_size import QuopMpiLayout

        layout = QuopMpiLayout.create_workers(1, mpi_comm)

        # With n_workers=1, should have 1 subcomm
        n_subcomms = layout.get_n_subcomms()
        assert n_subcomms == 1

        layout.free()

    def test_subcomms_get_subcomm_index(self, mpi_comm):
        """Test get_subcomm_index returns valid index."""
        from quop_mpi._utils._comm_size import QuopMpiLayout

        layout = QuopMpiLayout.create_workers(1, mpi_comm)

        idx = layout.get_subcomm_index()
        # Should be 0 for the single subcomm case
        assert idx == 0

        layout.free()
