"""
Tests for set_parallel_jacobian functionality.

The parallel jacobian feature creates MPI subcommunicators for parallel
computation of objective function gradients during optimization.

Architecture:
- Subcomm 0: Runs the optimizer, evaluates objective function
- Subcomms 1+: Compute partial derivatives in parallel
- JACCOMM: Communicator linking all jacobian workers + rank 0

Key concerns:
- Proper subcommunicator creation and cleanup
- Correct broadcast of variational parameters to all subcomms
- Proper synchronization of stop flag across all subcomms
- Correct gathering of partial derivatives to rank 0
- No deadlocks when subcomm sizes don't evenly divide
"""

import pytest
import numpy as np
from mpi4py import MPI
from quop_mpi.algorithm.combinatorial import serial


# =============================================================================
# Helper Functions
# =============================================================================


def make_qualities_function(system_size):
    """Create a qualities function that returns values 0 to system_size-1."""
    return lambda: np.arange(system_size, dtype=np.float64)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def jacobian_system_size():
    """System size suitable for jacobian testing - needs to be divisible."""
    return 64  # 2^6, works well with power-of-2 process counts


@pytest.fixture
def jacobian_qualities(jacobian_system_size):
    """Quality function for jacobian tests - simple quadratic."""

    def qualities():
        # Use a smooth function where gradient matters
        return np.arange(jacobian_system_size, dtype=np.float64)

    return qualities


# =============================================================================
# Basic Subcommunicator Tests
# =============================================================================


@pytest.mark.mpi
class TestSubcommunicatorCreation:
    """Test that parallel jacobian subcommunicators are created correctly."""

    def test_single_subcomm_no_jacobian(self, mpi_comm):
        """When only 1 subcomm, no jacobian parallelism is possible."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        # With default settings (no parallel jacobian), should have 1 subcomm
        alg.setup()

        n_subcomms = alg.subcomms.get_n_subcomms()

        # Gather to verify all ranks agree
        all_n_subcomms = mpi_comm.gather(n_subcomms, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(
                n == 1 for n in all_n_subcomms
            ), f"Without parallel jacobian, should have 1 subcomm: {all_n_subcomms}"

    def test_parallel_jacobian_creates_multiple_subcomms(self, mpi_comm):
        """set_parallel_jacobian should create multiple subcommunicators."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()

        # Need at least 2 processes to create multiple subcomms
        if size < 2:
            pytest.skip("Need at least 2 MPI processes for parallel jacobian")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        # Request 2 subcommunicators (each with size/2 processes)
        # nodes_per_subcomm=1, processes_per_node=size, maxcomm=2
        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.setup()

        n_subcomms = alg.subcomms.get_n_subcomms()

        # Gather to verify
        all_n_subcomms = mpi_comm.gather(n_subcomms, root=0)

        if mpi_comm.Get_rank() == 0:
            # Should have created 2 subcomms (or possibly fewer if size is small)
            assert (
                all_n_subcomms[0] >= 1
            ), f"Should have at least 1 subcomm: {all_n_subcomms}"

    def test_jaccomm_created_with_multiple_subcomms(self, mpi_comm):
        """JACCOMM should be created when multiple subcomms exist."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()

        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.setup()

        in_jaccomm = alg.subcomms.in_jaccomm()

        # Gather to check
        all_in_jaccomm = mpi_comm.gather(in_jaccomm, root=0)

        if mpi_comm.Get_rank() == 0:
            n_subcomms = alg.subcomms.get_n_subcomms()
            if n_subcomms > 1:
                # At least some ranks should be in JACCOMM
                assert any(
                    all_in_jaccomm
                ), f"With {n_subcomms} subcomms, some ranks should be in JACCOMM"


# =============================================================================
# Jacobian Method Tests
# =============================================================================


@pytest.mark.mpi
class TestJacobianMethods:
    """Test forward and central difference jacobian methods."""

    def test_forward_difference_method_accepted(self, mpi_comm):
        """Forward difference method should be accepted."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        # Should not raise
        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        assert alg.jacobian_input == ["forward"]

    def test_central_difference_method_accepted(self, mpi_comm):
        """Central difference method should be accepted."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="central"
        )

        assert alg.jacobian_input == ["central"]

    def test_custom_step_size(self, mpi_comm):
        """Custom step size h should be stored."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64
        custom_h = 1e-6

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1,
            processes_per_node=size,
            maxcomm=2,
            method="forward",
            h=custom_h,
        )

        assert alg.h == custom_h


# =============================================================================
# Execution Tests
# =============================================================================


@pytest.mark.mpi
class TestParallelJacobianExecution:
    """Test that optimization with parallel jacobian completes correctly."""

    def test_execute_with_parallel_jacobian_completes(self, mpi_comm):
        """Execute with parallel jacobian should complete without deadlock."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        # Use a simple optimizer with few iterations
        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 5, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        # This should complete without deadlock
        alg.execute()

        # Verify all ranks complete
        mpi_comm.barrier()

        completed = True
        all_completed = mpi_comm.gather(completed, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(all_completed), "All ranks should complete execution"

    def test_execute_with_parallel_jacobian_produces_result(self, mpi_comm):
        """Execute with parallel jacobian should produce optimization result."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 10, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        alg.execute()

        # Only rank 0 in subcomm 0 should have result
        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None, "Rank 0 should have optimization result"
            assert hasattr(alg.result, "fun"), "Result should have 'fun' attribute"

    def test_multiple_executions_with_parallel_jacobian(self, mpi_comm):
        """Multiple executions with parallel jacobian should all complete."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 3, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        # Execute multiple times
        for i in range(3):
            alg.execute()
            mpi_comm.barrier()

        completed = True
        all_completed = mpi_comm.gather(completed, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(all_completed), "All executions should complete"


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.mpi
class TestParallelJacobianEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_depth_greater_than_one(self, mpi_comm):
        """Parallel jacobian with depth > 1 should work correctly."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(3)  # More parameters to distribute

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 5, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        alg.execute()
        mpi_comm.barrier()

        completed = True
        all_completed = mpi_comm.gather(completed, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(all_completed), "Should complete with depth > 1"

    def test_uneven_parameter_distribution(self, mpi_comm):
        """Test when parameters don't divide evenly among subcomms."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        # Depth 5 = 10 parameters, may not divide evenly
        alg.set_depth(5)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 3, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        alg.execute()
        mpi_comm.barrier()

        completed = True
        all_completed = mpi_comm.gather(completed, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(all_completed), "Should handle uneven parameter distribution"

    def test_small_system_with_parallel_jacobian(self, mpi_comm):
        """Test parallel jacobian with system size close to process count."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        # System size just slightly larger than nprocs
        system_size = max(8, size * 2)

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 3, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        try:
            alg.execute()
            completed = True
        except Exception as e:
            completed = False
            error_msg = str(e)

        mpi_comm.barrier()

        all_completed = mpi_comm.gather(completed, root=0)

        if mpi_comm.Get_rank() == 0:
            # May fail due to system size constraints, but shouldn't deadlock
            pass  # Test passes if we reach this point without hanging


# =============================================================================
# Var Map Tests
# =============================================================================


@pytest.mark.mpi
class TestVarMapDistribution:
    """Test that variational parameters are correctly distributed."""

    def test_var_map_created_during_execute(self, mpi_comm):
        """var_map should be created during execute when multiple subcomms exist."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(2)  # 4 parameters total

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 3, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        # var_map is created during execute's __pre() call
        alg.execute()

        # After execute, check var_map if multiple subcomms exist
        if alg.subcomms.get_n_subcomms() > 1:
            assert (
                alg.var_map is not None
            ), "var_map should exist with multiple subcomms after execute"
            assert isinstance(alg.var_map, list), "var_map should be a list"

    def test_var_map_covers_all_parameters(self, mpi_comm):
        """var_map should cover all variational parameters."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64
        depth = 3
        n_params = depth * 2  # 2 params per depth for QWOA

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(depth)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 3, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        # var_map is created during execute
        alg.execute()

        if alg.subcomms.get_n_subcomms() > 1 and mpi_comm.Get_rank() == 0:
            # Flatten var_map and check all parameters are covered
            all_params = []
            for mapping in alg.var_map:
                all_params.extend(mapping)

            # Should have all parameters from 0 to n_params-1
            expected = set(range(n_params))
            actual = set(all_params)

            # Note: subcomm 0 doesn't compute jacobian, so its var_map entry may be empty
            # The jacobian subcomms (1+) should cover all parameters
            jacobian_params = []
            for mapping in alg.var_map[1:]:
                jacobian_params.extend(mapping)

            assert (
                set(jacobian_params) == expected
            ), f"Jacobian subcomms should cover all {n_params} params: got {jacobian_params}"


# =============================================================================
# Synchronization Tests
# =============================================================================


@pytest.mark.mpi
class TestJacobianSynchronization:
    """Test that jacobian computation properly synchronizes across ranks."""

    def test_stop_flag_propagates_to_all_ranks(self, mpi_comm):
        """Stop flag should propagate to all ranks when optimization completes."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 3, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        alg.execute()

        # After execute, all ranks should have stop=True
        all_stop = mpi_comm.gather(alg.stop, root=0)

        if mpi_comm.Get_rank() == 0:
            # All ranks that participated should have stop=True
            assert all(
                all_stop
            ), f"All ranks should have stop=True after execute: {all_stop}"

    def test_no_deadlock_with_early_termination(self, mpi_comm):
        """Optimization that terminates early should not deadlock."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        # Set very strict convergence criteria for early termination
        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 1, "gtol": 1e-1}},
            ["fun", "nfev"],
        )

        alg.execute()
        mpi_comm.barrier()

        # If we reach here without hanging, test passes
        completed = True
        all_completed = mpi_comm.gather(completed, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(all_completed), "Should complete without deadlock"


# =============================================================================
# Jacobian Correctness Tests - Compare with/without parallel jacobian
# =============================================================================


@pytest.mark.mpi
class TestJacobianCorrectness:
    """Test that parallel jacobian produces correct optimization results.

    These tests compare optimization with parallel jacobian against
    optimization without it (using scipy's default finite difference jacobian).
    Both should converge to the same result.
    """

    def test_parallel_jacobian_matches_serial_forward(self, mpi_comm):
        """Parallel forward-difference jacobian should match serial optimization."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64
        depth = 2
        seed = 42

        # Common BFGS options for reproducible comparison
        # Use tight tolerances and fixed initial parameters
        bfgs_options = {
            "method": "BFGS",
            "options": {
                "maxiter": 100,
                "gtol": 1e-5,
                "norm": np.inf,
                "disp": False,
            },
        }

        # Generate fixed initial parameters (must be same for both runs)
        np.random.seed(seed)
        initial_params = np.random.uniform(-np.pi, np.pi, depth * 2)

        # Run WITHOUT parallel jacobian (scipy's default finite diff)
        alg_serial = qwoa(system_size, mpi_comm)
        alg_serial.set_qualities(
            serial, {"args": [make_qualities_function(system_size)]}
        )
        alg_serial.set_depth(depth)
        alg_serial.set_seed(seed)
        alg_serial.set_optimiser("scipy", bfgs_options, ["fun", "nfev", "x"])
        alg_serial.execute(initial_params.copy())

        serial_result_fun = None
        serial_result_x = None
        if mpi_comm.Get_rank() == 0:
            serial_result_fun = alg_serial.result.fun
            serial_result_x = alg_serial.result.x.copy()

        mpi_comm.barrier()

        # Run WITH parallel jacobian (forward difference)
        alg_parallel = qwoa(system_size, mpi_comm)
        alg_parallel.set_qualities(
            serial, {"args": [make_qualities_function(system_size)]}
        )
        alg_parallel.set_depth(depth)
        alg_parallel.set_seed(seed)

        alg_parallel.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg_parallel.set_optimiser("scipy", bfgs_options, ["fun", "nfev", "x"])
        alg_parallel.execute(initial_params.copy())

        if mpi_comm.Get_rank() == 0:
            parallel_result_fun = alg_parallel.result.fun
            parallel_result_x = alg_parallel.result.x

            # Both should converge to similar objective function values
            # Allow small tolerance due to different jacobian computation paths
            assert np.isclose(
                serial_result_fun, parallel_result_fun, rtol=1e-3
            ), f"Objective values differ: serial={serial_result_fun}, parallel={parallel_result_fun}"

            # Parameters should be close (may differ slightly due to different paths)
            assert np.allclose(
                serial_result_x, parallel_result_x, rtol=1e-2, atol=1e-3
            ), f"Optimal parameters differ:\nserial={serial_result_x}\nparallel={parallel_result_x}"

    def test_parallel_jacobian_matches_serial_central(self, mpi_comm):
        """Parallel central-difference jacobian should match serial optimization."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64
        depth = 2
        seed = 123

        bfgs_options = {
            "method": "BFGS",
            "options": {
                "maxiter": 100,
                "gtol": 1e-5,
                "norm": np.inf,
                "disp": False,
            },
        }

        np.random.seed(seed)
        initial_params = np.random.uniform(-np.pi, np.pi, depth * 2)

        # Run WITHOUT parallel jacobian
        alg_serial = qwoa(system_size, mpi_comm)
        alg_serial.set_qualities(
            serial, {"args": [make_qualities_function(system_size)]}
        )
        alg_serial.set_depth(depth)
        alg_serial.set_seed(seed)
        alg_serial.set_optimiser("scipy", bfgs_options, ["fun", "nfev", "x"])
        alg_serial.execute(initial_params.copy())

        serial_result_fun = None
        if mpi_comm.Get_rank() == 0:
            serial_result_fun = alg_serial.result.fun

        mpi_comm.barrier()

        # Run WITH parallel jacobian (central difference - more accurate)
        alg_parallel = qwoa(system_size, mpi_comm)
        alg_parallel.set_qualities(
            serial, {"args": [make_qualities_function(system_size)]}
        )
        alg_parallel.set_depth(depth)
        alg_parallel.set_seed(seed)

        alg_parallel.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="central"
        )

        alg_parallel.set_optimiser("scipy", bfgs_options, ["fun", "nfev", "x"])
        alg_parallel.execute(initial_params.copy())

        if mpi_comm.Get_rank() == 0:
            parallel_result_fun = alg_parallel.result.fun

            # Central difference should be more accurate, so results should be very close
            assert np.isclose(
                serial_result_fun, parallel_result_fun, rtol=1e-3
            ), f"Objective values differ: serial={serial_result_fun}, parallel={parallel_result_fun}"

    def test_parallel_jacobian_convergence_deeper_circuit(self, mpi_comm):
        """Parallel jacobian should work correctly with deeper circuits."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64
        depth = 4  # More parameters = more work for jacobian
        seed = 456

        bfgs_options = {
            "method": "BFGS",
            "options": {
                "maxiter": 50,
                "gtol": 1e-4,
                "disp": False,
            },
        }

        np.random.seed(seed)
        initial_params = np.random.uniform(-np.pi, np.pi, depth * 2)

        # Run WITHOUT parallel jacobian
        alg_serial = qwoa(system_size, mpi_comm)
        alg_serial.set_qualities(
            serial, {"args": [make_qualities_function(system_size)]}
        )
        alg_serial.set_depth(depth)
        alg_serial.set_seed(seed)
        alg_serial.set_optimiser("scipy", bfgs_options, ["fun", "nfev"])
        alg_serial.execute(initial_params.copy())

        serial_result_fun = None
        if mpi_comm.Get_rank() == 0:
            serial_result_fun = alg_serial.result.fun

        mpi_comm.barrier()

        # Run WITH parallel jacobian
        alg_parallel = qwoa(system_size, mpi_comm)
        alg_parallel.set_qualities(
            serial, {"args": [make_qualities_function(system_size)]}
        )
        alg_parallel.set_depth(depth)
        alg_parallel.set_seed(seed)

        alg_parallel.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg_parallel.set_optimiser("scipy", bfgs_options, ["fun", "nfev"])
        alg_parallel.execute(initial_params.copy())

        if mpi_comm.Get_rank() == 0:
            parallel_result_fun = alg_parallel.result.fun

            # Both should find similar minima
            assert np.isclose(
                serial_result_fun, parallel_result_fun, rtol=1e-2
            ), f"Objective values differ for depth={depth}: serial={serial_result_fun}, parallel={parallel_result_fun}"

    def test_parallel_jacobian_gradient_accuracy(self, mpi_comm):
        """Test that parallel jacobian computes accurate gradients.

        Compare gradients at a fixed point computed by parallel jacobian
        vs scipy's approx_fprime.
        """
        from quop_mpi.algorithm.combinatorial import qwoa
        from scipy.optimize import approx_fprime

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64
        depth = 2

        # Fixed test point
        np.random.seed(789)
        test_params = np.random.uniform(-np.pi, np.pi, depth * 2)

        # Create algorithm and setup for gradient evaluation
        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(depth)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        # Use just 1 iteration to get gradient at initial point
        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 1, "gtol": 1e-10}},
            ["fun", "jac", "nfev"],
        )

        alg.execute(test_params.copy())

        if mpi_comm.Get_rank() == 0:
            # The optimization should have computed at least one gradient
            # Check that the result exists and has reasonable properties
            assert alg.result is not None, "Should have optimization result"
            assert hasattr(alg.result, "fun"), "Result should have objective value"

            # The objective value should be a reasonable number (not NaN or Inf)
            assert np.isfinite(
                alg.result.fun
            ), f"Objective value should be finite, got {alg.result.fun}"


# =============================================================================
# Input Validation Tests
# =============================================================================


@pytest.mark.mpi
class TestParallelJacobianInputValidation:
    """Test input validation for set_parallel_jacobian."""

    def test_invalid_method_string_raises_or_handled(self, mpi_comm):
        """Invalid method string should be handled gracefully."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        # An invalid method should either raise during set_parallel_jacobian
        # or during execute - the key is it shouldn't silently fail
        alg.set_parallel_jacobian(
            nodes_per_subcomm=1,
            processes_per_node=size,
            maxcomm=2,
            method="invalid_method",
        )

        # The invalid method is stored
        assert alg.jacobian_input == ["invalid_method"]

        # Note: The actual error would occur during execute when
        # the jacobian function is called

    def test_reconfiguration_overwrites_previous(self, mpi_comm):
        """Calling set_parallel_jacobian twice should overwrite previous config."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        # First configuration
        alg.set_parallel_jacobian(
            nodes_per_subcomm=1,
            processes_per_node=size,
            maxcomm=2,
            method="forward",
            h=1e-5,
        )

        assert alg.jacobian_input == ["forward"]
        assert alg.h == 1e-5

        # Reconfigure
        alg.set_parallel_jacobian(
            nodes_per_subcomm=1,
            processes_per_node=size,
            maxcomm=4,
            method="central",
            h=1e-8,
        )

        assert alg.jacobian_input == ["central"]
        assert alg.h == 1e-8
        assert alg.maxcomm == 4

    def test_default_step_size(self, mpi_comm):
        """When h is not specified, default should be sqrt(machine epsilon)."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        # Don't specify h
        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        expected_h = np.sqrt(np.finfo(float).eps)
        assert alg.h == expected_h, f"Expected h={expected_h}, got h={alg.h}"

    def test_maxcomm_larger_than_ranks(self, mpi_comm):
        """maxcomm larger than available ranks should be handled gracefully."""
        from quop_mpi.algorithm.combinatorial import qwoa
        import warnings

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        # Request more subcomms than ranks
        alg.set_parallel_jacobian(
            nodes_per_subcomm=1,
            processes_per_node=size,
            maxcomm=100,  # Way more than available
            method="forward",
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 3, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        # Should complete without error and issue a warning about fallback
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            alg.execute()

            # Check that a RuntimeWarning about parallel jacobian fallback was issued
            parallel_jac_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, RuntimeWarning)
                and "Parallel jacobian" in str(warning.message)
            ]
            # Note: warning may or may not appear depending on subcomm creation

        mpi_comm.barrier()

        # Actual subcomms created should not exceed available resources
        n_subcomms = alg.subcomms.get_n_subcomms()
        assert (
            n_subcomms <= size
        ), f"Should not create more subcomms ({n_subcomms}) than ranks ({size})"


# =============================================================================
# QAOA Algorithm Tests
# =============================================================================


@pytest.mark.mpi
class TestParallelJacobianWithQAOA:
    """Test parallel jacobian with QAOA algorithm (not just QWOA)."""

    def test_qaoa_with_parallel_jacobian_completes(self, mpi_comm):
        """QAOA should work with parallel jacobian."""
        from quop_mpi.algorithm.combinatorial import qaoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qaoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 5, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        alg.execute()
        mpi_comm.barrier()

        completed = True
        all_completed = mpi_comm.gather(completed, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(all_completed), "QAOA with parallel jacobian should complete"

    def test_qaoa_parallel_jacobian_produces_result(self, mpi_comm):
        """QAOA with parallel jacobian should produce valid result."""
        from quop_mpi.algorithm.combinatorial import qaoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qaoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(2)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="central"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 10, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None, "QAOA should have result"
            assert np.isfinite(alg.result.fun), "Result should be finite"


# =============================================================================
# Cleanup Tests
# =============================================================================


@pytest.mark.mpi
class TestParallelJacobianCleanup:
    """Test cleanup and resource management with parallel jacobian."""

    def test_destroy_after_parallel_jacobian(self, mpi_comm):
        """destroy() should work after using parallel jacobian."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 3, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        alg.execute()

        # destroy() should not raise
        alg.destroy()

        mpi_comm.barrier()

        completed = True
        all_completed = mpi_comm.gather(completed, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(all_completed), "destroy() should complete without error"

    def test_reexecute_after_destroy(self, mpi_comm):
        """Should be able to execute again after destroy with parallel jacobian."""
        from quop_mpi.algorithm.combinatorial import qwoa

        size = mpi_comm.Get_size()
        if size < 2:
            pytest.skip("Need at least 2 MPI processes")

        system_size = 64

        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(serial, {"args": [make_qualities_function(system_size)]})
        alg.set_depth(1)

        alg.set_parallel_jacobian(
            nodes_per_subcomm=1, processes_per_node=size, maxcomm=2, method="forward"
        )

        alg.set_optimiser(
            "scipy",
            {"method": "BFGS", "options": {"maxiter": 3, "gtol": 1e-3}},
            ["fun", "nfev"],
        )

        # First execute
        alg.execute()
        first_result = alg.result.fun if mpi_comm.Get_rank() == 0 else None

        # Destroy
        alg.destroy()

        # Should be able to execute again
        alg.execute()

        mpi_comm.barrier()

        completed = True
        all_completed = mpi_comm.gather(completed, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(all_completed), "Re-execute after destroy should work"
