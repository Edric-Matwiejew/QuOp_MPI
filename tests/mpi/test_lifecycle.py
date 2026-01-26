"""
Tests for Ansatz lifecycle: setup, destroy, and resource management.

These tests verify that the Ansatz properly manages its lifecycle,
including initialization, setup, execution, and cleanup.

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_lifecycle.py -v --with-mpi
"""
import pytest
import numpy as np
from mpi4py import MPI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import TestOracle


@pytest.mark.mpi
class TestSetup:
    """Test the setup() method and related initialization."""

    def test_setup_completes_without_error(self, mpi_comm, simple_oracle):
        """Verify setup() runs to completion."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        # Should complete without error
        alg.setup()
        
        assert alg.setup_called == True

    def test_setup_sets_correct_flags(self, mpi_comm, simple_oracle):
        """Verify setup() properly manages state flags."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        # Before setup
        assert alg.setup_called == False
        
        alg.setup()
        
        # After setup
        assert alg.setup_called == True
        assert alg.reset == False

    def test_setup_can_be_called_multiple_times(self, mpi_comm, simple_oracle):
        """Verify setup() is idempotent."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        # Multiple setup calls should not cause errors
        alg.setup()
        alg.setup()
        alg.setup()
        
        assert alg.setup_called == True

    def test_setup_after_config_change(self, mpi_comm, simple_oracle):
        """Verify setup() works after configuration changes."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        
        # Change configuration
        alg.set_depth(2)
        
        # Setup should work again
        alg.setup()
        
        assert alg.setup_called == True


@pytest.mark.mpi
class TestDestroy:
    """Test the destroy() method and resource cleanup."""

    def test_destroy_before_setup_is_safe(self, mpi_comm, small_system_size):
        """Verify destroy() is safe before setup()."""
        from quop_mpi import Ansatz
        
        alg = Ansatz(small_system_size, mpi_comm)
        
        # Should not raise any errors
        alg.destroy()

    def test_destroy_after_setup(self, mpi_comm, simple_oracle):
        """Verify destroy() works after setup()."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        
        # Should not raise
        alg.destroy()

    def test_destroy_after_evolve(self, mpi_comm, simple_oracle):
        """Verify destroy() works after state evolution."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        
        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)
        
        # Should not raise
        alg.destroy()

    def test_destroy_after_execute(self, mpi_comm, simple_oracle):
        """Verify destroy() works after execute()."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        alg.execute()
        
        # Should not raise
        alg.destroy()

    def test_destroy_can_be_called_multiple_times(self, mpi_comm, simple_oracle):
        """Verify destroy() is idempotent."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()
        
        # Multiple destroy calls should not cause errors
        alg.destroy()
        alg.destroy()
        alg.destroy()


@pytest.mark.mpi
class TestLifecycleSequences:
    """Test various lifecycle sequences."""

    def test_setup_evolve_destroy(self, mpi_comm, simple_oracle):
        """Test setup -> evolve -> destroy sequence."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        alg.setup()
        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)
        alg.destroy()

    def test_multiple_evolve_calls(self, mpi_comm, simple_oracle):
        """Test multiple evolve_state calls in sequence."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        
        params = simple_oracle.optimal_params(depth=1)
        
        # Multiple evolutions should work
        for _ in range(3):
            alg.evolve_state(params)
        
        alg.destroy()

    def test_execute_includes_implicit_setup(self, mpi_comm, simple_oracle):
        """Verify execute() calls setup() implicitly if needed."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        # Don't call setup() explicitly
        assert alg.setup_called == False
        
        alg.execute()
        
        # execute() should have called setup()
        assert alg.setup_called == True

    def test_reinitialize_after_destroy(self, mpi_comm, simple_oracle):
        """Test creating new instance after destroying old one."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        # First instance
        alg1 = qaoa(simple_oracle.system_size, mpi_comm)
        alg1.set_qualities(simple_oracle.qualities_function())
        alg1.set_depth(1)
        alg1.execute()
        
        if mpi_comm.Get_rank() == 0:
            result1 = alg1.result['fun']
        
        alg1.destroy()
        
        # Second instance should work independently
        alg2 = qaoa(simple_oracle.system_size, mpi_comm)
        alg2.set_qualities(simple_oracle.qualities_function())
        alg2.set_depth(1)
        alg2.execute()
        
        if mpi_comm.Get_rank() == 0:
            result2 = alg2.result['fun']
            # Both should have produced valid results
            assert result1 is not None and result2 is not None
        
        alg2.destroy()


@pytest.mark.mpi
class TestDestroyFunctionality:
    """Tests for destroy() method bug #5 from known_bugs.md.
    
    Bug #5: setup_parallel was never set to False, so __post_parallel() was never called.
    
    The destroy() condition `if not self.reset or not self.setup_called: return` is 
    intentional - it skips cleanup when resources are still valid (reset=False means 
    no configuration change since last setup). Cleanup only happens when configuration 
    changes (reset=True) AND setup was called.
    """

    def test_setup_parallel_flag_after_setup(self, mpi_comm, simple_oracle):
        """Verify setup_parallel is set to False after setup().
        
        Bug #5: setup_parallel was never set to False, so the cleanup code path
        was never reachable even when destroy() was called with reset=True.
        """
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        # Before setup, setup_parallel should be True (no parallel resources yet)
        assert alg.setup_parallel == True
        
        alg.setup()
        
        # After setup, setup_parallel should be False (parallel resources allocated)
        assert alg.setup_parallel == False, \
            "Bug #5: setup_parallel should be False after setup() to indicate cleanup needed"

    def test_destroy_calls_post_parallel_on_config_change(self, mpi_comm, simple_oracle):
        """Verify destroy() calls __post_parallel() when configuration changes.
        
        The destroy() condition correctly skips cleanup when reset=False (no config 
        change). Cleanup only occurs when reset=True AND setup_called=True.
        """
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()
        
        # Track if __post_parallel was called
        original_post_parallel = alg._Ansatz__post_parallel
        post_parallel_called = [False]
        
        def mock_post_parallel():
            post_parallel_called[0] = True
            original_post_parallel()
        
        alg._Ansatz__post_parallel = mock_post_parallel
        
        # Trigger configuration change - this sets reset=True
        alg.set_unitaries(alg.unitaries)
        assert alg.reset == True, "set_unitaries should set reset=True"
        
        alg.destroy()
        
        # Verify __post_parallel was called (Bug #5 fix makes this work)
        assert post_parallel_called[0], \
            "Bug #5: __post_parallel() should be called during destroy() when reset=True"
        assert alg.setup_parallel == True, \
            "setup_parallel should be True after cleanup completed"

    def test_destroy_calls_post_unitaries_on_config_change(self, mpi_comm, simple_oracle):
        """Verify destroy() calls __post_unitaries() when configuration changes."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()
        
        # After execute, setup_unitaries should be False (unitaries generated)
        assert alg.setup_unitaries == False, \
            "setup_unitaries should be False after execute()"
        
        # Track if __post_unitaries was called
        original_post_unitaries = alg._Ansatz__post_unitaries
        post_unitaries_called = [False]
        
        def mock_post_unitaries():
            post_unitaries_called[0] = True
            original_post_unitaries()
        
        alg._Ansatz__post_unitaries = mock_post_unitaries
        
        # Trigger configuration change
        alg.set_unitaries(alg.unitaries)
        
        alg.destroy()
        
        # Verify __post_unitaries was called
        assert post_unitaries_called[0], \
            "__post_unitaries() should be called during destroy() when reset=True"

    def test_destroy_skips_cleanup_when_no_config_change(self, mpi_comm, simple_oracle):
        """Verify destroy() skips cleanup when there's no configuration change.
        
        This is intentional behavior - resources are still valid if configuration
        hasn't changed since last setup.
        """
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        
        # No configuration change, so reset=False
        assert alg.reset == False
        
        # Track if cleanup methods were called
        post_parallel_called = [False]
        original_post_parallel = alg._Ansatz__post_parallel
        def mock_post_parallel():
            post_parallel_called[0] = True
            original_post_parallel()
        alg._Ansatz__post_parallel = mock_post_parallel
        
        alg.destroy()
        
        # Cleanup should NOT have been called (reset=False means resources still valid)
        assert not post_parallel_called[0], \
            "destroy() should skip cleanup when reset=False (no config change)"

    def test_subcomms_freed_on_config_change(self, mpi_comm, simple_oracle):
        """Verify MPI subcommunicators are properly freed when config changes."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()
        
        # Track if subcomms.free() was called
        original_free = alg.subcomms.free
        free_called = [False]
        
        def mock_free():
            free_called[0] = True
            original_free()
        
        alg.subcomms.free = mock_free
        
        # Trigger configuration change
        alg.set_unitaries(alg.unitaries)
        
        alg.destroy()
        
        # Verify free() was called (Bug #5 fix makes this work)
        assert free_called[0], \
            "Bug #5: subcomms.free() should be called during destroy() when reset=True"


@pytest.mark.mpi
class TestResourceManagement:
    """Test that resources are properly managed."""

    def test_multiple_instances_independent(self, mpi_comm):
        """Verify multiple Ansatz instances are independent."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        oracle1 = TestOracle(system_size=32, n_marked=2, seed=111)
        oracle2 = TestOracle(system_size=64, n_marked=4, seed=222)
        
        alg1 = qaoa(oracle1.system_size, mpi_comm)
        alg1.set_qualities(oracle1.qualities_function())
        alg1.set_depth(1)
        
        alg2 = qaoa(oracle2.system_size, mpi_comm)
        alg2.set_qualities(oracle2.qualities_function())
        alg2.set_depth(2)
        
        # Both should setup independently
        alg1.setup()
        alg2.setup()
        
        assert alg1.system_size == 32
        assert alg2.system_size == 64
        
        alg1.destroy()
        alg2.destroy()

    def test_sequential_executions(self, mpi_comm, simple_oracle):
        """Test running multiple sequential optimizations."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        results = []
        
        for i in range(3):
            alg = qaoa(simple_oracle.system_size, mpi_comm)
            alg.set_qualities(simple_oracle.qualities_function())
            alg.set_depth(1)
            
            alg.execute()
            
            if mpi_comm.Get_rank() == 0:
                results.append(alg.result['fun'])
            
            alg.destroy()
        
        if mpi_comm.Get_rank() == 0:
            # All executions should produce valid results
            assert len(results) == 3
            assert all(r is not None for r in results)


@pytest.mark.mpi
class TestDelCleanup:
    """Test that `del` properly cleans up resources via __del__."""

    def test_del_before_setup(self, mpi_comm, small_system_size):
        """Verify del is safe before setup()."""
        from quop_mpi import Ansatz
        
        alg = Ansatz(small_system_size, mpi_comm)
        
        # Should not raise any errors
        del alg

    def test_del_after_setup(self, mpi_comm, simple_oracle):
        """Verify del properly cleans up after setup()."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        
        # Should not raise
        del alg

    def test_del_after_evolve(self, mpi_comm, simple_oracle):
        """Verify del properly cleans up after state evolution."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        
        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)
        
        # Should not raise
        del alg

    def test_del_after_execute(self, mpi_comm, simple_oracle):
        """Verify del properly cleans up after execute()."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        
        alg.execute()
        
        # Should not raise
        del alg

    def test_sequential_del_creates_independent_instances(self, mpi_comm, simple_oracle):
        """Test creating new instance after deleting old one."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        # First instance
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()
        
        if mpi_comm.Get_rank() == 0:
            result1 = alg.result['fun']
        else:
            result1 = None
        result1 = mpi_comm.bcast(result1, root=0)
        
        del alg
        
        # Second instance should work independently
        alg = qaoa(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()
        
        if mpi_comm.Get_rank() == 0:
            result2 = alg.result['fun']
        else:
            result2 = None
        result2 = mpi_comm.bcast(result2, root=0)
        
        # Both should have produced valid results
        assert result1 is not None and result2 is not None
        
        del alg
