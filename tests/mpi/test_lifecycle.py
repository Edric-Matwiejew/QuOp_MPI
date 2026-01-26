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
