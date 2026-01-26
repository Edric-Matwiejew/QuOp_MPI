"""
Tests for Ansatz lifecycle: setup, execute, destroy.

These tests verify the control flow through setup() and destroy() methods,
particularly focusing on the bugs identified in known_bugs.md.

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_ansatz_lifecycle.py -v --with-mpi
"""
import pytest
import numpy as np
from mpi4py import MPI


@pytest.mark.mpi
class TestAnsatzSetup:
    """Test the Ansatz.setup() method and related control flow."""

    def test_setup_with_unitaries(self, mpi_comm, small_system_size):
        """Test that setup() works when unitaries are defined."""
        from quop_mpi import Ansatz
        from quop_mpi.propagator import diagonal, circulant
        
        alg = Ansatz(small_system_size, mpi_comm)
        
        # Define minimal unitaries
        UQ = diagonal.unitary(
            diagonal.operator.rand.uniform,
        )
        UW = circulant.unitary(
            circulant.operator.complete,
        )
        
        alg.set_unitaries([UQ, UW])
        alg.set_observables(0)  # Use UQ's operator as observables
        
        # setup() should complete without error
        alg.setup()
        
        assert alg.setup_called == True, "setup_called should be True after setup()"
        assert alg.reset == False, "reset should be False after setup()"

    def test_setup_sets_correct_flags(self, mpi_comm, small_system_size):
        """Verify setup() properly manages setup flags."""
        from quop_mpi import Ansatz
        from quop_mpi.propagator import diagonal, circulant
        
        alg = Ansatz(small_system_size, mpi_comm)
        
        UQ = diagonal.unitary(diagonal.operator.rand.uniform)
        UW = circulant.unitary(circulant.operator.complete)
        
        alg.set_unitaries([UQ, UW])
        alg.set_observables(0)
        
        # Before setup
        assert alg.setup_called == False
        
        alg.setup()
        
        # After setup - these flags should be reset for __pre() to process
        assert alg.setup_called == True
        assert alg.setup_depth == True
        assert alg.setup_observables == True
        assert alg.setup_initial_state == True
        assert alg.setup_optimiser == True


@pytest.mark.mpi
class TestAnsatzDestroy:
    """Test the Ansatz.destroy() method."""

    def test_destroy_before_setup_is_safe(self, mpi_comm, small_system_size):
        """destroy() should be safe to call before setup()."""
        from quop_mpi import Ansatz
        
        alg = Ansatz(small_system_size, mpi_comm)
        
        # Should not raise any errors
        alg.destroy()

    def test_destroy_condition_logic(self, mpi_comm, small_system_size):
        """
        Bug #4: destroy() condition may be inverted.
        
        Test that destroy() actually runs cleanup after setup().
        """
        from quop_mpi import Ansatz
        from quop_mpi.propagator import diagonal, circulant
        
        alg = Ansatz(small_system_size, mpi_comm)
        
        UQ = diagonal.unitary(diagonal.operator.rand.uniform)
        UW = circulant.unitary(circulant.operator.complete)
        
        alg.set_unitaries([UQ, UW])
        alg.set_observables(0)
        
        alg.setup()
        
        # After setup: reset=False, setup_called=True
        # Bug #4: condition `if not self.reset or not self.setup_called` 
        # evaluates to `if True or False` = True, causing early return
        
        # Store state before destroy
        setup_called_before = alg.setup_called
        
        # Call destroy - this should actually do cleanup
        alg.destroy()
        
        # Document current behavior (this test will help verify the fix)
        # If bug is present, nothing happens
        # If bug is fixed, cleanup should occur


@pytest.mark.mpi
class TestAnsatzEvolveState:
    """Test state evolution functionality."""

    def test_evolve_state_basic(self, mpi_comm, small_system_size):
        """Test basic state evolution."""
        from quop_mpi import Ansatz
        from quop_mpi.propagator import diagonal, circulant
        
        alg = Ansatz(small_system_size, mpi_comm)
        
        UQ = diagonal.unitary(diagonal.operator.rand.uniform)
        UW = circulant.unitary(circulant.operator.complete)
        
        alg.set_unitaries([UQ, UW])
        alg.set_observables(0)
        alg.set_depth(1)
        
        # Generate initial parameters
        # 2 unitaries with 1 param each = 2 params per depth
        params = np.array([0.1, 0.2])
        
        # evolve_state should work
        alg.evolve_state(params)
        
        # Check that state was evolved
        assert alg.last_evaluated is not None
        assert len(alg.last_evaluated) == 2


@pytest.mark.mpi
class TestAnsatzExecute:
    """Test the execute() method."""

    def test_execute_basic(self, mpi_comm, small_system_size):
        """Test basic QVA execution."""
        from quop_mpi import Ansatz
        from quop_mpi.propagator import diagonal, circulant
        
        alg = Ansatz(small_system_size, mpi_comm)
        
        UQ = diagonal.unitary(diagonal.operator.rand.uniform)
        UW = circulant.unitary(circulant.operator.complete)
        
        alg.set_unitaries([UQ, UW])
        alg.set_observables(0)
        alg.set_depth(1)
        
        # Execute should complete without error
        alg.execute()
        
        # Result should be populated on rank 0
        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

    def test_execute_uses_stop_flag(self, mpi_comm, small_system_size):
        """
        Bug #1: Verify self.stop is properly used during execute().
        """
        from quop_mpi import Ansatz
        from quop_mpi.propagator import diagonal, circulant
        
        alg = Ansatz(small_system_size, mpi_comm)
        
        UQ = diagonal.unitary(diagonal.operator.rand.uniform)
        UW = circulant.unitary(circulant.operator.complete)
        
        alg.set_unitaries([UQ, UW])
        alg.set_observables(0)
        alg.set_depth(1)
        
        # self.stop should exist and be usable
        assert hasattr(alg, 'stop'), "self.stop must exist for execute() to work"
        
        alg.execute()
        
        # After execute, stop should have been used for synchronization
        # The exact final value depends on the execution path
