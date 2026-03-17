"""
Tests for Ansatz class initialization.

These tests verify that all required attributes are properly initialized
and that the known bugs documented in known_bugs.md are fixed.

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_ansatz_init.py -v --with-mpi
"""

import pytest
from mpi4py import MPI


@pytest.mark.mpi
class TestAnsatzAttributeInitialization:
    """Test that Ansatz initializes all required attributes correctly."""

    def test_ansatz_creation(self, mpi_comm, small_system_size):
        """Basic test that Ansatz can be instantiated."""
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)

        assert alg is not None
        assert alg.system_size == small_system_size

        alg.destroy()

    def test_stop_attribute_initialized(self, mpi_comm, small_system_size):
        """
        Bug #1: self.stop should be initialized to False.

        Previously, the initialization was merged with a comment on line 229.
        """
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)

        assert hasattr(alg, "stop"), "self.stop attribute is missing"
        assert not alg.stop, "self.stop should be initialized to False"

        alg.destroy()

    def test_no_duplicate_optimiser_assignment(self, mpi_comm, small_system_size):
        """
        Bug #6: self.optimiser should only be assigned once.

        This is a code quality check - the attribute should exist and be None.
        """
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)

        assert hasattr(alg, "optimiser"), "self.optimiser attribute is missing"
        assert alg.optimiser is None, "self.optimiser should initialize to None"

        alg.destroy()

    def test_samples_attribute_consistent(self, mpi_comm, small_system_size):
        """
        Bug #7: self.samples should have a consistent initial value.

        Previously it was set to [] then immediately overwritten with None.
        """
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)

        assert hasattr(alg, "samples"), "self.samples attribute is missing"
        # Should be one or the other, not undefined behavior
        assert alg.samples is None or isinstance(alg.samples, list)

        alg.destroy()

    def test_setup_called_initialized_once(self, mpi_comm, small_system_size):
        """
        Bug #8: self.setup_called should only be initialized once.
        """
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)

        assert hasattr(alg, "setup_called"), "self.setup_called attribute is missing"
        assert not alg.setup_called, "self.setup_called should initialize to False"

        alg.destroy()


@pytest.mark.mpi
class TestAnsatzSetupFlags:
    """Test the setup flag system for consistency."""

    def test_all_setup_flags_exist(self, mpi_comm, small_system_size):
        """Verify all setup_* flags are initialized."""
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)

        expected_flags = [
            "setup_log",
            "setup_objective",
            "setup_sampling",
            "setup_var_map",
        ]

        for flag in expected_flags:
            assert hasattr(alg, flag), f"Missing setup flag: {flag}"

        alg.destroy()


@pytest.mark.mpi
class TestAnsatzMPIIntegration:
    """Test MPI-specific initialization."""

    def test_mpi_comm_stored(self, mpi_comm, small_system_size):
        """Test that MPI communicator is properly stored."""
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)

        assert hasattr(alg, "MPI_COMM_WORLD")
        assert alg.MPI_COMM_WORLD is not None
        assert MPI.Comm.Compare(alg.MPI_COMM_WORLD, mpi_comm) in (
            MPI.IDENT,
            MPI.CONGRUENT,
        )

        alg.destroy()

    def test_all_ranks_have_same_system_size(self, mpi_comm, small_system_size):
        """Verify all MPI ranks agree on system size."""
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)

        all_sizes = mpi_comm.allgather(alg.system_size)
        assert all(
            s == small_system_size for s in all_sizes
        ), "All ranks should have the same system_size"

        alg.destroy()
