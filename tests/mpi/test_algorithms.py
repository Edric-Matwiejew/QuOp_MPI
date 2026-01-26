"""
Tests for predefined algorithm classes (QAOA, QWOA, etc.).

These tests verify the algorithm subclasses work correctly and
that bug #3 (undefined self.rank) is addressed.

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_algorithms.py -v --with-mpi
"""
import pytest
import numpy as np
from mpi4py import MPI


@pytest.mark.mpi
class TestQAOA:
    """Test the QAOA algorithm class."""

    def test_qaoa_creation(self, mpi_comm):
        """Test that QAOA can be instantiated."""
        from quop_mpi.algorithm.combinatorial import qaoa
        
        system_size = 16
        alg = qaoa(system_size, mpi_comm)
        
        assert alg is not None
        assert alg.system_size == system_size

    def test_qaoa_with_qualities(self, mpi_comm):
        """Test QAOA with quality function defined."""
        from quop_mpi.algorithm.combinatorial import qaoa
        import networkx as nx
        
        # Create a simple graph
        G = nx.complete_graph(4)
        n_vertices = len(G.nodes)
        system_size = 2 ** n_vertices
        
        # Simple quality function
        def qualities(local_i, local_i_offset):
            return np.random.rand(local_i)
        
        alg = qaoa(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        
        alg.execute()
        
        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

    def test_qaoa_error_message_formatting(self, mpi_comm):
        """
        Bug #3: QAOA uses self.rank in error message but it's not defined.
        
        This test verifies the error path doesn't crash due to missing self.rank.
        """
        from quop_mpi.algorithm.combinatorial import qaoa
        
        system_size = 16
        alg = qaoa(system_size, mpi_comm)
        
        # Don't set qualities - this should trigger the error path
        # The error message uses self.rank which may not be defined
        
        # We expect a RuntimeError about qualities not being defined
        # NOT an AttributeError about 'rank'
        with pytest.raises(RuntimeError) as excinfo:
            alg.setup()
        
        # Verify we got the expected error, not an AttributeError
        assert "qualities" in str(excinfo.value).lower() or \
               "Solution" in str(excinfo.value), \
               f"Expected quality-related error, got: {excinfo.value}"


@pytest.mark.mpi
class TestQWOA:
    """Test the QWOA algorithm class."""

    def test_qwoa_creation(self, mpi_comm):
        """Test that QWOA can be instantiated."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        system_size = 16
        alg = qwoa(system_size, mpi_comm)
        
        assert alg is not None
        assert alg.system_size == system_size

    def test_qwoa_with_qualities(self, mpi_comm):
        """Test QWOA with quality function defined."""
        from quop_mpi.algorithm.combinatorial import qwoa
        
        system_size = 16
        
        def qualities(local_i, local_i_offset):
            return np.random.rand(local_i)
        
        alg = qwoa(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)
        
        alg.execute()
        
        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None

    def test_qwoa_error_message_formatting(self, mpi_comm):
        """
        Bug #3: QWOA uses self.rank in error message but it's not defined.
        """
        from quop_mpi.algorithm.combinatorial import qwoa
        
        system_size = 16
        alg = qwoa(system_size, mpi_comm)
        
        # Don't set qualities - should trigger error
        with pytest.raises(RuntimeError) as excinfo:
            alg.setup()
        
        assert "qualities" in str(excinfo.value).lower() or \
               "Solution" in str(excinfo.value), \
               f"Expected quality-related error, got: {excinfo.value}"
