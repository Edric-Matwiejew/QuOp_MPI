"""
Tests for predefined algorithm classes (QAOA, QWOA, etc.).

These tests verify the algorithm subclasses work correctly and
that bug #3 (undefined self.rank) is addressed.

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_algorithms.py -v --with-mpi
"""

import numpy as np
import pytest


@pytest.mark.mpi
class TestQAOA:
    """Test the QAOA algorithm class."""

    def test_qaoa_creation(self, mpi_comm, small_system_size):
        """Test that QAOA can be instantiated."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(small_system_size, mpi_comm)

        assert alg is not None
        assert alg.system_size == small_system_size

        alg.destroy()

    def test_qaoa_with_qualities(self, mpi_comm, medium_system_size):
        """Test QAOA with quality function defined."""
        import networkx as nx

        from quop_mpi.algorithm.combinatorial import QAOA

        # Keep the graph-derived system size aligned with the active MPI test size.
        n_vertices = max(4, medium_system_size.bit_length() - 1)
        G = nx.complete_graph(n_vertices)  # noqa: N806
        n_vertices = len(G.nodes)
        system_size = 2**n_vertices

        # Simple quality function
        def qualities(local_i, local_i_offset):
            return np.random.RandomState(42 + local_i_offset).random(local_i)

        alg = QAOA(system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert np.isfinite(alg.result["fun"])
            assert len(alg.result["x"]) == alg.n_free_params

        alg.destroy()

    def test_qaoa_error_message_formatting(self, mpi_comm, small_system_size):
        """
        Bug #3: QAOA uses self.rank in error message but it's not defined.

        This test verifies the error path doesn't crash due to missing self.rank.
        """
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(small_system_size, mpi_comm)

        # Don't set qualities - this should trigger the error path
        # The error message uses self.rank which may not be defined

        # We expect a RuntimeError about qualities not being defined
        # NOT an AttributeError about 'rank'
        with pytest.raises(RuntimeError) as excinfo:
            alg.setup()

        # Verify we got the expected error, not an AttributeError
        assert "qualities" in str(excinfo.value).lower() or "Solution" in str(
            excinfo.value
        ), f"Expected quality-related error, got: {excinfo.value}"

        alg.destroy()


@pytest.mark.mpi
class TestQAOAVariants:
    """Test the sparse and transverse-field QAOA variants."""

    def test_qaoa_sparse_creation(self, mpi_comm, small_system_size):
        """Test that QAOASparse can be instantiated."""
        from quop_mpi.algorithm.combinatorial import QAOASparse

        alg = QAOASparse(small_system_size, mpi_comm)

        assert alg is not None
        assert alg.system_size == small_system_size

        alg.destroy()

    def test_qaoa_transverse_field_creation(self, mpi_comm, small_system_size):
        """Test that QAOATransverseField can be instantiated."""
        from quop_mpi.algorithm.combinatorial import QAOATransverseField

        alg = QAOATransverseField(small_system_size, mpi_comm)

        assert alg is not None
        assert alg.system_size == small_system_size

        alg.destroy()

    def test_qaoa_transverse_field_matches_sparse_probabilities(
        self, mpi_comm, small_system_size
    ):
        """The transverse-field QAOA variant should preserve QAOA parameter semantics."""
        from mpi4py import MPI

        from quop_mpi.algorithm.combinatorial import QAOASparse, QAOATransverseField

        def qualities(local_i, local_i_offset):
            return np.cos(np.arange(local_i, dtype=np.float64) + local_i_offset)

        params = np.array([0.23, 0.41], dtype=np.float64)

        sparse_alg = QAOASparse(small_system_size, mpi_comm)
        sparse_alg.set_qualities(qualities)
        sparse_alg.set_depth(1)
        sparse_alg.evolve_state(params)
        sparse_probs = sparse_alg.get_probabilities()

        transverse_field_alg = QAOATransverseField(small_system_size, mpi_comm)
        transverse_field_alg.set_qualities(qualities)
        transverse_field_alg.set_depth(1)
        transverse_field_alg.evolve_state(params)
        transverse_field_probs = transverse_field_alg.get_probabilities()

        did_compare = int(
            sparse_probs is not None and transverse_field_probs is not None
        )

        if sparse_probs is not None and transverse_field_probs is not None:
            np.testing.assert_allclose(
                transverse_field_probs,
                sparse_probs,
                rtol=1e-8,
                atol=1e-10,
            )

        assert mpi_comm.allreduce(did_compare, op=MPI.SUM) == 1

        sparse_alg.destroy()
        transverse_field_alg.destroy()


@pytest.mark.mpi
class TestQWOA:
    """Test the QWOA algorithm class."""

    def test_qwoa_creation(self, mpi_comm, small_system_size):
        """Test that QWOA can be instantiated."""
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(small_system_size, mpi_comm)

        assert alg is not None
        assert alg.system_size == small_system_size

        alg.destroy()

    def test_qwoa_with_qualities(self, mpi_comm, medium_system_size):
        """Test QWOA with quality function defined."""
        from quop_mpi.algorithm.combinatorial import QWOA

        def qualities(local_i, local_i_offset):
            return np.random.RandomState(42 + local_i_offset).random(local_i)

        alg = QWOA(medium_system_size, mpi_comm)
        alg.set_qualities(qualities)
        alg.set_depth(1)

        alg.execute()

        if mpi_comm.Get_rank() == 0:
            assert alg.result is not None
            assert np.isfinite(alg.result["fun"])
            assert len(alg.result["x"]) == alg.n_free_params

        alg.destroy()

    def test_qwoa_error_message_formatting(self, mpi_comm, small_system_size):
        """
        Bug #3: QWOA uses self.rank in error message but it's not defined.
        """
        from quop_mpi.algorithm.combinatorial import QWOA

        alg = QWOA(small_system_size, mpi_comm)

        # Don't set qualities - should trigger error
        with pytest.raises(RuntimeError) as excinfo:
            alg.setup()

        assert "qualities" in str(excinfo.value).lower() or "Solution" in str(
            excinfo.value
        ), f"Expected quality-related error, got: {excinfo.value}"

        alg.destroy()
