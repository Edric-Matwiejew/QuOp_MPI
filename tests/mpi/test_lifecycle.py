"""
Tests for Ansatz lifecycle: setup, destroy, and resource management.

These tests verify that the Ansatz properly manages its lifecycle,
including initialization, setup, execution, and cleanup.

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_lifecycle.py -v --with-mpi
"""

import gc
import weakref

import pytest
from mpi4py import MPI

from tests.conftest import TestOracle


@pytest.mark.mpi
class TestSetup:
    """Test the setup() method and related initialization."""

    def test_setup_completes_without_error(self, mpi_comm, simple_oracle):
        """Verify setup() runs to completion."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Should complete without error
        alg.setup()

        assert alg.setup_called

        alg.destroy()

    def test_setup_sets_correct_flags(self, mpi_comm, simple_oracle):
        """Verify setup() properly manages state flags."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Before setup
        assert not alg.setup_called

        alg.setup()

        # After setup
        assert alg.setup_called

        alg.destroy()

    def test_setup_can_be_called_multiple_times(self, mpi_comm, simple_oracle):
        """Verify setup() is idempotent."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Multiple setup calls should not cause errors
        alg.setup()
        alg.setup()
        alg.setup()

        assert alg.setup_called

        alg.destroy()

    def test_setup_after_config_change(self, mpi_comm, simple_oracle):
        """Verify setup() works after configuration changes."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()

        # Change configuration
        alg.set_depth(2)

        # Setup should work again
        alg.setup()

        assert alg.setup_called

        alg.destroy()


@pytest.mark.mpi
class TestDestroy:
    """Test the destroy() method and resource cleanup."""

    def test_destroy_before_setup_is_safe(self, mpi_comm, small_system_size):
        """Verify destroy() is safe before setup()."""
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)

        # Should not raise any errors
        alg.destroy()

        assert alg.MPI_COMM_WORLD is not None
        assert MPI.Comm.Compare(alg.MPI_COMM_WORLD, mpi_comm) in (
            MPI.IDENT,
            MPI.CONGRUENT,
        )

    def test_destroy_after_setup(self, mpi_comm, simple_oracle):
        """Verify destroy() works after setup()."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()

        # Should not raise
        alg.destroy()

    def test_destroy_after_evolve(self, mpi_comm, simple_oracle):
        """Verify destroy() works after state evolution."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)

        # Should not raise
        alg.destroy()

    def test_destroy_after_execute(self, mpi_comm, simple_oracle):
        """Verify destroy() works after execute()."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        alg.execute()

        # Should not raise
        alg.destroy()

    def test_destroy_can_be_called_multiple_times(self, mpi_comm, simple_oracle):
        """Verify destroy() is idempotent."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
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
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        alg.setup()
        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)
        alg.destroy()

    def test_multiple_evolve_calls(self, mpi_comm, simple_oracle):
        """Test multiple evolve_state calls in sequence."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
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
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        # Don't call setup() explicitly
        assert not alg.setup_called

        alg.execute()

        # execute() should have called setup()
        assert alg.setup_called

    def test_reinitialize_after_destroy(self, mpi_comm, simple_oracle):
        """Test creating new instance after destroying old one."""
        from quop_mpi.algorithm.combinatorial import QAOA

        # First instance
        alg1 = QAOA(simple_oracle.system_size, mpi_comm)
        alg1.set_qualities(simple_oracle.qualities_function())
        alg1.set_depth(1)
        alg1.execute()

        if mpi_comm.Get_rank() == 0:
            result1 = alg1.result["fun"]

        alg1.destroy()

        # Second instance should work independently
        alg2 = QAOA(simple_oracle.system_size, mpi_comm)
        alg2.set_qualities(simple_oracle.qualities_function())
        alg2.set_depth(1)
        alg2.execute()

        if mpi_comm.Get_rank() == 0:
            result2 = alg2.result["fun"]
            # Both should have produced valid results
            assert result1 is not None and result2 is not None

        alg2.destroy()


@pytest.mark.mpi
class TestDestroyFunctionality:
    """Tests for destroy() method -- end-of-life resource cleanup.

    In the dirty-flag model, destroy() is an unconditional end-of-life
    operation that frees all resources.  It does not depend on
    configuration-change booleans.
    """

    def test_destroy_always_frees_resources(self, mpi_comm, simple_oracle):
        """Verify destroy() always frees resources when setup was completed."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()

        # Track if cleanup methods were called
        post_parallel_called = [False]
        original_post_parallel = alg._post_parallel

        def mock_post_parallel():
            post_parallel_called[0] = True
            original_post_parallel()

        alg._post_parallel = mock_post_parallel

        alg.destroy()

        # End-of-life destroy() ALWAYS frees resources
        assert post_parallel_called[
            0
        ], "destroy() should free resources unconditionally (end-of-life)"

    def test_destroy_resets_dirty_flags(self, mpi_comm, simple_oracle):
        """Verify destroy() resets dirty flags so re-setup fully re-inits."""
        from quop_mpi.algorithm.combinatorial import QAOA
        from quop_mpi.ansatz import _Dirty

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()

        alg.destroy()

        # After destroy, all major dirty bits should be set
        assert alg._dirty & _Dirty.NEGOTIATION
        assert alg._dirty & _Dirty.CONTEXT
        assert alg._dirty & _Dirty.PLANS
        assert not alg.setup_called

    def test_subcomms_freed_on_destroy(self, mpi_comm, simple_oracle):
        """Verify MPI subcommunicators are properly freed on destroy()."""
        from unittest.mock import patch

        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()

        # Trigger configuration change so destroy needs to free the layout
        alg.set_unitaries(alg.unitaries)

        # Patch the layout's free method via the Ansatz _layout attribute
        free_called = [False]
        original_free = alg._layout.free

        def tracking_free():
            free_called[0] = True
            original_free()

        with patch.object(alg, "_layout") as mock_layout:
            mock_layout.free = tracking_free
            # Forward other attribute access to original
            mock_layout.SUBCOMM = alg._layout.SUBCOMM
            mock_layout.in_subcomm.return_value = alg._layout.in_subcomm()
            alg.destroy()

        # Verify free() was called during destroy
        assert free_called[0], "layout.free() should be called during destroy()"


@pytest.mark.mpi
class TestResourceManagement:
    """Test that resources are properly managed."""

    def test_multiple_instances_independent(self, mpi_comm):
        """Verify multiple Ansatz instances are independent."""
        from quop_mpi.algorithm.combinatorial import QAOA

        oracle1 = TestOracle(system_size=32, n_marked=2, seed=111)
        oracle2 = TestOracle(system_size=64, n_marked=4, seed=222)

        alg1 = QAOA(oracle1.system_size, mpi_comm)
        alg1.set_qualities(oracle1.qualities_function())
        alg1.set_depth(1)

        alg2 = QAOA(oracle2.system_size, mpi_comm)
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
        from quop_mpi.algorithm.combinatorial import QAOA

        results = []

        for _ in range(3):
            alg = QAOA(simple_oracle.system_size, mpi_comm)
            alg.set_qualities(simple_oracle.qualities_function())
            alg.set_depth(1)

            alg.execute()

            if mpi_comm.Get_rank() == 0:
                results.append(alg.result["fun"])

            alg.destroy()

        if mpi_comm.Get_rank() == 0:
            # All executions should produce valid results
            assert len(results) == 3
            assert all(r is not None for r in results)


@pytest.mark.mpi
class TestDelCleanup:
    """Test that ``del`` is inert and explicit ``destroy()`` owns cleanup."""

    def test_del_before_setup(self, mpi_comm, small_system_size):
        """Verify del is safe before setup()."""
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)

        # Should not raise any errors
        del alg

    def test_del_releases_instance_before_exit(self, mpi_comm, small_system_size):
        """Verify ``del`` does not keep the instance artificially alive."""
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)
        alg_ref = weakref.ref(alg)

        del alg
        gc.collect()

        assert alg_ref() is None

    def test_del_does_not_call_destroy(self, mpi_comm, small_system_size, monkeypatch):
        """Verify ``__del__`` never delegates to ``destroy()``."""
        from quop_mpi import Ansatz

        alg = Ansatz(small_system_size, mpi_comm)
        destroy_called = [False]

        def fake_destroy():
            destroy_called[0] = True

        monkeypatch.setattr(alg, "destroy", fake_destroy)

        alg.__del__()

        assert destroy_called[0] is False

    def test_del_after_setup(self, mpi_comm, simple_oracle):
        """Verify ``del`` is safe after explicit destroy following setup()."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()
        alg.destroy()

        # Should not raise
        del alg

    def test_del_after_evolve(self, mpi_comm, simple_oracle):
        """Verify ``del`` is safe after explicit destroy following evolve()."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.setup()

        params = simple_oracle.optimal_params(depth=1)
        alg.evolve_state(params)
        alg.destroy()

        # Should not raise
        del alg

    def test_del_after_execute(self, mpi_comm, simple_oracle):
        """Verify ``del`` is safe after explicit destroy following execute()."""
        from quop_mpi.algorithm.combinatorial import QAOA

        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)

        alg.execute()
        alg.destroy()

        # Should not raise
        del alg

    def test_sequential_del_creates_independent_instances(self, mpi_comm, simple_oracle):
        """Test creating new instance after explicit destroy and deletion."""
        from quop_mpi.algorithm.combinatorial import QAOA

        # First instance
        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()

        if mpi_comm.Get_rank() == 0:
            result1 = alg.result["fun"]
        else:
            result1 = None
        result1 = mpi_comm.bcast(result1, root=0)

        alg.destroy()
        del alg

        # Second instance should work independently
        alg = QAOA(simple_oracle.system_size, mpi_comm)
        alg.set_qualities(simple_oracle.qualities_function())
        alg.set_depth(1)
        alg.execute()

        if mpi_comm.Get_rank() == 0:
            result2 = alg.result["fun"]
        else:
            result2 = None
        result2 = mpi_comm.bcast(result2, root=0)

        # Both should have produced valid results
        assert result1 is not None and result2 is not None

        alg.destroy()
        del alg


@pytest.mark.mpi
class TestContextManagerCleanup:
    """Test deterministic cleanup via the context-manager exit path."""

    def test_with_block_destroys_after_execute(self, mpi_comm, simple_oracle):
        """Verify ``with`` delegates to ``destroy()`` on normal exit."""
        from quop_mpi.algorithm.combinatorial import QAOA

        with QAOA(simple_oracle.system_size, mpi_comm) as alg:
            alg.set_qualities(simple_oracle.qualities_function())
            alg.set_depth(1)
            alg.execute()

            assert alg.MPI_COMM_WORLD is not None

        assert alg.MPI_COMM_WORLD is not None
        assert alg.context is None
        assert alg.layout is None

    def test_with_block_destroys_on_exception(self, mpi_comm, small_system_size):
        """Verify ``with`` still destroys resources when the body raises."""
        from quop_mpi import Ansatz

        with pytest.raises(RuntimeError, match="expected lifecycle failure"):
            with Ansatz(small_system_size, mpi_comm) as alg:
                raise RuntimeError("expected lifecycle failure")

        assert alg.MPI_COMM_WORLD is not None
