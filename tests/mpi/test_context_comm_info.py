"""
Stage 3 tests -- Context reads from quop_mpi_layout_t.

Verifies that the ``context`` class accepts a ``QuopMpiLayout`` handle and
correctly allocates state/observable arrays through the Fortran
``context_wrapper.setup(ci_ptr)`` path.

Run with:
    mpiexec -n 2 python -m pytest tests/mpi/test_context_comm_info.py -v --with-mpi --backend mpi
"""

from importlib import import_module

import numpy as np
import pytest
from mpi4py import MPI

from quop_mpi import config
from quop_mpi._lib.context import Context
from quop_mpi._utils._comm_size import QuopMpiLayout

# -- helpers ------------------------------------------------------------


def _get_backend():
    """Load the backend module from the current config."""
    return import_module(f"quop_mpi._lib.{config.backend}")


def _block_partition(system_size, size, rank):
    """Return (local_i, offset) for a standard block distribution."""
    base = system_size // size
    remainder = system_size % size
    local_i = base + (1 if rank < remainder else 0)
    offset = sum(base + (1 if r < remainder else 0) for r in range(rank))
    return local_i, offset


def _make_layout(system_size, mpi_comm):
    """Create a correctly-populated QuopMpiLayout via the real negotiate path.

    Uses create_workers + negotiate with zero propagators, which is the
    same path the Ansatz takes.  This ensures SUBCOMM and NODECOMM are
    initialised on all backends, and DEVCOMM/DEVCOMM_NODE are initialised
    on wavefront.
    """
    from quop_mpi._lib.comm_info_wrapper import comm_info_wrapper as _ciw

    backend_flag = 1 if config.backend == "wavefront" else 0

    layout = QuopMpiLayout.create_workers(1, mpi_comm)  # backend auto-detected
    split_ptr = layout.split_ptr
    topo_ptr = layout.topo_ptr

    prop_ptrs = np.array([], dtype=np.int64)
    cb_ptrs = np.array([], dtype=np.int64)
    layout_ptr, status = _ciw.wrapper_negotiate(
        split_ptr,
        topo_ptr,
        np.int64(system_size),
        np.int32(backend_flag),
        prop_ptrs,
        cb_ptrs,
    )
    layout_ptr = int(layout_ptr)
    if status not in (0, -1):
        raise RuntimeError(f"Fortran negotiate failed with status {status}")
    layout.set_layout_ptr(layout_ptr)
    if status == -1:
        layout.mark_excluded()
    return layout


# -- tests --------------------------------------------------------------


@pytest.mark.mpi
class TestContextFromLayout:
    """T3.1: Create context from locked QuopMpiLayout -> state has correct size."""

    def test_layout_nodecomm_matches_shared_memory_group(self, mpi_comm):
        """Negotiated layouts expose the node-local communicator on all backends."""
        system_size = max(32, mpi_comm.Get_size())
        layout = _make_layout(system_size, mpi_comm)
        expected_nodecomm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)

        try:
            assert layout.nodecomm is not None
            assert layout.nodecomm.Get_size() == expected_nodecomm.Get_size()
            assert layout.nodecomm.Get_rank() == expected_nodecomm.Get_rank()
        finally:
            expected_nodecomm.Free()
            layout.destroy()

    def test_state_size_matches_alloc_local(self, mpi_comm):
        """Context state array length equals alloc_local from the layout."""
        system_size = 100
        layout = _make_layout(system_size, mpi_comm)

        expected_alloc_local = layout.alloc_local

        ctx = Context(_get_backend(), comm_info=layout)

        state = ctx.state
        assert state is not None
        assert len(state) == expected_alloc_local

        ctx.destroy()
        layout.destroy()

    def test_observables_size_matches_local_i(self, mpi_comm):
        """Context observables array length equals local_i from the layout."""
        system_size = 100
        layout = _make_layout(system_size, mpi_comm)

        expected_local_i = layout.local_i

        ctx = Context(_get_backend(), comm_info=layout)

        obs = ctx.observables
        assert obs is not None
        assert len(obs) == expected_local_i

        ctx.destroy()
        layout.destroy()

    def test_host_alloc_local_matches_layout_contract(self, mpi_comm):
        """Context reads alloc_local from the negotiated layout."""
        system_size = 50
        layout = _make_layout(system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)

        assert ctx.host_alloc_local == layout.alloc_local
        assert ctx.host_alloc_local >= layout.local_i
        assert ctx.host_local_i == layout.local_i
        assert ctx.host_local_i_offset == layout.local_i_offset

        ctx.destroy()
        layout.destroy()

    def test_padded_alloc_local_controls_state_buffer(self, mpi_comm, mpi_rank, mpi_size):
        """MPI contexts honor alloc_local even when it exceeds local_i."""
        if config.backend != "mpi":
            pytest.skip("MPI backend-specific allocation padding regression")

        system_size = max(8, mpi_size * 4)
        local_i, offset = _block_partition(system_size, mpi_size, mpi_rank)
        layout = QuopMpiLayout.from_partition(
            mpi_comm,
            system_size=system_size,
            local_i=local_i,
            local_i_offset=offset,
        )
        padded_alloc_local = local_i + 3

        layout.set_alloc_local(padded_alloc_local)
        layout.lock()

        ctx = Context(_get_backend(), comm_info=layout)
        try:
            test_state = (np.arange(padded_alloc_local, dtype=np.float64) + 1.0 + 0.5j).astype(
                np.complex128
            )
            ctx.state = test_state

            retrieved = ctx.state
            assert ctx.host_alloc_local == padded_alloc_local
            assert ctx.host_alloc_local > ctx.host_local_i
            assert len(retrieved) == padded_alloc_local
            np.testing.assert_array_equal(retrieved, test_state)
        finally:
            ctx.destroy()
            layout.destroy()


@pytest.mark.mpi
class TestContextDestroyLayoutSurvives:
    """T3.2: Context destroy succeeds; QuopMpiLayout still valid after."""

    def test_layout_valid_after_context_destroy(self, mpi_comm):
        """After context.destroy(), layout fields are still accessible."""
        system_size = 64
        layout = _make_layout(system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)
        ctx.destroy()

        # Layout should still be usable
        assert layout.system_size == system_size
        assert layout.local_i > 0
        assert layout.subcomm is not None

        layout.destroy()

    def test_context_manager_destroys_context_only(self, mpi_comm):
        """Context-manager exit destroys the context but preserves the layout."""
        system_size = 64
        layout = _make_layout(system_size, mpi_comm)

        with Context(_get_backend(), comm_info=layout) as ctx:
            assert ctx.initialised is True
            assert layout.subcomm is not None

        assert ctx.initialised is False
        assert layout.system_size == system_size
        assert layout.subcomm is not None

        layout.destroy()

    def test_double_destroy_context(self, mpi_comm):
        """Calling context.destroy() twice does not crash."""
        system_size = 32
        layout = _make_layout(system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)
        ctx.destroy()
        ctx.destroy()  # should be idempotent

        layout.destroy()


@pytest.mark.mpi
class TestContextStateRoundTrip:
    """T3.3: Set state -> get state -> values match."""

    def test_set_get_state(self, mpi_comm):
        """State round-trip through Fortran context preserves values."""
        system_size = 100
        layout = _make_layout(system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)

        test_state = (np.arange(ctx.host_alloc_local, dtype=np.float64) + 1.0 + 0.5j).astype(
            np.complex128
        )

        ctx.state = test_state
        retrieved = ctx.state

        np.testing.assert_array_equal(retrieved, test_state)

        ctx.destroy()
        layout.destroy()

    def test_set_get_observables(self, mpi_comm):
        """Observables round-trip through Fortran context preserves values."""
        system_size = 100
        layout = _make_layout(system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)

        local_i = ctx.host_local_i
        test_obs = np.arange(local_i, dtype=np.float64) * 0.1

        ctx.observables = test_obs
        retrieved = ctx.observables

        np.testing.assert_allclose(retrieved, test_obs)

        ctx.destroy()
        layout.destroy()

    def test_expectation_value(self, mpi_comm):
        """Expectation value computes correctly across ranks.

        Uses uniform state |psi_i|^2 = 1/N and obs_i = i,
        so <obs> = sum(i for i in 0..N-1) / N = (N-1)/2.
        """
        system_size = 100
        layout = _make_layout(system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)

        local_i = ctx.host_local_i
        offset = ctx.host_local_i_offset

        # Uniform probability state: |psi_i| = 1/sqrt(N)
        amplitude = 1.0 / np.sqrt(system_size)
        state = np.full(local_i, amplitude + 0j, dtype=np.complex128)
        ctx.state = state

        # Observables: obs[i] = global_index
        obs = np.arange(offset, offset + local_i, dtype=np.float64)
        ctx.observables = obs

        exp_val = ctx.get_expectation_value()

        expected = (system_size - 1) / 2.0
        assert exp_val == pytest.approx(expected, rel=1e-10)

        ctx.destroy()
        layout.destroy()

    def test_state_norm(self, mpi_comm):
        """State norm is 1.0 for a properly normalized state."""
        system_size = 100
        layout = _make_layout(system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)

        local_i = ctx.host_local_i
        amplitude = 1.0 / np.sqrt(system_size)
        state = np.full(local_i, amplitude + 0j, dtype=np.complex128)
        ctx.state = state

        norm = ctx.get_state_norm()

        assert norm == pytest.approx(1.0, rel=1e-10)

        ctx.destroy()
        layout.destroy()
