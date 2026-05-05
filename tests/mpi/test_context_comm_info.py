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


@pytest.fixture
def context_nodecomm_system_size(mpi_sizing):
    """Power-of-two size large enough to activate all ranks for nodecomm checks."""
    return mpi_sizing.power_of_two(base=32, min_per_rank=1)


@pytest.fixture
def context_regular_system_size(mpi_sizing):
    """Representative successful system size for state and observable buffers."""
    return mpi_sizing.multiple(base=100, per_rank=16)


@pytest.fixture
def context_alloc_contract_system_size(mpi_sizing):
    """Non-power-of-two size for alloc_local and offset contract checks."""
    return mpi_sizing.multiple(base=50, per_rank=8, remainder=2)


@pytest.fixture
def context_lifecycle_system_size(mpi_sizing):
    """Lifecycle size that keeps the layout active without oversizing buffers."""
    return mpi_sizing.power_of_two(base=64, min_per_rank=1)


@pytest.fixture
def context_small_system_size(mpi_sizing):
    """Small power-of-two size for destroy/idempotence checks."""
    return mpi_sizing.power_of_two(base=32, min_per_rank=1)


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

    def test_layout_nodecomm_matches_shared_memory_group(
        self, mpi_comm, context_nodecomm_system_size
    ):
        """Negotiated layouts expose the node-local communicator on all backends."""
        layout = _make_layout(context_nodecomm_system_size, mpi_comm)
        expected_nodecomm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)

        try:
            assert layout.nodecomm is not None
            assert layout.nodecomm.Get_size() == expected_nodecomm.Get_size()
            assert layout.nodecomm.Get_rank() == expected_nodecomm.Get_rank()
        finally:
            expected_nodecomm.Free()
            layout.destroy()

    def test_state_size_matches_alloc_local(self, mpi_comm, context_regular_system_size):
        """Context state array length equals alloc_local from the layout."""
        layout = _make_layout(context_regular_system_size, mpi_comm)

        expected_alloc_local = layout.alloc_local

        ctx = Context(_get_backend(), comm_info=layout)

        state = ctx.state
        assert state is not None
        assert len(state) == expected_alloc_local

        ctx.destroy()
        layout.destroy()

    def test_observables_size_matches_local_i(self, mpi_comm, context_regular_system_size):
        """Context observables array length equals local_i from the layout."""
        layout = _make_layout(context_regular_system_size, mpi_comm)

        expected_local_i = layout.local_i

        ctx = Context(_get_backend(), comm_info=layout)

        obs = ctx.observables
        assert obs is not None
        assert len(obs) == expected_local_i

        ctx.destroy()
        layout.destroy()

    def test_host_alloc_local_matches_layout_contract(
        self, mpi_comm, context_alloc_contract_system_size
    ):
        """Context reads alloc_local from the negotiated layout."""
        layout = _make_layout(context_alloc_contract_system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)

        assert ctx.host_alloc_local == layout.alloc_local
        assert ctx.host_alloc_local >= layout.local_i
        assert ctx.host_local_i == layout.local_i
        assert ctx.host_local_i_offset == layout.local_i_offset

        ctx.destroy()
        layout.destroy()

    def test_padded_alloc_local_controls_state_buffer(self, mpi_comm, mpi_rank, mpi_size):
        """Contexts honor alloc_local even when it exceeds local_i."""
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
            # Property-level contract: both backends must report the
            # padded alloc_local through the Context.
            assert ctx.host_alloc_local == padded_alloc_local
            assert ctx.host_alloc_local > ctx.host_local_i

            # State write/read requires the wavefront device communicator
            # hierarchy (DEVCOMM, DEVCOMM_NODE). Explicit from_partition
            # layouts do not build that hierarchy, so verify the buffer
            # round-trip on MPI only.
            if config.backend == "mpi":
                test_state = (
                    np.arange(padded_alloc_local, dtype=np.float64) + 1.0 + 0.5j
                ).astype(np.complex128)
                ctx.state = test_state

                retrieved = ctx.state
                assert len(retrieved) == padded_alloc_local
                np.testing.assert_array_equal(retrieved, test_state)
        finally:
            ctx.destroy()
            layout.destroy()


@pytest.mark.mpi
class TestContextDestroyLayoutSurvives:
    """T3.2: Context destroy succeeds; QuopMpiLayout still valid after."""

    def test_layout_valid_after_context_destroy(self, mpi_comm, context_lifecycle_system_size):
        """After context.destroy(), layout fields are still accessible."""
        layout = _make_layout(context_lifecycle_system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)
        ctx.destroy()

        # Layout should still be usable
        assert layout.system_size == context_lifecycle_system_size
        assert layout.local_i > 0
        assert layout.subcomm is not None

        layout.destroy()

    def test_context_manager_destroys_context_only(self, mpi_comm, context_lifecycle_system_size):
        """Context-manager exit destroys the context but preserves the layout."""
        layout = _make_layout(context_lifecycle_system_size, mpi_comm)

        with Context(_get_backend(), comm_info=layout) as ctx:
            assert ctx.initialised is True
            assert layout.subcomm is not None

        assert ctx.initialised is False
        assert layout.system_size == context_lifecycle_system_size
        assert layout.subcomm is not None

        layout.destroy()

    def test_double_destroy_context(self, mpi_comm, context_small_system_size):
        """Calling context.destroy() twice does not crash."""
        layout = _make_layout(context_small_system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)
        ctx.destroy()
        ctx.destroy()  # should be idempotent

        layout.destroy()


@pytest.mark.mpi
class TestContextStateRoundTrip:
    """T3.3: Set state -> get state -> values match."""

    def test_set_get_state(self, mpi_comm, context_regular_system_size):
        """State round-trip through Fortran context preserves values."""
        layout = _make_layout(context_regular_system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)

        test_state = (np.arange(ctx.host_alloc_local, dtype=np.float64) + 1.0 + 0.5j).astype(
            np.complex128
        )

        ctx.state = test_state
        retrieved = ctx.state

        np.testing.assert_array_equal(retrieved, test_state)

        ctx.destroy()
        layout.destroy()

    def test_set_get_observables(self, mpi_comm, context_regular_system_size):
        """Observables round-trip through Fortran context preserves values."""
        layout = _make_layout(context_regular_system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)

        local_i = ctx.host_local_i
        test_obs = np.arange(local_i, dtype=np.float64) * 0.1

        ctx.observables = test_obs
        retrieved = ctx.observables

        np.testing.assert_allclose(retrieved, test_obs)

        ctx.destroy()
        layout.destroy()

    def test_expectation_value(self, mpi_comm, context_regular_system_size):
        """Expectation value computes correctly across ranks.

        Uses uniform state |psi_i|^2 = 1/N and obs_i = i,
        so <obs> = sum(i for i in 0..N-1) / N = (N-1)/2.
        """
        layout = _make_layout(context_regular_system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)

        local_i = ctx.host_local_i
        offset = ctx.host_local_i_offset

        # Uniform probability state: |psi_i| = 1/sqrt(N)
        amplitude = 1.0 / np.sqrt(context_regular_system_size)
        state = np.full(local_i, amplitude + 0j, dtype=np.complex128)
        ctx.state = state

        # Observables: obs[i] = global_index
        obs = np.arange(offset, offset + local_i, dtype=np.float64)
        ctx.observables = obs

        exp_val = ctx.get_expectation_value()

        expected = (context_regular_system_size - 1) / 2.0
        assert exp_val == pytest.approx(expected, rel=1e-10)

        ctx.destroy()
        layout.destroy()

    def test_state_norm(self, mpi_comm, context_regular_system_size):
        """State norm is 1.0 for a properly normalized state."""
        layout = _make_layout(context_regular_system_size, mpi_comm)

        ctx = Context(_get_backend(), comm_info=layout)

        local_i = ctx.host_local_i
        amplitude = 1.0 / np.sqrt(context_regular_system_size)
        state = np.full(local_i, amplitude + 0j, dtype=np.complex128)
        ctx.state = state

        norm = ctx.get_state_norm()

        assert norm == pytest.approx(1.0, rel=1e-10)

        ctx.destroy()
        layout.destroy()

    def test_state_view_aliases_native_buffer(
        self, mpi_comm, context_regular_system_size
    ):
        """In-place mutation of ``ctx.state`` is visible to Fortran.

        Locks the zero-copy invariant: the NumPy view returned by
        ``Context.state`` shares storage with the Fortran ``ctx%state``
        pointer attached at setup, so writes through the view must be
        observed by collective routines that read through the pointer
        (here, ``get_state_norm`` and ``get_expectation_value``, which
        compute over ``self%state(:ci_local_i)`` in Fortran).
        """
        layout = _make_layout(context_regular_system_size, mpi_comm)
        ctx = Context(_get_backend(), comm_info=layout)

        local_i = ctx.host_local_i

        # Fetch the cached buffer view, then mutate in place WITHOUT
        # going through the ctx.state setter (which would still copy).
        view = ctx.state
        amplitude = 1.0 / np.sqrt(context_regular_system_size)
        view[:local_i] = amplitude + 0j

        # A second fetch should return the SAME ndarray object — the
        # Context caches a single PyArrayObject across get_state calls.
        view_again = ctx.state
        assert view_again is view

        # Fortran reads through self%state must see the in-place write.
        norm = ctx.get_state_norm()
        assert norm == pytest.approx(1.0, rel=1e-10)

        # Mutate observables in place too and confirm Fortran sees it.
        obs_view = ctx.observables
        obs_view[:local_i] = np.full(local_i, 2.0, dtype=np.float64)
        # Re-fetch returns the same object.
        assert ctx.observables is obs_view

        # <obs> = sum_i p_i * obs_i = 2.0 * sum_i p_i = 2.0 (norm=1).
        exp_val = ctx.get_expectation_value()
        assert exp_val == pytest.approx(2.0, rel=1e-10)

        ctx.destroy()
        layout.destroy()
