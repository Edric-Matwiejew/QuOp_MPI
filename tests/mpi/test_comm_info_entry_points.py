"""
Python tests for the entry-point pipeline:
    discover_topology -> split_workers -> negotiate -> create_jaccomm

Also exercises error paths (invalid system_size, invalid n_jacobian_workers),
topology/split lifecycle, and the new JACCOMM accessor.

Run with:
    mpiexec -n 2 python -m pytest tests/mpi/test_comm_info_entry_points.py -v --with-mpi
    mpiexec -n 4 python -m pytest tests/mpi/test_comm_info_entry_points.py -v --with-mpi
"""

import os

import numpy as np
import pytest
from mpi4py import MPI

from quop_mpi._lib import comm_info_wrapper

_ciw = comm_info_wrapper.comm_info_wrapper


def _world_handle():
    return MPI.COMM_WORLD.py2f()


def _backend_flag():
    """Return backend flag expected by Fortran wrappers (0=MPI, 1=wavefront)."""
    backend = os.environ.get("QUOP_BACKEND", "mpi").lower()
    return 1 if "wavefront" in backend else 0


def _discover_topology_handle():
    """Create a topology handle and assert wrapper success."""
    topo, status = _ciw.wrapper_discover_topology(_world_handle(), _backend_flag())
    assert status == 0
    return topo


# =============================================================================
# discover_topology
# =============================================================================


@pytest.mark.mpi
class TestDiscoverTopology:

    def test_returns_nonzero_ptr(self, mpi_comm):
        topo = _discover_topology_handle()
        assert topo != 0
        topo = _ciw.wrapper_destroy_topology(topo)

    def test_destroy_returns_zero(self, mpi_comm):
        """destroy_topology returns 0 so Python can zero its handle."""
        topo = _discover_topology_handle()
        topo = _ciw.wrapper_destroy_topology(topo)
        assert topo == 0

    def test_destroy_idempotent(self, mpi_comm):
        """Calling destroy twice (with zeroed handle) is safe."""
        topo = _discover_topology_handle()
        topo = _ciw.wrapper_destroy_topology(topo)
        assert topo == 0
        # Second call with zeroed handle should be a no-op
        topo = _ciw.wrapper_destroy_topology(topo)
        assert topo == 0

    def test_destroy_null_is_noop(self, mpi_comm):
        """destroy_topology on 0 should not crash."""
        result = _ciw.wrapper_destroy_topology(0)
        assert result == 0


# =============================================================================
# split_workers
# =============================================================================


@pytest.mark.mpi
class TestSplitWorkers:

    def test_single_worker(self, mpi_comm, mpi_size):
        """n_jacobian_workers=1 -> worker_id=0, status=0."""
        topo = _discover_topology_handle()
        split, wid, status = _ciw.wrapper_split_workers(_world_handle(), topo, 1, _backend_flag())
        assert status == 0
        assert wid == 0
        assert split != 0
        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)

    def test_invalid_n_workers(self, mpi_comm, mpi_size):
        """n_jacobian_workers > nprocs -> status=1."""
        topo = _discover_topology_handle()
        split, wid, status = _ciw.wrapper_split_workers(
            _world_handle(), topo, mpi_size + 1, _backend_flag()
        )
        assert status == 1
        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)

    def test_zero_n_workers(self, mpi_comm, mpi_size):
        """n_jacobian_workers=0 -> status=1."""
        topo = _discover_topology_handle()
        split, wid, status = _ciw.wrapper_split_workers(_world_handle(), topo, 0, _backend_flag())
        assert status == 1
        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)

    @pytest.mark.requires_nprocs(2)
    def test_two_workers(self, mpi_comm, mpi_rank, mpi_size):
        """n_jacobian_workers=2 -> each rank gets worker_id in {0,1}."""
        topo = _discover_topology_handle()
        split, wid, status = _ciw.wrapper_split_workers(_world_handle(), topo, 2, _backend_flag())
        assert status == 0
        assert wid in (0, 1)
        assert _ciw.get_n_workers(split) == 2
        assert _ciw.get_worker_id(split) == wid
        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)

    def test_split_info_accessors(self, mpi_comm, mpi_size):
        """get_worker_id and get_n_workers return correct values."""
        topo = _discover_topology_handle()
        split, wid, status = _ciw.wrapper_split_workers(_world_handle(), topo, 1, _backend_flag())
        assert _ciw.get_worker_id(split) == 0
        assert _ciw.get_n_workers(split) == 1
        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)


# =============================================================================
# negotiate
# =============================================================================


@pytest.mark.mpi
class TestNegotiate:

    def _setup_split(self, n_workers=1):
        topo = _discover_topology_handle()
        split, wid, status = _ciw.wrapper_split_workers(
            _world_handle(), topo, n_workers, _backend_flag()
        )
        return topo, split

    def test_success(self, mpi_comm, mpi_rank, mpi_size):
        """negotiate with valid system_size -> status=0, locked layout."""
        topo, split = self._setup_split()
        props = np.array([], dtype=np.int64)
        cbs = np.array([], dtype=np.int64)
        layout, status = _ciw.wrapper_negotiate(split, topo, 100, _backend_flag(), props, cbs)
        assert status == 0
        assert layout != 0

        # Layout should be locked
        assert _ciw.is_locked(layout) == 1

        # MPI_COMM should be set (not null)
        comm_handle = _ciw.get_mpi_comm(layout)
        assert comm_handle != MPI.COMM_NULL.py2f()

        # alloc_local should be > 0
        assert _ciw.get_alloc_local(layout) > 0

        # system_size correct
        assert _ciw.get_system_size(layout) == 100

        _ciw.destroy(layout)
        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)

    def test_zero_system_size(self, mpi_comm):
        """negotiate with system_size=0 -> status=1, null layout."""
        topo, split = self._setup_split()
        props = np.array([], dtype=np.int64)
        cbs = np.array([], dtype=np.int64)
        layout, status = _ciw.wrapper_negotiate(split, topo, 0, _backend_flag(), props, cbs)
        assert status == 1
        assert layout == 0

        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)

    def test_negative_system_size(self, mpi_comm):
        """negotiate with system_size=-1 -> status=1."""
        topo, split = self._setup_split()
        props = np.array([], dtype=np.int64)
        cbs = np.array([], dtype=np.int64)
        layout, status = _ciw.wrapper_negotiate(split, topo, -1, _backend_flag(), props, cbs)
        assert status == 1
        assert layout == 0

        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)

    def test_partition_table_correct(self, mpi_comm, mpi_rank, mpi_size):
        """After negotiate, partition table matches block distribution."""
        system_size = 97
        topo, split = self._setup_split()
        props = np.array([], dtype=np.int64)
        cbs = np.array([], dtype=np.int64)
        layout, status = _ciw.wrapper_negotiate(
            split, topo, system_size, _backend_flag(), props, cbs
        )
        assert status == 0

        n = _ciw.get_partition_table_size(layout)
        assert n == mpi_size + 1
        table = _ciw.get_partition_table(layout, n)
        assert table[0] == 1
        assert table[-1] == system_size + 1

        _ciw.destroy(layout)
        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)

    @pytest.mark.requires_nprocs(2)
    def test_negotiate_shrinks_to_system_size_when_oversubscribed(
        self, mpi_comm, mpi_rank, mpi_size
    ):
        """When system_size < comm_size, negotiate keeps at most system_size ranks."""
        system_size = 1
        topo, split = self._setup_split()
        props = np.array([], dtype=np.int64)
        cbs = np.array([], dtype=np.int64)
        layout, status = _ciw.wrapper_negotiate(
            split, topo, system_size, _backend_flag(), props, cbs
        )
        assert status in (0, -1)

        if mpi_rank == 0:
            assert layout != 0
            assert _ciw.get_subcomm(layout) != MPI.COMM_NULL.py2f()
            assert _ciw.get_n_processes(layout) == 1
            assert _ciw.get_local_i(layout) == 1
            assert _ciw.get_local_i_offset(layout) == 0
            assert _ciw.validate(layout, system_size) == 0
        else:
            assert layout != 0
            assert _ciw.get_subcomm(layout) == MPI.COMM_NULL.py2f()
            assert _ciw.get_n_processes(layout) == 0
            assert _ciw.get_local_i(layout) == 0
            assert _ciw.get_local_i_offset(layout) == 0

        _ciw.destroy(layout)
        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)


# =============================================================================
# create_jaccomm
# =============================================================================


@pytest.mark.mpi
class TestCreateJaccomm:

    def test_single_worker_jaccomm(self, mpi_comm, mpi_rank, mpi_size):
        """With 1 worker, JACCOMM has size 1 for rank 0, null for others."""
        topo = _discover_topology_handle()
        split, wid, status = _ciw.wrapper_split_workers(_world_handle(), topo, 1, _backend_flag())
        props = np.array([], dtype=np.int64)
        cbs = np.array([], dtype=np.int64)
        layout, status = _ciw.wrapper_negotiate(split, topo, 100, _backend_flag(), props, cbs)

        _ciw.wrapper_create_jaccomm(_world_handle(), split, layout)

        jac_handle = _ciw.get_jaccomm(split)

        if mpi_rank == 0:
            jac = MPI.Comm.f2py(jac_handle)
            assert jac != MPI.COMM_NULL
            assert jac.Get_size() == 1
        else:
            assert jac_handle == MPI.COMM_NULL.py2f()

        _ciw.destroy(layout)
        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)

    @pytest.mark.requires_nprocs(2)
    def test_two_workers_jaccomm(self, mpi_comm, mpi_rank, mpi_size):
        """With 2 workers, JACCOMM includes all worker ranks + optimizer root."""
        topo = _discover_topology_handle()
        split, wid, status = _ciw.wrapper_split_workers(_world_handle(), topo, 2, _backend_flag())
        props = np.array([], dtype=np.int64)
        cbs = np.array([], dtype=np.int64)
        layout, status = _ciw.wrapper_negotiate(split, topo, 100, _backend_flag(), props, cbs)

        _ciw.wrapper_create_jaccomm(_world_handle(), split, layout)

        jac_handle = _ciw.get_jaccomm(split)
        subcomm_handle = _ciw.get_subcomm(layout)
        worker_id = _ciw.get_worker_id(split)

        if subcomm_handle != MPI.COMM_NULL.py2f():
            subcomm = MPI.Comm.f2py(subcomm_handle)
            sub_rank = subcomm.Get_rank()
            if worker_id > 0:
                # Worker subcomm: ALL ranks get JACCOMM
                jac = MPI.Comm.f2py(jac_handle)
                assert jac != MPI.COMM_NULL
            elif sub_rank == 0:
                # Optimizer subcomm leader gets JACCOMM
                jac = MPI.Comm.f2py(jac_handle)
                assert jac != MPI.COMM_NULL
            else:
                # Optimizer subcomm non-leader: no JACCOMM
                assert jac_handle == MPI.COMM_NULL.py2f()

        _ciw.destroy(layout)
        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)

    def test_null_layout_does_not_crash(self, mpi_comm):
        """create_jaccomm with null layout (from failed negotiate) should not crash."""
        topo = _discover_topology_handle()
        split, wid, status = _ciw.wrapper_split_workers(_world_handle(), topo, 1, _backend_flag())
        props = np.array([], dtype=np.int64)
        cbs = np.array([], dtype=np.int64)
        # negotiate with system_size=0 -> null layout
        layout, neg_status = _ciw.wrapper_negotiate(split, topo, 0, _backend_flag(), props, cbs)
        assert neg_status == 1
        assert layout == 0

        # This should not crash (issue #2 fix)
        _ciw.wrapper_create_jaccomm(_world_handle(), split, layout)

        # JACCOMM should be null (all ranks get MPI_UNDEFINED color)
        jac_handle = _ciw.get_jaccomm(split)
        assert jac_handle == MPI.COMM_NULL.py2f()

        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)


# =============================================================================
# Full Pipeline Integration
# =============================================================================


@pytest.mark.mpi
class TestFullPipeline:

    def test_discover_split_negotiate_jaccomm_destroy(self, mpi_comm, mpi_rank, mpi_size):
        """Full lifecycle: discover -> split -> negotiate -> jaccomm -> destroy all."""
        system_size = 200

        # Phase 0: discover
        topo = _discover_topology_handle()
        assert topo != 0

        # Phase 0b: split
        split, wid, status = _ciw.wrapper_split_workers(_world_handle(), topo, 1, _backend_flag())
        assert status == 0

        # Phases 1-5: negotiate
        props = np.array([], dtype=np.int64)
        cbs = np.array([], dtype=np.int64)
        layout, status = _ciw.wrapper_negotiate(
            split, topo, system_size, _backend_flag(), props, cbs
        )
        assert status == 0
        assert _ciw.is_locked(layout) == 1

        # Verify partitioning
        assert _ciw.get_system_size(layout) == system_size
        n = _ciw.get_partition_table_size(layout)
        table = _ciw.get_partition_table(layout, n)
        assert table[0] == 1
        assert table[-1] == system_size + 1

        # Phase 6: jaccomm
        _ciw.wrapper_create_jaccomm(_world_handle(), split, layout)

        if mpi_rank == 0:
            jac_handle = _ciw.get_jaccomm(split)
            assert jac_handle != MPI.COMM_NULL.py2f()

        # Teardown (reverse order)
        _ciw.destroy(layout)
        _ciw.destroy_split(split)
        topo = _ciw.wrapper_destroy_topology(topo)
