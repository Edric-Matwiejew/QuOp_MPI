"""
Unit tests for QuopMpiLayout (Stage 2).

These exercise the Python wrapper around the Fortran quop_mpi_layout_t handle.
All tests work with a single MPI process (no --with-mpi required) but
the class still needs MPI for the Fortran communicator internals, so we
import mpi4py.

Run with:
    python -m pytest tests/unit/test_quop_mpi_layout.py -v
    mpiexec -n 1 python -m pytest tests/unit/test_quop_mpi_layout.py -v --with-mpi
"""

import pytest
from mpi4py import MPI

from quop_mpi._utils import _comm_size as comm_size_module
from quop_mpi._utils._comm_size import QuopMpiLayout


def _make_layout(comm, system_size, local_i, offset, alloc_local=None):
    """Build a layout via the supported explicit partition API."""
    return QuopMpiLayout.from_partition(
        comm,
        system_size=system_size,
        local_i=local_i,
        local_i_offset=offset,
        alloc_local=alloc_local,
    )


# =============================================================================
# T2.1 -- Create and verify all properties
# =============================================================================


class TestCreateAndProperties:
    """T2.1: Create QuopMpiLayout with known data, verify all properties match."""

    def test_basic_properties(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        system_size = 100
        local_i = system_size // size
        remainder = system_size % size
        if rank < remainder:
            local_i += 1
        offset = sum((system_size // size) + (1 if r < remainder else 0) for r in range(rank))

        layout = _make_layout(comm, system_size, local_i, offset)

        assert layout.n_processes == size
        assert layout.local_i == local_i
        assert layout.local_i_offset == offset
        assert layout.system_size == system_size
        assert layout.device_n_processes == 0
        assert layout.device_local_i == 0
        assert layout.device_local_i_offset == 0
        assert layout.comm is comm

        layout.destroy()

    def test_comm_properties(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        # mpi_comm should match the root communicator
        assert layout.mpi_comm is not None

        # subcomm should be valid (dup of MPI_COMM)
        assert layout.subcomm is not None
        assert layout.in_subcomm() is True

        layout.destroy()

    def test_handle_is_nonzero(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)
        assert layout.handle != 0
        assert layout.handle is not None
        layout.destroy()

    def test_constructor_starts_unpartitioned(self):
        comm = MPI.COMM_WORLD
        layout = QuopMpiLayout(comm, system_size=10)

        assert layout.system_size == 10
        assert layout.local_i == 0
        assert layout.local_i_offset == 0
        assert layout.alloc_local == 0

        layout.destroy()

    def test_from_partition_defaults_alloc_local_to_local_i(self):
        comm = MPI.COMM_WORLD
        layout = QuopMpiLayout.from_partition(comm, system_size=10, local_i=10, local_i_offset=0)

        assert layout.alloc_local == 10

        layout.destroy()

    def test_context_manager_destroys_layout_handles(self):
        comm = MPI.COMM_WORLD

        with QuopMpiLayout.from_partition(
            comm,
            system_size=10,
            local_i=10,
            local_i_offset=0,
        ) as layout:
            assert layout.handle is not None
            assert layout.handle != 0

        assert layout.handle is None

    def test_constructor_raises_when_create_returns_error(self, monkeypatch):
        class _FakeCiw:
            @staticmethod
            def create(comm_f):
                return 0, 7

        with monkeypatch.context() as mp:
            mp.setattr(comm_size_module, "_ciw", _FakeCiw())
            with pytest.raises(RuntimeError, match="create failed with status 7"):
                QuopMpiLayout(MPI.COMM_WORLD, system_size=10)


# =============================================================================
# T2.3 -- Lock / unlock
# =============================================================================


class TestLockUnlock:
    """T2.3: is_locked starts False; after lock() is True; after unlock() is False."""

    def test_lock_lifecycle(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        assert layout.is_locked is False

        layout.lock()
        assert layout.is_locked is True

        layout.unlock()
        assert layout.is_locked is False

        layout.destroy()


# =============================================================================
# T2.3b -- Python-side mutator/error translation
# =============================================================================


class TestMutationErrors:
    """Python wrapper translates native status codes into exceptions."""

    def test_lock_twice_raises(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        layout.lock()
        with pytest.raises(RuntimeError, match="already locked"):
            layout.lock()

        layout.destroy()

    def test_unlock_when_unlocked_raises(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        with pytest.raises(RuntimeError, match="already unlocked"):
            layout.unlock()

        layout.destroy()

    def test_locked_setters_raise(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        layout.lock()

        with pytest.raises(RuntimeError, match="locked layout"):
            layout.set_system_size(20)

        with pytest.raises(RuntimeError, match="locked layout"):
            layout.set_partitioning(10, 0)

        with pytest.raises(RuntimeError, match="locked layout"):
            layout.set_alloc_local(12)

        with pytest.raises(RuntimeError, match="locked layout"):
            layout.set_n_processes(comm.Get_size())

        layout.destroy()

    def test_invalid_n_processes_raises_value_error(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        with pytest.raises(ValueError, match="valid SUBCOMM range"):
            layout.set_n_processes(comm.Get_size() + 1)

        layout.destroy()

    def test_invalid_shrink_raises_value_error(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        with pytest.raises(ValueError, match="valid SUBCOMM range"):
            layout.shrink(comm.Get_size() + 1)

        with pytest.raises(ValueError, match="valid SUBCOMM range"):
            layout.shrink(0)

        layout.destroy()

    def test_shrink_partition_table_rebuild_failure_raises_runtime_error(self, monkeypatch):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        class _FakeCiw:
            @staticmethod
            def shrink(ptr, new_size):
                return 3

        with monkeypatch.context() as mp:
            mp.setattr(comm_size_module, "_ciw", _FakeCiw())
            with pytest.raises(RuntimeError, match="rebuild partition_table"):
                layout.shrink(1)

        layout.destroy()

    def test_rebuild_communicators_locked_raises(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        layout.lock()
        with pytest.raises(RuntimeError, match="locked layout"):
            layout.rebuild_communicators()

        layout.destroy()


# =============================================================================
# T2.4 -- partition_table
# =============================================================================


class TestPartitionTable:
    """T2.4: partition_table is correct for uniform partitioning."""

    def test_partition_table_none_before_build(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        # Before build, table should be None
        assert layout.partition_table is None

        layout.destroy()

    def test_partition_table_after_build(self):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        system_size = 100
        local_i = system_size // size
        remainder = system_size % size
        if rank < remainder:
            local_i += 1
        offset = sum((system_size // size) + (1 if r < remainder else 0) for r in range(rank))

        layout = _make_layout(comm, system_size, local_i, offset)

        # build_partition_table is collective
        layout.build_partition_table()

        table = layout.partition_table
        assert table is not None
        assert len(table) == size + 1
        # Fortran 1-based indexing: starts at 1, ends at system_size + 1
        assert table[0] == 1
        assert table[-1] == system_size + 1

        layout.destroy()

    def test_build_partition_table_locked_raises(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        layout.lock()

        with pytest.raises(RuntimeError, match="locked layout"):
            layout.build_partition_table()

        layout.destroy()


# =============================================================================
# T2.5 -- destroy() idempotent
# =============================================================================


class TestDestroyIdempotent:
    """T2.5: destroy() is idempotent (call twice, no crash)."""

    def test_double_destroy(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        layout.destroy()
        assert layout._ptr is None

        # Second call should be a no-op
        layout.destroy()
        assert layout._ptr is None

    def test_repr_after_destroy(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)
        layout.destroy()
        assert "destroyed" in repr(layout)


class TestCreateWorkersCleanup:
    """Failure paths release split/topology handles before raising."""

    def test_create_workers_cleans_handles_on_split_failure(self, monkeypatch):
        calls = []

        class _FakeCiw:
            @staticmethod
            def wrapper_discover_topology(comm_f, backend_flag):
                return 111, 0

            @staticmethod
            def wrapper_split_workers(comm_f, topo_ptr, n_workers, backend_flag):
                return 222, -1, 5

            @staticmethod
            def destroy_split(ptr):
                calls.append(("destroy_split", ptr))

            @staticmethod
            def wrapper_destroy_topology(ptr):
                calls.append(("destroy_topology", ptr))

        with monkeypatch.context() as mp:
            mp.setattr(comm_size_module, "_ciw", _FakeCiw())
            with pytest.raises(RuntimeError, match="split_workers failed with status 5"):
                QuopMpiLayout.create_workers(2, MPI.COMM_WORLD, backend_flag=0)

        assert calls == [("destroy_split", 222), ("destroy_topology", 111)]

    def test_create_workers_raises_when_discover_topology_fails(self, monkeypatch):
        class _FakeCiw:
            @staticmethod
            def wrapper_discover_topology(comm_f, backend_flag):
                return 0, 9

        with monkeypatch.context() as mp:
            mp.setattr(comm_size_module, "_ciw", _FakeCiw())
            with pytest.raises(RuntimeError, match="discover_topology failed with status 9"):
                QuopMpiLayout.create_workers(2, MPI.COMM_WORLD, backend_flag=0)


# =============================================================================
# T2.6 -- in_subcomm
# =============================================================================


class TestInSubcomm:
    """T2.6: in_subcomm is True for all ranks when no shrinking."""

    def test_in_subcomm_true(self):
        comm = MPI.COMM_WORLD
        layout = _make_layout(comm, 10, 10, 0)

        assert layout.in_subcomm() is True

        layout.destroy()
