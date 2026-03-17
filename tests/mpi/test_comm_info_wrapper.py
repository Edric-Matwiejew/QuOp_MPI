"""
Python interface tests for the comm_info_wrapper f2py extension module.

Tests the quop_mpi_layout_t Fortran type through its f2py wrapper API,
verifying create/destroy lifecycle, scalar field access, partition table
building, lock/unlock semantics,
validation, and communicator shrink.

Run with:
    mpiexec -n 2 python -m pytest tests/mpi/test_comm_info_wrapper.py -v --with-mpi
    mpiexec -n 4 python -m pytest tests/mpi/test_comm_info_wrapper.py -v --with-mpi
"""

import os

import numpy as np
import pytest
from mpi4py import MPI

from quop_mpi._lib import comm_info_wrapper

# Convenience alias
_ciw = comm_info_wrapper.comm_info_wrapper


# =============================================================================
# Helpers
# =============================================================================


def _world_handle():
    """Return the Fortran integer handle for MPI_COMM_WORLD."""
    return MPI.COMM_WORLD.py2f()


def _compute_partition(system_size, rank, nprocs):
    """Compute (local_i, local_i_offset) using the same distribution
    strategy as the Fortran module (first `remainder` ranks get +1)."""
    base = system_size // nprocs
    rem = system_size % nprocs
    if rank < rem:
        local_i = base + 1
        offset = rank * (base + 1)
    else:
        local_i = base
        offset = rank * base + rem
    return local_i, offset


def _backend_flag():
    """Return backend flag expected by Fortran wrappers (0=MPI, 1=wavefront)."""
    backend = os.environ.get("QUOP_BACKEND", "mpi").lower()
    return 1 if "wavefront" in backend else 0


def _create_layout_handle():
    """Create a raw layout handle and assert wrapper success."""
    ci, status = _ciw.create(_world_handle())
    assert status == 0
    return ci


def _discover_topology_handle():
    """Create a topology handle and assert wrapper success."""
    topo, status = _ciw.wrapper_discover_topology(_world_handle(), _backend_flag())
    assert status == 0
    return topo


def _create_layout_via_pipeline(system_size):
    """Create a layout using the full topology-backed setup pipeline."""
    backend_flag = _backend_flag()
    topo = _discover_topology_handle()
    split, _, split_status = _ciw.wrapper_split_workers(_world_handle(), topo, 1, backend_flag)
    assert split_status == 0

    empty = np.array([], dtype=np.int64)
    layout, negotiate_status = _ciw.wrapper_negotiate(
        split, topo, system_size, backend_flag, empty, empty
    )
    assert negotiate_status == 0

    return topo, split, layout


# =============================================================================
# Lifecycle Tests
# =============================================================================


@pytest.mark.mpi
class TestLifecycle:
    """Create / destroy and basic opaque-pointer contract."""

    def test_create_returns_nonzero_pointer(self, mpi_comm):
        """create() returns a non-zero opaque int64 pointer."""
        ci = _create_layout_handle()
        assert ci != 0
        _ciw.destroy(ci)

    def test_create_stores_mpi_comm(self, mpi_comm):
        """The stored MPI_COMM handle matches what was passed in."""
        ci = _create_layout_handle()
        stored = _ciw.get_mpi_comm(ci)
        assert stored == _world_handle()
        _ciw.destroy(ci)

    def test_create_dups_subcomm(self, mpi_comm):
        """SUBCOMM is a duplicate (different handle, same size)."""
        ci = _create_layout_handle()
        subcomm_handle = _ciw.get_subcomm(ci)
        # Different handle from MPI_COMM
        assert subcomm_handle != _world_handle()
        # But same size
        subcomm = MPI.Comm.f2py(subcomm_handle)
        assert subcomm.Get_size() == mpi_comm.Get_size()
        _ciw.destroy(ci)

    def test_n_processes_equals_comm_size(self, mpi_comm):
        """n_processes is set from SUBCOMM size at create time."""
        ci = _create_layout_handle()
        assert _ciw.get_n_processes(ci) == mpi_comm.Get_size()
        _ciw.destroy(ci)

    def test_destroy_is_idempotent_safe(self, mpi_comm):
        """destroy() doesn't crash (basic smoke test)."""
        ci = _create_layout_handle()
        _ciw.destroy(ci)
        # If we get here, it didn't segfault


# =============================================================================
# Lock / Unlock
# =============================================================================


@pytest.mark.mpi
class TestLockUnlock:
    """Lock/unlock semantics through the wrapper."""

    def test_starts_unlocked(self, mpi_comm):
        ci = _create_layout_handle()
        assert _ciw.is_locked(ci) == 0
        _ciw.destroy(ci)

    def test_lock_then_is_locked(self, mpi_comm):
        ci = _create_layout_handle()
        _ciw.lock(ci)
        assert _ciw.is_locked(ci) == 1
        _ciw.destroy(ci)

    def test_unlock_clears_lock(self, mpi_comm):
        ci = _create_layout_handle()
        _ciw.lock(ci)
        _ciw.unlock(ci)
        assert _ciw.is_locked(ci) == 0
        _ciw.destroy(ci)

    def test_double_lock_returns_error(self, mpi_comm):
        ci = _create_layout_handle()
        assert _ciw.lock(ci) == 0
        assert _ciw.lock(ci) == 1
        _ciw.destroy(ci)

    def test_unlock_unlocked_returns_error(self, mpi_comm):
        ci = _create_layout_handle()
        assert _ciw.unlock(ci) == 1
        _ciw.destroy(ci)


# =============================================================================
# Scalar Field Accessors
# =============================================================================


@pytest.mark.mpi
class TestScalarAccessors:
    """Verify get/set for scalar partitioning fields."""

    def test_set_system_size(self, mpi_comm):
        ci = _create_layout_handle()
        assert _ciw.set_system_size(ci, 256) == 0
        assert _ciw.get_system_size(ci) == 256
        _ciw.destroy(ci)

    def test_set_n_processes_matches_subcomm_size(self, mpi_comm, mpi_size):
        ci = _create_layout_handle()
        assert _ciw.set_n_processes(ci, mpi_size) == 0
        assert _ciw.get_n_processes(ci) == mpi_size
        _ciw.destroy(ci)

    def test_set_n_processes_rejects_mismatch(self, mpi_comm, mpi_size):
        ci = _create_layout_handle()
        assert _ciw.set_n_processes(ci, mpi_size + 1) == 2
        assert _ciw.get_n_processes(ci) == mpi_size
        _ciw.destroy(ci)

    def test_set_partitioning_readback(self, mpi_comm, mpi_rank, mpi_size):
        """set_partitioning stores local_i and local_i_offset correctly."""
        ci = _create_layout_handle()
        system_size = 100
        assert _ciw.set_system_size(ci, system_size) == 0

        local_i, offset = _compute_partition(system_size, mpi_rank, mpi_size)
        assert _ciw.set_partitioning(ci, local_i, offset) == 0

        assert _ciw.get_local_i(ci) == local_i
        assert _ciw.get_local_i_offset(ci) == offset
        _ciw.destroy(ci)

    def test_alloc_local_defaults_to_zero(self, mpi_comm):
        """alloc_local starts at 0 before any propagator sets it."""
        ci = _create_layout_handle()
        assert _ciw.get_alloc_local(ci) == 0
        _ciw.destroy(ci)

    def test_device_fields_default_to_zero(self, mpi_comm):
        """All device_* fields are zero on a non-GPU layout."""
        ci = _create_layout_handle()
        assert _ciw.get_device_local_i(ci) == 0
        assert _ciw.get_device_local_i_offset(ci) == 0
        assert _ciw.get_device_n_processes(ci) == 0
        assert _ciw.get_device_alloc_local(ci) == 0
        _ciw.destroy(ci)


# =============================================================================
# Partition Table
# =============================================================================


@pytest.mark.mpi
class TestPartitionTable:
    """build_partition_table and partition table accessors."""

    def _setup_partitioned(self, system_size):
        """Create a layout with a valid partition already applied."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        ci = _create_layout_handle()
        assert _ciw.set_system_size(ci, system_size) == 0
        local_i, offset = _compute_partition(system_size, rank, nprocs)
        assert _ciw.set_partitioning(ci, local_i, offset) == 0
        assert _ciw.build_partition_table(ci) == 0
        return ci

    def test_partition_table_size(self, mpi_comm, mpi_size):
        """Partition table has nprocs + 1 entries."""
        ci = self._setup_partitioned(100)
        n = _ciw.get_partition_table_size(ci)
        assert n == mpi_size + 1
        _ciw.destroy(ci)

    def test_partition_table_boundaries(self, mpi_comm, mpi_size):
        """First element = 1, last element = system_size + 1 (1-based)."""
        system_size = 100
        ci = self._setup_partitioned(system_size)
        n = _ciw.get_partition_table_size(ci)
        table = _ciw.get_partition_table(ci, n)

        assert table[0] == 1
        assert table[-1] == system_size + 1
        _ciw.destroy(ci)

    def test_partition_table_monotonic(self, mpi_comm, mpi_size):
        """Partition table entries are strictly increasing."""
        ci = self._setup_partitioned(100)
        n = _ciw.get_partition_table_size(ci)
        table = _ciw.get_partition_table(ci, n)

        for i in range(len(table) - 1):
            assert table[i + 1] > table[i], (
                f"partition_table not monotonic at index {i}: " f"{table[i]} >= {table[i+1]}"
            )
        _ciw.destroy(ci)

    def test_partition_table_returns_numpy_array(self, mpi_comm):
        """get_partition_table returns an int64 numpy array."""
        ci = self._setup_partitioned(100)
        n = _ciw.get_partition_table_size(ci)
        table = _ciw.get_partition_table(ci, n)

        assert isinstance(table, np.ndarray)
        assert table.dtype == np.int64
        _ciw.destroy(ci)

    def test_partition_table_not_allocated_before_build(self, mpi_comm):
        """Partition table size is 0 before build_partition_table."""
        ci = _create_layout_handle()
        assert _ciw.get_partition_table_size(ci) == 0
        _ciw.destroy(ci)


# =============================================================================
# Validate
# =============================================================================


@pytest.mark.mpi
class TestValidate:
    """validate() on a well-formed partition should not abort."""

    def test_validate_good_partition(self, mpi_comm, mpi_rank, mpi_size):
        """A correct even partition passes validation."""
        system_size = 100
        ci = _create_layout_handle()
        assert _ciw.set_system_size(ci, system_size) == 0

        local_i, offset = _compute_partition(system_size, mpi_rank, mpi_size)
        assert _ciw.set_partitioning(ci, local_i, offset) == 0
        assert _ciw.set_alloc_local(ci, local_i) == 0
        assert _ciw.build_partition_table(ci) == 0

        # Should not raise / abort
        assert _ciw.validate(ci, system_size) == 0
        _ciw.destroy(ci)

    def test_validate_prime_system_size(self, mpi_comm, mpi_rank, mpi_size):
        """Validation works for a prime system_size (uneven distribution)."""
        system_size = 97
        ci = _create_layout_handle()
        assert _ciw.set_system_size(ci, system_size) == 0

        local_i, offset = _compute_partition(system_size, mpi_rank, mpi_size)
        assert _ciw.set_partitioning(ci, local_i, offset) == 0
        assert _ciw.set_alloc_local(ci, local_i) == 0
        assert _ciw.build_partition_table(ci) == 0

        assert _ciw.validate(ci, system_size) == 0
        _ciw.destroy(ci)


# =============================================================================
# Shrink
# =============================================================================


@pytest.mark.mpi
class TestShrink:
    """Test communicator shrink through the wrapper."""

    def test_shrink_locked_layout_returns_error(self, mpi_comm):
        ci = _create_layout_handle()
        _ciw.lock(ci)

        assert _ciw.shrink(ci, 1) == 1
        _ciw.destroy(ci)

    def test_shrink_invalid_size_returns_error(self, mpi_comm, mpi_size):
        ci = _create_layout_handle()

        assert _ciw.shrink(ci, mpi_size + 1) == 2
        assert _ciw.shrink(ci, 0) == 2
        _ciw.destroy(ci)

    @pytest.mark.requires_nprocs(2)
    def test_shrink_to_one(self, mpi_comm, mpi_rank, mpi_size):
        """shrink(1) gives rank 0 a valid SUBCOMM and others MPI_COMM_NULL.

        Uses the full backend pipeline so both MPI and wavefront follow the
        same topology-backed setup path.
        """
        topo = split = ci = 0
        try:
            topo, split, ci = _create_layout_via_pipeline(system_size=max(128, mpi_size * 2))
            _ciw.unlock(ci)
            _ciw.shrink(ci, 1)

            subcomm_handle = _ciw.get_subcomm(ci)

            if mpi_rank == 0:
                # Rank 0 keeps a valid communicator
                subcomm = MPI.Comm.f2py(subcomm_handle)
                assert subcomm != MPI.COMM_NULL
                assert subcomm.Get_size() == 1
                assert _ciw.get_n_processes(ci) == 1
                assert _ciw.get_local_i(ci) == max(128, mpi_size * 2)
                assert _ciw.get_local_i_offset(ci) == 0
                assert _ciw.get_alloc_local(ci) == max(128, mpi_size * 2)
                assert _ciw.get_partition_table_size(ci) == 2
                assert _ciw.validate(ci, max(128, mpi_size * 2)) == 0
            else:
                # Other ranks get MPI_COMM_NULL (Fortran handle)
                assert subcomm_handle == MPI.COMM_NULL.py2f()
                assert _ciw.get_n_processes(ci) == 0
                assert _ciw.get_local_i(ci) == 0
                assert _ciw.get_local_i_offset(ci) == 0
                assert _ciw.get_alloc_local(ci) == 0
                assert _ciw.get_partition_table_size(ci) == 0
        finally:
            if ci:
                _ciw.destroy(ci)
            if split:
                _ciw.destroy_split(split)
            if topo:
                _ciw.wrapper_destroy_topology(topo)

    @pytest.mark.requires_nprocs(4)
    def test_shrink_to_half(self, mpi_comm, mpi_rank, mpi_size):
        """shrink(nprocs/2) keeps the first half of ranks after negotiate."""
        half = mpi_size // 2
        system_size = max(128, mpi_size * 2)
        topo = split = ci = 0
        try:
            topo, split, ci = _create_layout_via_pipeline(system_size=system_size)
            _ciw.unlock(ci)
            _ciw.shrink(ci, half)

            subcomm_handle = _ciw.get_subcomm(ci)

            if mpi_rank < half:
                subcomm = MPI.Comm.f2py(subcomm_handle)
                assert subcomm != MPI.COMM_NULL
                assert subcomm.Get_size() == half
                expected_local_i, expected_offset = _compute_partition(system_size, mpi_rank, half)
                assert _ciw.get_n_processes(ci) == half
                assert _ciw.get_local_i(ci) == expected_local_i
                assert _ciw.get_local_i_offset(ci) == expected_offset
                assert _ciw.get_alloc_local(ci) == expected_local_i
                assert _ciw.get_partition_table_size(ci) == half + 1
                assert _ciw.validate(ci, system_size) == 0
            else:
                assert subcomm_handle == MPI.COMM_NULL.py2f()
                assert _ciw.get_n_processes(ci) == 0
                assert _ciw.get_local_i(ci) == 0
                assert _ciw.get_local_i_offset(ci) == 0
                assert _ciw.get_alloc_local(ci) == 0
                assert _ciw.get_partition_table_size(ci) == 0
        finally:
            if ci:
                _ciw.destroy(ci)
            if split:
                _ciw.destroy_split(split)
            if topo:
                _ciw.wrapper_destroy_topology(topo)


# =============================================================================
# Communicator Handle Accessors
# =============================================================================


@pytest.mark.mpi
class TestCommHandles:
    """Test communicator handle getter wrappers."""

    def test_nodecomm_default_null(self, mpi_comm):
        """NODECOMM is MPI_COMM_NULL on the raw create() path."""
        ci = _create_layout_handle()
        # NODECOMM is not set by create(), should be null
        handle = _ciw.get_nodecomm(ci)
        assert handle == MPI.COMM_NULL.py2f()
        _ciw.destroy(ci)

    def test_nodecomm_created_after_negotiate(self, mpi_comm):
        """negotiate() creates NODECOMM for active ranks on all backends."""
        topo = split = ci = 0
        expected_nodecomm = mpi_comm.Split_type(MPI.COMM_TYPE_SHARED)

        try:
            topo, split, ci = _create_layout_via_pipeline(system_size=max(128, mpi_comm.Get_size()))

            handle = _ciw.get_nodecomm(ci)
            assert handle != MPI.COMM_NULL.py2f()

            nodecomm = MPI.Comm.f2py(handle)
            assert nodecomm.Get_size() == expected_nodecomm.Get_size()
            assert nodecomm.Get_rank() == expected_nodecomm.Get_rank()
        finally:
            expected_nodecomm.Free()
            if ci:
                _ciw.destroy(ci)
            if split:
                _ciw.destroy_split(split)
            if topo:
                _ciw.wrapper_destroy_topology(topo)

    def test_devcomm_default_null(self, mpi_comm):
        """DEVCOMM is MPI_COMM_NULL until explicitly set."""
        ci = _create_layout_handle()
        assert _ciw.get_devcomm(ci) == MPI.COMM_NULL.py2f()
        _ciw.destroy(ci)

    def test_devcomm_node_default_null(self, mpi_comm):
        """DEVCOMM_NODE is MPI_COMM_NULL until explicitly set."""
        ci = _create_layout_handle()
        assert _ciw.get_devcomm_node(ci) == MPI.COMM_NULL.py2f()
        _ciw.destroy(ci)


# =============================================================================
# Rebuild Communicators
# =============================================================================


@pytest.mark.mpi
class TestRebuildCommunicators:
    """rebuild_communicators follows the unlocked layout lifecycle."""

    def test_rebuild_is_callable(self, mpi_comm, mpi_rank, mpi_size):
        system_size = 100
        ci = _create_layout_handle()
        _ciw.set_system_size(ci, system_size)
        local_i, offset = _compute_partition(system_size, mpi_rank, mpi_size)
        _ciw.set_partitioning(ci, local_i, offset)

        assert _ciw.rebuild_communicators(ci) == 0
        _ciw.destroy(ci)

    def test_rebuild_locked_layout_returns_error(self, mpi_comm):
        ci = _create_layout_handle()
        _ciw.lock(ci)

        assert _ciw.rebuild_communicators(ci) == 1
        _ciw.destroy(ci)


# =============================================================================
# Locked Mutation Guards
# =============================================================================


@pytest.mark.mpi
class TestLockedMutationGuards:
    """Locked layouts reject direct mutation through the wrapper."""

    def test_scalar_setters_return_error_when_locked(self, mpi_comm, mpi_size):
        ci = _create_layout_handle()
        _ciw.lock(ci)

        assert _ciw.set_system_size(ci, 128) == 1
        assert _ciw.set_n_processes(ci, mpi_size) == 1
        assert _ciw.set_alloc_local(ci, 64) == 1

        _ciw.destroy(ci)

    def test_build_partition_table_locked_layout_returns_error(self, mpi_comm, mpi_rank, mpi_size):
        ci = _create_layout_handle()
        system_size = 100
        local_i, offset = _compute_partition(system_size, mpi_rank, mpi_size)

        assert _ciw.set_system_size(ci, system_size) == 0
        assert _ciw.set_partitioning(ci, local_i, offset) == 0
        _ciw.lock(ci)

        assert _ciw.build_partition_table(ci) == 1

        _ciw.destroy(ci)


# =============================================================================
# Full Workflow Integration
# =============================================================================


@pytest.mark.mpi
class TestFullWorkflow:
    """End-to-end workflow: create -> partition -> build -> validate -> destroy."""

    def test_complete_lifecycle(self, mpi_comm, mpi_rank, mpi_size):
        """Exercise the full typical usage pattern."""
        system_size = 200

        # 1. Create
        ci = _create_layout_handle()
        assert _ciw.is_locked(ci) == 0

        # 2. Set system size and partitioning
        assert _ciw.set_system_size(ci, system_size) == 0
        local_i, offset = _compute_partition(system_size, mpi_rank, mpi_size)
        assert _ciw.set_partitioning(ci, local_i, offset) == 0
        assert _ciw.set_alloc_local(ci, local_i) == 0

        # 3. Build partition table
        assert _ciw.build_partition_table(ci) == 0
        n = _ciw.get_partition_table_size(ci)
        assert n == mpi_size + 1
        table = _ciw.get_partition_table(ci, n)
        assert table[0] == 1
        assert table[-1] == system_size + 1

        # 4. Validate
        assert _ciw.validate(ci, system_size) == 0

        # 5. Verify fields via getters
        assert _ciw.get_n_processes(ci) == mpi_size
        assert _ciw.get_local_i(ci) == local_i
        assert _ciw.get_local_i_offset(ci) == offset

        # 6. Lock -> verify locked -> unlock
        _ciw.lock(ci)
        assert _ciw.is_locked(ci) == 1
        _ciw.unlock(ci)
        assert _ciw.is_locked(ci) == 0

        # 7. Destroy
        _ciw.destroy(ci)

    def test_repartition_via_setters(self, mpi_comm, mpi_rank, mpi_size):
        """After setting fields via setters, rebuild and re-validate succeeds."""
        system_size = 64
        ci = _create_layout_handle()
        assert _ciw.set_system_size(ci, system_size) == 0

        local_i, offset = _compute_partition(system_size, mpi_rank, mpi_size)
        assert _ciw.set_partitioning(ci, local_i, offset) == 0
        assert _ciw.set_alloc_local(ci, local_i) == 0
        assert _ciw.build_partition_table(ci) == 0

        # Read fields from first layout
        n_procs = _ciw.get_n_processes(ci)
        li = _ciw.get_local_i(ci)
        lio = _ciw.get_local_i_offset(ci)

        # Create a second layout and populate via setters
        ci2 = _create_layout_handle()
        assert _ciw.set_system_size(ci2, system_size) == 0
        assert _ciw.set_n_processes(ci2, n_procs) == 0
        assert _ciw.set_partitioning(ci2, li, lio) == 0
        assert _ciw.set_alloc_local(ci2, li) == 0
        assert _ciw.build_partition_table(ci2) == 0
        assert _ciw.validate(ci2, system_size) == 0

        # Fields should match
        assert _ciw.get_local_i(ci2) == local_i
        assert _ciw.get_local_i_offset(ci2) == offset
        assert _ciw.get_system_size(ci2) == system_size

        _ciw.destroy(ci)
        _ciw.destroy(ci2)

    @pytest.mark.requires_nprocs(2)
    def test_multiple_system_sizes(self, mpi_comm, mpi_rank, mpi_size):
        """Re-use a layout with different system sizes (re-partition)."""
        ci = _create_layout_handle()

        for system_size in [50, 100, 97, 256, 1]:
            assert _ciw.set_system_size(ci, system_size) == 0
            local_i, offset = _compute_partition(system_size, mpi_rank, mpi_size)
            assert _ciw.set_partitioning(ci, local_i, offset) == 0
            assert _ciw.set_alloc_local(ci, local_i) == 0
            assert _ciw.build_partition_table(ci) == 0
            assert _ciw.validate(ci, system_size) == 0

            assert _ciw.get_system_size(ci) == system_size
            assert _ciw.get_local_i(ci) == local_i
            assert _ciw.get_local_i_offset(ci) == offset

            n = _ciw.get_partition_table_size(ci)
            table = _ciw.get_partition_table(ci, n)
            assert table[0] == 1
            assert table[-1] == system_size + 1

        _ciw.destroy(ci)
