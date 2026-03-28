import numpy as np
from mpi4py import MPI

from quop_mpi._lib.comm_info_wrapper import comm_info_wrapper as _ciw


def _unwrap_pointer_status(result, operation):
    """Return a native pointer from an f2py `(ptr, status)` result tuple."""
    ptr, status = result
    ptr = int(ptr)
    status = int(status)
    if status != 0:
        if operation == "create":
            raise RuntimeError(f"Fortran comm_info create failed with status {status}")
        if operation == "discover_topology":
            raise RuntimeError(f"Fortran discover_topology failed with status {status}")
        raise RuntimeError(f"Fortran {operation} failed with status {status}")
    return ptr


# =============================================================================
# QuopMpiLayout  -- thin Python wrapper around the Fortran quop_mpi_layout_t
# =============================================================================


class QuopMpiLayout:
    """Single source of truth for statevector partitioning and worker configuration.

    Wraps a Fortran-level ``quop_mpi_layout_t`` instance exposed as an opaque
    ``c_ptr`` handle, plus ``split_info_t`` for worker-level communicators.
    All properties delegate to Fortran accessor functions -- there is no
    Python-side copy of the data.

    Parameters
    ----------
    MPI_COMM : MPI.Intracomm
        Root communicator.  Stored on both the Python side (for guards)
        and the Fortran side (as the invariant root).  The caller owns
        this communicator -- it is never freed by ``QuopMpiLayout``.
    system_size : int, optional
        Total problem size.  If not supplied, defaults to 0.
    """

    __slots__ = (
        "_ptr",
        "_comm",
        "_system_size",
        "_owns_handle",
        "_split_ptr",
        "_topo_ptr",
        "_worker_id",
        "_n_workers",
        "_roots",
    )

    def __init__(self, MPI_COMM, system_size=0):  # noqa: N803
        self._ptr = None
        self._comm = MPI_COMM
        self._system_size = int(system_size)
        self._owns_handle = True
        self._split_ptr = None
        self._topo_ptr = None
        self._worker_id = 0
        self._n_workers = 1
        self._roots = None

        self._ptr = _unwrap_pointer_status(
            _ciw.create(MPI_COMM.py2f()),
            "create",
        )
        try:
            self.set_system_size(self._system_size)
        except Exception:
            _ciw.destroy(self._ptr)
            self._ptr = None
            raise

    @classmethod
    def from_partition(
        cls,
        MPI_COMM,  # noqa: N803
        system_size,
        local_i,
        local_i_offset,
        alloc_local=None,
    ):
        """Create a layout handle populated with explicit partition fields."""
        obj = cls(MPI_COMM, system_size=system_size)
        try:
            obj.set_partitioning(int(local_i), int(local_i_offset))
            if alloc_local is None:
                alloc_local = local_i
            obj.set_alloc_local(int(alloc_local))
        except Exception:
            obj.destroy()
            raise
        return obj

    @classmethod
    def from_handle(cls, ptr, MPI_COMM, split_ptr=None, topo_ptr=None):  # noqa: N803
        """Wrap existing Fortran quop_mpi_layout_t and split_info_t handles.

        Used by Ansatz after Fortran-side negotiate() returns a layout_ptr.
        The handles are owned by this object -- ``destroy()`` will free them.

        Parameters
        ----------
        ptr : int
            Opaque pointer to Fortran quop_mpi_layout_t.
        MPI_COMM : MPI.Intracomm
            Root (world) communicator.
        split_ptr : int, optional
            Opaque pointer to Fortran split_info_t (for worker split info).
        topo_ptr : int, optional
            Opaque pointer to Fortran gpu_topology_t.
        """
        obj = cls.__new__(cls)
        obj._ptr = int(ptr) if ptr else None
        obj._comm = MPI_COMM
        obj._system_size = 0
        obj._owns_handle = True
        obj._split_ptr = int(split_ptr) if split_ptr else None
        obj._topo_ptr = int(topo_ptr) if topo_ptr else None
        obj._roots = None

        # Extract worker_id and n_workers from split_info_t
        if obj._split_ptr:
            obj._worker_id = int(_ciw.get_worker_id(obj._split_ptr))
            obj._n_workers = int(_ciw.get_n_workers(obj._split_ptr))
        else:
            obj._worker_id = 0
            obj._n_workers = 1

        return obj

    def set_layout_ptr(self, ptr):
        """Set the layout pointer after negotiate completes.

        Called after Fortran negotiate returns a layout_ptr. This allows
        the same QuopMpiLayout object to be used before and after negotiate.

        Parameters
        ----------
        ptr : int
            Opaque pointer to Fortran quop_mpi_layout_t.
        """
        if self._ptr is not None:
            raise RuntimeError("Layout pointer already set")
        self._ptr = int(ptr) if ptr else None

    def mark_excluded(self):
        """Mark this rank as excluded (e.g., after communicator shrink).

        Sets worker_id to -1 so in_subcomm() returns False.
        """
        self._worker_id = -1

    @classmethod
    def create_workers(cls, n_workers, MPI_COMM, backend_flag=None):  # noqa: N803
        """Create a layout by calling Fortran split_workers directly.

        This creates the topology and split_info_t but NOT the layout_ptr
        (which requires negotiate). The layout_ptr will be set later after
        negotiate completes.

        On the wavefront backend, ``n_workers`` is clamped to the total
        number of device-rank slots so that every subcomm is guaranteed
        at least one GPU.  The effective (possibly clamped) worker count
        is available via ``get_n_subcomms()``.

        Raises ``RuntimeError`` if a heterogeneous GPU topology is
        detected (different device-slot counts across nodes).

        Parameters
        ----------
        n_workers : int
            Number of worker subcommunicators to create.
        MPI_COMM : MPI.Intracomm
            Parent (world) communicator.
        backend_flag : int, optional
            Backend flag (0 = MPI, 1 = wavefront). If ``None`` (default),
            auto-detected from ``quop_mpi.config.backend``.

        Returns
        -------
        QuopMpiLayout
            A partial layout with split_ptr and topo_ptr set, but _ptr=None.
        """
        if backend_flag is None:
            from quop_mpi import config

            backend_flag = 1 if config.backend == "wavefront" else 0

        topo_ptr = 0
        split_ptr = 0
        # Phase 0: Discover topology
        topo_ptr = _unwrap_pointer_status(
            _ciw.wrapper_discover_topology(MPI_COMM.py2f(), backend_flag),
            "discover_topology",
        )

        try:
            # On wavefront, cap workers to total device-rank slots so every
            # subcomm is guaranteed at least one GPU.
            if backend_flag == 1 and n_workers > 1:
                n_gpus, rpg, node_size = _ciw.wrapper_get_topology_info(topo_ptr)
                n_gpus = int(n_gpus)
                rpg = int(rpg)
                node_size = int(node_size)
                device_slots = n_gpus * max(rpg, 1)

                # Reject heterogeneous GPU topology — the GPU-aware
                # worker split assumes every node has the same number
                # of device slots.
                ds = np.array([device_slots], dtype=np.int32)
                min_ds = np.zeros(1, dtype=np.int32)
                max_ds = np.zeros(1, dtype=np.int32)
                MPI_COMM.Allreduce(ds, min_ds, op=MPI.MIN)
                MPI_COMM.Allreduce(ds, max_ds, op=MPI.MAX)
                if min_ds[0] != max_ds[0]:
                    raise RuntimeError(
                        f"Heterogeneous GPU topology detected: device "
                        f"slots per node range from {min_ds[0]} to "
                        f"{max_ds[0]}. The wavefront backend requires "
                        f"uniform GPU configuration across all nodes."
                    )

                # Each of the node_size co-located ranks contributes
                # (device_slots / node_size) so the sum across all ranks
                # equals the total device slots cluster-wide.
                if device_slots > 0 and node_size > 0:
                    local_slots = np.array(
                        [float(device_slots) / node_size], dtype=np.float64
                    )
                else:
                    local_slots = np.zeros(1, dtype=np.float64)
                total_slots = np.zeros(1, dtype=np.float64)
                MPI_COMM.Allreduce(local_slots, total_slots, op=MPI.SUM)
                max_workers = int(total_slots[0] + 0.5)
                if max_workers > 0 and n_workers > max_workers:
                    n_workers = max_workers

            # Phase 0b: Split into worker groups
            split_ptr, worker_id, status = _ciw.wrapper_split_workers(
                MPI_COMM.py2f(),
                topo_ptr,
                np.int32(n_workers),
                np.int32(backend_flag),
            )
            split_ptr = int(split_ptr)

            if status != 0:
                raise RuntimeError(f"Fortran split_workers failed with status {status}")

            # Create layout wrapper (no layout_ptr yet - set after negotiate)
            # ROOTCOMM is already created in Fortran by split_workers.
            return cls.from_handle(None, MPI_COMM, split_ptr=split_ptr, topo_ptr=topo_ptr)
        except Exception:
            if split_ptr:
                _ciw.destroy_split(split_ptr)
            if topo_ptr:
                _ciw.wrapper_destroy_topology(topo_ptr)
            raise

    # -- Named field access (all delegate to Fortran) ----------------

    @property
    def system_size(self):
        if self._ptr is None:
            return 0
        return int(_ciw.get_system_size(self._ptr))

    @property
    def n_processes(self):
        if self._ptr is None:
            return 0
        return int(_ciw.get_n_processes(self._ptr))

    @property
    def local_i(self):
        if self._ptr is None:
            return 0
        return int(_ciw.get_local_i(self._ptr))

    @property
    def local_i_offset(self):
        if self._ptr is None:
            return 0
        return int(_ciw.get_local_i_offset(self._ptr))

    @property
    def alloc_local(self):
        if self._ptr is None:
            return 0
        return int(_ciw.get_alloc_local(self._ptr))

    @property
    def device_alloc_local(self):
        if self._ptr is None:
            return 0
        return int(_ciw.get_device_alloc_local(self._ptr))

    @property
    def device_n_processes(self):
        if self._ptr is None:
            return 0
        return int(_ciw.get_device_n_processes(self._ptr))

    @property
    def device_local_i(self):
        if self._ptr is None:
            return 0
        return int(_ciw.get_device_local_i(self._ptr))

    @property
    def device_local_i_offset(self):
        if self._ptr is None:
            return 0
        return int(_ciw.get_device_local_i_offset(self._ptr))

    @property
    def partition_table(self):
        """Partition table, or ``None`` if not yet built."""
        if self._ptr is None:
            return None
        n = int(_ciw.get_partition_table_size(self._ptr))
        if n == 0:
            return None
        return _ciw.get_partition_table(self._ptr, n)

    @property
    def comm(self):
        """The root MPI_COMM ('world') -- never changes."""
        return self._comm

    # -- Communicator access (Fortran int32 -> mpi4py Intracomm) ------

    @property
    def mpi_comm(self):
        """The root 'world' communicator from the Fortran side."""
        if self._ptr is None:
            return self._comm
        handle = int(_ciw.get_mpi_comm(self._ptr))
        return MPI.Comm.f2py(handle) if handle != MPI.COMM_NULL.py2f() else None

    @property
    def subcomm(self):
        """The working communicator (may shrink after negotiate)."""
        # After negotiate: use layout's SUBCOMM (may be shrunken)
        if self._ptr is not None:
            handle = int(_ciw.get_subcomm(self._ptr))
            return MPI.Comm.f2py(handle) if handle != MPI.COMM_NULL.py2f() else None
        # Before negotiate: use split_info_t's SUBCOMM
        if self._split_ptr is not None:
            handle = int(_ciw.get_split_subcomm(self._split_ptr))
            return MPI.Comm.f2py(handle) if handle != MPI.COMM_NULL.py2f() else None
        return None

    @property
    def nodecomm(self):
        if self._ptr is None:
            return None
        handle = int(_ciw.get_nodecomm(self._ptr))
        return MPI.Comm.f2py(handle) if handle != MPI.COMM_NULL.py2f() else None

    @property
    def devcomm(self):
        if self._ptr is None:
            return None
        handle = int(_ciw.get_devcomm(self._ptr))
        return MPI.Comm.f2py(handle) if handle != MPI.COMM_NULL.py2f() else None

    @property
    def devcomm_node(self):
        if self._ptr is None:
            return None
        handle = int(_ciw.get_devcomm_node(self._ptr))
        return MPI.Comm.f2py(handle) if handle != MPI.COMM_NULL.py2f() else None

    # -- Worker-level communicator access (from split_info_t) --------

    @property
    def jaccomm(self):
        """Jacobian communicator (workers + optimiser leader), or None."""
        if self._split_ptr is None:
            return None
        handle = int(_ciw.get_jaccomm(self._split_ptr))
        return MPI.Comm.f2py(handle) if handle != MPI.COMM_NULL.py2f() else None

    @property
    def rootcomm(self):
        """Communicator connecting subcomm leaders, or None."""
        if self._split_ptr is not None:
            handle = int(_ciw.get_rootcomm(self._split_ptr))
            if handle != MPI.COMM_NULL.py2f():
                return MPI.Comm.f2py(handle)
        return None

    @property
    def worker_id(self):
        """This rank's worker ID (subcomm index)."""
        return self._worker_id

    @property
    def n_workers(self):
        """Number of worker subcommunicators."""
        return self._n_workers

    # -- Compatibility aliases (uppercase) ---------------------------

    @property
    def SUBCOMM(self):  # noqa: N802
        """Alias for ``subcomm`` (compatibility)."""
        return self.subcomm

    @property
    def JACCOMM(self):  # noqa: N802
        """Alias for ``jaccomm`` (compatibility)."""
        return self.jaccomm

    @property
    def ROOTCOMM(self):  # noqa: N802
        """Alias for ``rootcomm`` (compatibility)."""
        return self.rootcomm

    @property
    def MPI_COMM(self):  # noqa: N802
        """Alias for ``comm`` (compatibility)."""
        return self._comm

    # -- Worker-level query methods ----------------------------------

    def get_n_subcomms(self):
        """Return number of worker subcommunicators."""
        return self._n_workers

    def get_subcomm_index(self):
        """Return this rank's worker ID (colour)."""
        return self._worker_id

    def get_subcomm_roots(self):
        """Return world ranks of each SUBCOMM leader.

        Returns a list of length n_workers where element i is the world rank
        of rank 0 in worker i's SUBCOMM.

        This is computed on-demand via Allgather.
        """
        if self._roots is not None:
            return self._roots

        # Gather (worker_id, world_rank) for all ranks then extract roots
        world_rank = self._comm.Get_rank()
        local_info = np.array([self._worker_id, world_rank], dtype=np.int32)
        all_info = np.empty(self._comm.Get_size() * 2, dtype=np.int32)
        self._comm.Allgather(local_info, all_info)

        # Reshape to (n_ranks, 2), each row is (worker_id, world_rank)
        all_info = all_info.reshape(-1, 2)

        # Find root (rank 0 in SUBCOMM) for each worker
        # Root is the rank with lowest world_rank for each worker_id
        roots = [-1] * self._n_workers
        for wid, wrank in all_info:
            if wid >= 0:
                if roots[wid] == -1 or wrank < roots[wid]:
                    roots[wid] = wrank

        self._roots = roots
        return roots

    def get_root_world(self):
        """Return world rank of worker 0's root (or 0 if undefined)."""
        roots = self.get_subcomm_roots()
        return roots[0] if roots and roots[0] >= 0 else 0

    @staticmethod
    def _raise_layout_status(action, error_code, status_map=None):
        """Raise a Python exception for a non-zero native status code."""
        if error_code == 0:
            return

        if status_map is None:
            status_map = {}

        exc_type, message = status_map.get(
            error_code,
            (
                RuntimeError,
                f"Cannot {action}: backend returned status {error_code}",
            ),
        )
        raise exc_type(message)

    def _require_layout_handle(self, action):
        """Raise if the Fortran layout handle is not available."""
        if self._ptr is None:
            raise RuntimeError(f"Cannot {action}: layout not negotiated.")

    # -- Mutators ----------------------------------------------------

    def set_system_size(self, system_size):
        """Set the global logical system size on an unlocked layout."""
        self._require_layout_handle("set system size")
        error_code = _ciw.set_system_size(self._ptr, int(system_size))
        self._raise_layout_status(
            "set system size",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot set system size on a locked layout",
                ),
            },
        )

    def set_n_processes(self, n_processes):
        """Set the active rank count on an unlocked layout."""
        self._require_layout_handle("set n_processes")
        error_code = _ciw.set_n_processes(self._ptr, int(n_processes))
        self._raise_layout_status(
            "set n_processes",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot set n_processes on a locked layout",
                ),
                2: (
                    ValueError,
                    "Cannot set n_processes outside the valid SUBCOMM range",
                ),
            },
        )

    def set_partitioning(self, local_i, local_i_offset):
        """Set host partitioning on an unlocked layout."""
        self._require_layout_handle("set partitioning")
        error_code = _ciw.set_partitioning(
            self._ptr,
            int(local_i),
            int(local_i_offset),
        )
        self._raise_layout_status(
            "set partitioning",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot set partitioning on a locked layout",
                ),
            },
        )

    def set_alloc_local(self, alloc_local):
        """Set the required host allocation length on an unlocked layout."""
        self._require_layout_handle("set alloc_local")
        error_code = _ciw.set_alloc_local(self._ptr, int(alloc_local))
        self._raise_layout_status(
            "set alloc_local",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot set alloc_local on a locked layout",
                ),
            },
        )

    # -- Lock / unlock -----------------------------------------------

    def lock(self):
        self._require_layout_handle("lock layout")
        error_code = _ciw.lock(self._ptr)
        self._raise_layout_status(
            "lock layout",
            error_code,
            {
                1: (RuntimeError, "Layout is already locked"),
            },
        )

    def unlock(self):
        self._require_layout_handle("unlock layout")
        error_code = _ciw.unlock(self._ptr)
        self._raise_layout_status(
            "unlock layout",
            error_code,
            {
                1: (RuntimeError, "Layout is already unlocked"),
            },
        )

    @property
    def is_locked(self):
        if self._ptr is None:
            return False
        return bool(_ciw.is_locked(self._ptr))

    # -- Construction helpers ----------------------------------------

    def build_partition_table(self):
        """Collective over SUBCOMM: Allgather ``local_i`` to build the table."""
        self._require_layout_handle("build partition table")
        error_code = _ciw.build_partition_table(self._ptr)
        self._raise_layout_status(
            "build partition table",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot build partition table on a locked layout",
                ),
                2: (
                    RuntimeError,
                    "Cannot build partition table without a valid SUBCOMM",
                ),
            },
        )

    # -- Communicator management -------------------------------------

    def shrink(self, new_size):
        """Collective: shrink SUBCOMM to *new_size* ranks."""
        self._require_layout_handle("shrink layout")
        error_code = _ciw.shrink(self._ptr, int(new_size))
        self._raise_layout_status(
            "shrink layout",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot shrink a locked layout",
                ),
                2: (
                    ValueError,
                    "Cannot shrink outside the valid SUBCOMM range",
                ),
                3: (
                    RuntimeError,
                    "Cannot shrink layout: failed to rebuild"
                    " partition_table after communicator resize",
                ),
            },
        )

    def rebuild_communicators(self):
        """Collective over SUBCOMM: rebuild DEVCOMM/DEVCOMM_NODE before lock."""
        self._require_layout_handle("rebuild communicators")
        error_code = _ciw.rebuild_communicators(self._ptr)
        self._raise_layout_status(
            "rebuild communicators",
            error_code,
            {
                1: (
                    RuntimeError,
                    "Cannot rebuild communicators on a locked layout",
                ),
            },
        )

    # -- Communicator guards -----------------------------------------

    def in_subcomm(self):
        """True if this rank is a member of SUBCOMM."""
        return self._worker_id >= 0 and self.subcomm is not None

    def in_rootcomm(self):
        """True if this rank is a subcomm leader (rank 0 of SUBCOMM)."""
        if not self.in_subcomm():
            return False
        return self.subcomm.Get_rank() == 0

    def assert_matches_comm(self, MPI_COMM, label=""):  # noqa: N803
        """Raise ``RuntimeError`` if *MPI_COMM* does not match the root comm."""
        if self._comm == MPI.COMM_NULL or MPI_COMM == MPI.COMM_NULL:
            raise RuntimeError(
                f"QuopMpiLayout.assert_matches_comm"
                f"{f' ({label})' if label else ''}: "
                f"one of the communicators is MPI_COMM_NULL"
            )
        result = MPI.Comm.Compare(self._comm, MPI_COMM)
        if result not in (MPI.IDENT, MPI.CONGRUENT):
            raise RuntimeError(
                f"QuopMpiLayout.assert_matches_comm"
                f"{f' ({label})' if label else ''}: "
                f"communicator mismatch (MPI_Comm_compare={result})"
            )

    # -- Validation (collective) --------------------------------------

    # Bit flags returned by the Fortran layout_validate (must match
    # the LAYOUT_ERR_* parameters in comm_info_module.f90).
    _VALIDATE_FLAGS = {
        1: "non-negative check",
        2: "completeness",
        4: "rank_ordering",
        8: "contiguity",
        16: "node_contiguity",
        32: "device_ordering",
        64: "device_completeness",
    }

    def validate(self, system_size):
        """Collective over SUBCOMM: validate layout consistency.

        Delegates all checking logic to Fortran (``layout_validate``),
        which returns a bitmask error code (0 = all OK).  The bitmask is
        Allreduced inside Fortran so **all** ranks receive the same value,
        preventing MPI deadlocks inside ``pytest.raises`` blocks.

        Raises
        ------
        ValueError
            If any validation check fails.  The message lists which
            checks failed.
        """
        if self._ptr is None:
            raise RuntimeError("Cannot validate: layout not negotiated.")

        error_code = _ciw.validate(self._ptr, int(system_size))
        if error_code != 0:
            failed = [name for flag, name in self._VALIDATE_FLAGS.items() if error_code & flag]
            raise ValueError("Layout validation failed: " + ", ".join(failed))

    @property
    def handle(self):
        """The raw Fortran c_ptr (int64) for passing to other Fortran routines."""
        return self._ptr

    @property
    def split_ptr(self):
        """The raw Fortran split_info_t c_ptr (int64)."""
        return self._split_ptr

    @property
    def topo_ptr(self):
        """The raw Fortran gpu_topology_t c_ptr (int64)."""
        return self._topo_ptr

    def get_topology_info(self):
        """Return key GPU topology fields.

        When a negotiated layout exists, this returns the current topology
        cached on the Fortran ``quop_mpi_layout_t``. Before negotiate, it
        falls back to the original discovery-time topology handle.

        Returns
        -------
        dict
            ``n_physical_gpus`` : int
                Number of unique physical GPUs detected on this rank's node.
            ``ranks_per_gpu`` : int
                Current ``QUOP_RANKS_PER_GPU`` value (default 1).
            ``node_size`` : int
                Number of MPI ranks on this rank's node.
        """
        if self._ptr is not None and self._ptr != 0:
            n_physical_gpus, ranks_per_gpu, node_size = _ciw.wrapper_get_layout_topology_info(
                self._ptr
            )
        elif self._topo_ptr is not None and self._topo_ptr != 0:
            n_physical_gpus, ranks_per_gpu, node_size = _ciw.wrapper_get_topology_info(
                self._topo_ptr
            )
        else:
            return {"n_physical_gpus": 0, "ranks_per_gpu": 1, "node_size": 0}
        return {
            "n_physical_gpus": int(n_physical_gpus),
            "ranks_per_gpu": int(ranks_per_gpu),
            "node_size": int(node_size),
        }

    # -- Cleanup -----------------------------------------------------

    def destroy(self):
        """Free the Fortran-side handles.  Idempotent."""
        if self._ptr is not None and self._ptr != 0:
            _ciw.destroy(self._ptr)
            self._ptr = None
        if self._split_ptr is not None and self._split_ptr != 0:
            _ciw.destroy_split(self._split_ptr)
            self._split_ptr = None
        if self._topo_ptr is not None and self._topo_ptr != 0:
            _ciw.wrapper_destroy_topology(self._topo_ptr)
            self._topo_ptr = None

    def free(self):
        """Alias for ``destroy()`` (compatibility with subcomms)."""
        self.destroy()

    def __enter__(self):
        """Return ``self`` so layouts can be scoped with ``with``."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Deterministically release native layout handles at scope exit."""
        self.destroy()
        return False

    def __repr__(self):
        if self._ptr is None or self._ptr == 0:
            return "QuopMpiLayout(<destroyed>)"
        rank = self._comm.Get_rank() if self._comm != MPI.COMM_NULL else "?"
        return (
            f"QuopMpiLayout(rank={rank}, n_procs={self.n_processes}, "
            f"local_i={self.local_i}, offset={self.local_i_offset})"
        )
