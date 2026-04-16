! comm_info_module.f90
!
! Central module for QuOp_MPI layout types: quop_mpi_layout_t and split_info_t.
!
! quop_mpi_layout_t owns the per-worker communicator hierarchy, partitioning,
! partition table, and lock state for one worker group.
!
! split_info_t holds the per-worker SUBCOMM (ownership transfers to
! quop_mpi_layout_t during negotiate()) and the JACCOMM.
!
! Stage 1 (MPI backend): Device fields are plain integers (zero for MPI).
!          NODECOMM is created for active SUBCOMM ranks; DEVCOMM/DEVCOMM_NODE
!          remain MPI_COMM_NULL.
! Stage 6 (wavefront backend, WAVEFRONT_BACKEND defined): GPU topology is
!          detected during discover_topology(), and NODECOMM/DEVCOMM/
!          DEVCOMM_NODE are created during negotiate() and managed through
!          shrink/rebuild.

module comm_info_module

    use, intrinsic :: iso_fortran_env, only: int32, int64, real64, error_unit
    use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_f_pointer, c_null_ptr, &
                                                                  c_associated, c_funptr, c_f_procpointer, c_null_funptr
    use MPI
    use gpu_topology, only: gpu_topology_t
#ifdef WAVEFRONT_BACKEND
    use gpu_topology, only: init_gpu_topology
    use communicators, only: create_devcomm_with_topology, create_devcomm_with_data
    use partitions, only: DEVCOMM_NODE_layout_from_DEVCOMM, &
                          NODECOMM_layout_from_DEVCOMM_NODE
#endif

    implicit none
    private
    public :: quop_mpi_layout_t, split_info_t, negotiate_callback_iface
    public :: discover_topology, destroy_topology, get_topology_info, get_layout_topology_info, split_workers, negotiate
    public :: create_jaccomm, create_rootcomm
    public :: create_split_from_subcomm
    public :: init_explicit_wavefront_layout
    public :: dump_comm_info
#ifdef WAVEFRONT_BACKEND
    public :: sync_layout_from_device_partition
#endif

    ! -- Validation error-code bit flags (public for wrapper) --------
    integer(int32), parameter, public :: LAYOUT_ERR_NON_NEGATIVE = 1 ! bit 0
    integer(int32), parameter, public :: LAYOUT_ERR_COMPLETENESS = 2 ! bit 1
    integer(int32), parameter, public :: LAYOUT_ERR_RANK_ORDERING = 4 ! bit 2
    integer(int32), parameter, public :: LAYOUT_ERR_CONTIGUITY = 8 ! bit 3
    integer(int32), parameter, public :: LAYOUT_ERR_NODE_CONTIGUITY = 16 ! bit 4
    integer(int32), parameter, public :: LAYOUT_ERR_DEVICE_ORDERING = 32 ! bit 5
    integer(int32), parameter, public :: LAYOUT_ERR_DEVICE_COMPLETENESS = 64 ! bit 6

    ! -- Abstract interface for negotiate callbacks ------------------
    ! Each propagator_wrapper variant provides a bind(C) trampoline
    ! conforming to this interface.  negotiate() calls through
    ! c_funptr handles to dispatch max_comm_size on each propagator.
    abstract interface
        subroutine negotiate_callback_iface(prop_ptr, ci_ptr, error_code) bind(C)
            import :: c_ptr, int32
            type(c_ptr), value, intent(in) :: prop_ptr
            type(c_ptr), value, intent(in) :: ci_ptr
            integer(int32), intent(out) :: error_code
        end subroutine negotiate_callback_iface
    end interface

    ! -- split_info_t ------------------------------------------------
    ! Created by split_workers().  Holds the per-worker SUBCOMM
    ! (ownership transfers to quop_mpi_layout_t during negotiate())
    ! and the JACCOMM (populated by create_jaccomm() after negotiate).
    type :: split_info_t
        integer(int32) :: MPI_COMM = MPI_COMM_NULL ! parent comm (NOT owned)
        integer(int32) :: SUBCOMM = MPI_COMM_NULL
        integer(int32) :: JACCOMM = MPI_COMM_NULL
        integer(int32) :: ROOTCOMM = MPI_COMM_NULL
        integer(int32) :: worker_id = 0
        integer(int32) :: n_workers = 1
    contains
        procedure :: destroy => split_info_destroy
    end type split_info_t

    ! -- quop_mpi_layout_t ------------------------------------------
    type :: quop_mpi_layout_t

        private

        ! -- Lock state ----------------------------------------------
        logical :: locked = .false.
        integer(int32) :: backend_flag = 0

        ! -- Core partitioning ---------------------------------------
        integer(int64) :: system_size = 0
        integer(int64) :: n_processes = 0
        integer(int64) :: local_i = 0
        integer(int64) :: local_i_offset = 0
        integer(int64) :: alloc_local = 0

        ! -- Device partitioning (wavefront only, zero for MPI) ------
        integer(int64) :: device_n_processes = 0
        integer(int64) :: device_local_i = 0
        integer(int64) :: device_local_i_offset = 0
        integer(int64) :: device_alloc_local = 0

        ! -- Communicators -------------------------------------------
        integer(int32) :: MPI_COMM = MPI_COMM_NULL
        integer(int32) :: SUBCOMM = MPI_COMM_NULL
        integer(int32) :: NODECOMM = MPI_COMM_NULL
        integer(int32) :: DEVCOMM = MPI_COMM_NULL
        integer(int32) :: DEVCOMM_NODE = MPI_COMM_NULL

        ! -- GPU topology -----------------------------------------
        ! discover_topology() provides the hardware-intrinsic fields on
        ! MPI_COMM. negotiate() copies those invariants here, then refreshes
        ! communicator-derived fields (node rank/size, active node id/count,
        ! devcomm-node size, and local rank indices) whenever SUBCOMM or its
        ! children are rebuilt. For the MPI backend, GPU-specific fields stay
        ! at zero/false defaults.
        type(gpu_topology_t) :: topology

        ! -- Partition table (computed once, shared everywhere) ------
        integer(int64), allocatable :: partition_table(:)

        ! -- Whether any of the propagators requires a device work buffer (wavefront only, false for MPI) --
        logical :: requires_device_work_buffer = .false.

    contains

        ! Lifecycle
        procedure :: lock => layout_lock
        procedure :: unlock => layout_unlock
        procedure :: is_locked => layout_is_locked
        procedure :: destroy => layout_destroy

        ! Population
        procedure :: set_MPI_COMM => layout_set_MPI_COMM
        procedure :: set_SUBCOMM => layout_set_SUBCOMM
        procedure :: set_NODECOMM => layout_set_NODECOMM
        procedure :: set_DEVCOMM => layout_set_DEVCOMM
        procedure :: set_DEVCOMM_NODE => layout_set_DEVCOMM_NODE
        procedure :: set_system_size => layout_set_system_size
        procedure :: set_n_processes => layout_set_n_processes
        procedure :: set_partitioning => layout_set_partitioning
        procedure :: set_alloc_local => layout_set_alloc_local
        procedure :: set_device_alloc_local => layout_set_device_alloc_local
        procedure :: set_device_n_processes => layout_set_device_n_processes
        procedure :: set_topology => layout_set_topology
        procedure :: build_partition_table => layout_build_partition_table

        ! Validation (collective)
        procedure :: validate => layout_validate
        procedure :: validate_non_negative => layout_validate_non_negative
        procedure :: validate_completeness => layout_validate_completeness
        procedure :: validate_rank_ordering => layout_validate_rank_ordering
        procedure :: validate_contiguity => layout_validate_contiguity
        procedure :: validate_node_contiguity => layout_validate_node_contiguity
        procedure :: validate_device_ordering => layout_validate_device_ordering
        procedure :: validate_device_completeness => layout_validate_device_completeness

        ! Communicator management
        procedure :: shrink => layout_shrink
        procedure :: filter_active_ranks => layout_filter_active_ranks
        procedure :: rebuild_communicators => layout_rebuild_communicators

        ! Getters
        procedure :: get_system_size => layout_get_system_size
        procedure :: get_n_processes => layout_get_n_processes
        procedure :: get_local_i => layout_get_local_i
        procedure :: get_local_i_offset => layout_get_local_i_offset
        procedure :: get_alloc_local => layout_get_alloc_local
        procedure :: get_device_local_i => layout_get_device_local_i
        procedure :: get_device_local_i_offset => layout_get_device_local_i_offset
        procedure :: get_device_alloc_local => layout_get_device_alloc_local
        procedure :: get_device_n_processes => layout_get_device_n_processes
        procedure :: get_SUBCOMM => layout_get_SUBCOMM
        procedure :: get_NODECOMM => layout_get_NODECOMM
        procedure :: get_DEVCOMM => layout_get_DEVCOMM
        procedure :: get_DEVCOMM_NODE => layout_get_DEVCOMM_NODE
        procedure :: get_MPI_COMM => layout_get_MPI_COMM
        procedure :: get_partition_table => layout_get_partition_table
        procedure :: get_topology => layout_get_topology
        procedure :: get_requires_device_work_buffer => layout_get_requires_device_work_buffer

        ! Additional setters
        procedure :: set_requires_device_work_buffer => layout_set_requires_device_work_buffer

        ! Diagnostics
        procedure :: dump => layout_dump_comm_info

    end type quop_mpi_layout_t

contains

    subroutine create_layout_nodecomm(COMM, node_comm)
        !! Create a node-local communicator from COMM.
        !! This helper is backend-independent and is used by the negotiated
        !! layout on both MPI and wavefront backends.
        integer(int32), intent(in) :: COMM
        integer(int32), intent(out) :: node_comm
        integer(int32) :: rank, ierr

        if (COMM == MPI_COMM_NULL) then
            node_comm = MPI_COMM_NULL
            return
        end if

        call MPI_Comm_rank(COMM, rank, ierr)
        call MPI_Comm_split_type(COMM, MPI_COMM_TYPE_SHARED, rank, &
                                 MPI_INFO_NULL, node_comm, ierr)
    end subroutine create_layout_nodecomm

    subroutine layout_note_error(error_code, code, message)
        integer(int32), intent(in) :: code
        integer(int32), intent(out) :: error_code
        character(len=*), intent(in) :: message

        error_code = code
    end subroutine layout_note_error

    subroutine layout_sync_precondition_error(comm, local_error, error_code)
        integer(int32), intent(in) :: comm
        integer(int32), intent(inout) :: local_error
        integer(int32), intent(out) :: error_code
        integer(int32) :: ierr, synced_error

        synced_error = local_error
        ! Synchronize over the same communicator as the collective entry
        ! point so excluded ranks do not need to participate after shrink().
        if (comm /= MPI_COMM_NULL) then
            call MPI_Allreduce(local_error, synced_error, 1, MPI_INTEGER4, &
                               MPI_MAX, comm, ierr)
        end if

        local_error = synced_error
        error_code = synced_error
    end subroutine layout_sync_precondition_error

    subroutine refresh_layout_topology(self)
        class(quop_mpi_layout_t), intent(inout) :: self

        integer(int32) :: ierr
        integer(int32) :: sub_rank, node_rank, node_size
        integer(int32) :: is_leader, active_node_id, active_n_nodes
        integer(int32) :: local_active_gpu_rank
        integer(int32) :: i
        integer(int32), allocatable :: rank_numa_nodes(:), rank_gpu_indices(:)

        if (self%SUBCOMM == MPI_COMM_NULL .or. self%NODECOMM == MPI_COMM_NULL) then
            self%topology%node_rank = -1
            self%topology%node_size = 0
            self%topology%devcomm_node_size = 0
            self%topology%rank_within_cpu_numa = 0
            self%topology%rank_within_gpu = 0
            self%topology%node_id = -1
            self%topology%n_nodes = 0
            return
        end if

        call MPI_Comm_rank(self%SUBCOMM, sub_rank, ierr)
        call MPI_Comm_rank(self%NODECOMM, node_rank, ierr)
        call MPI_Comm_size(self%NODECOMM, node_size, ierr)

        self%topology%node_rank = node_rank
        self%topology%node_size = node_size

        is_leader = merge(1, 0, node_rank == 0)
        call MPI_Exscan(is_leader, active_node_id, 1, MPI_INTEGER, &
                        MPI_SUM, self%SUBCOMM, ierr)
        if (sub_rank == 0) active_node_id = 0
        call MPI_Bcast(active_node_id, 1, MPI_INTEGER, 0, self%NODECOMM, ierr)
        self%topology%node_id = active_node_id

        call MPI_Allreduce(is_leader, active_n_nodes, 1, MPI_INTEGER, &
                           MPI_SUM, self%SUBCOMM, ierr)
        self%topology%n_nodes = active_n_nodes

#ifdef WAVEFRONT_BACKEND
        if (self%backend_flag == 1) then
            local_active_gpu_rank = merge(1, 0, self%DEVCOMM_NODE /= MPI_COMM_NULL)
            call MPI_Allreduce(local_active_gpu_rank, self%topology%devcomm_node_size, 1, &
                               MPI_INTEGER, MPI_SUM, self%NODECOMM, ierr)
        else
            self%topology%devcomm_node_size = 0
        end if
#else
        self%topology%devcomm_node_size = 0
#endif

        allocate (rank_numa_nodes(node_size))
        call MPI_Allgather(self%topology%cpu_numa_node, 1, MPI_INTEGER, &
                           rank_numa_nodes, 1, MPI_INTEGER, self%NODECOMM, ierr)
        self%topology%rank_within_cpu_numa = 0
        if (self%topology%cpu_numa_node >= 0) then
            do i = 1, node_rank
                if (rank_numa_nodes(i) == self%topology%cpu_numa_node) then
                    self%topology%rank_within_cpu_numa = self%topology%rank_within_cpu_numa + 1
                end if
            end do
        end if
        deallocate (rank_numa_nodes)

        self%topology%rank_within_gpu = 0
#ifdef WAVEFRONT_BACKEND
        if (self%backend_flag == 1 .and. self%topology%my_gpu_index >= 0) then
            allocate (rank_gpu_indices(node_size))
            call MPI_Allgather(self%topology%my_gpu_index, 1, MPI_INTEGER, &
                               rank_gpu_indices, 1, MPI_INTEGER, self%NODECOMM, ierr)
            do i = 1, node_rank
                if (rank_gpu_indices(i) == self%topology%my_gpu_index) then
                    self%topology%rank_within_gpu = self%topology%rank_within_gpu + 1
                end if
            end do
            deallocate (rank_gpu_indices)
        end if
#endif
    end subroutine refresh_layout_topology

    ! ====================================================================
    ! split_info_t methods
    ! ====================================================================

    subroutine split_info_destroy(self)
        class(split_info_t), intent(inout) :: self
        integer(int32) :: ierr

        if (self%SUBCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%SUBCOMM, ierr)
            self%SUBCOMM = MPI_COMM_NULL
        end if

        if (self%JACCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%JACCOMM, ierr)
            self%JACCOMM = MPI_COMM_NULL
        end if

        if (self%ROOTCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%ROOTCOMM, ierr)
            self%ROOTCOMM = MPI_COMM_NULL
        end if

        ! MPI_COMM is NOT freed -- caller owns it
        self%MPI_COMM = MPI_COMM_NULL
        self%worker_id = 0
        self%n_workers = 1
    end subroutine split_info_destroy

    ! ====================================================================
    ! quop_mpi_layout_t -- Lock / Unlock
    ! ====================================================================

    subroutine layout_lock(self, error_code)
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int32), intent(out) :: error_code
        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: quop_mpi_layout_t already locked")
            return
        end if
        self%locked = .true.
    end subroutine layout_lock

    subroutine layout_unlock(self, error_code)
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int32), intent(out) :: error_code
        error_code = 0
        if (.not. self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: quop_mpi_layout_t already unlocked")
            return
        end if
        self%locked = .false.
    end subroutine layout_unlock

    logical function layout_is_locked(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_is_locked = self%locked
    end function layout_is_locked

    ! ====================================================================
    ! quop_mpi_layout_t -- Destroy
    ! ====================================================================

    subroutine layout_destroy(self)
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int32) :: ierr

        ! Free owned communicators (but NOT MPI_COMM -- caller owns it)
        if (self%SUBCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%SUBCOMM, ierr)
            self%SUBCOMM = MPI_COMM_NULL
        end if

        if (self%NODECOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%NODECOMM, ierr)
            self%NODECOMM = MPI_COMM_NULL
        end if

        if (self%DEVCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%DEVCOMM, ierr)
            self%DEVCOMM = MPI_COMM_NULL
        end if

        if (self%DEVCOMM_NODE /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%DEVCOMM_NODE, ierr)
            self%DEVCOMM_NODE = MPI_COMM_NULL
        end if

        ! Deallocate partition table
        if (allocated(self%partition_table)) then
            deallocate (self%partition_table)
        end if

        ! Reset scalars
        self%locked = .false.
        self%backend_flag = 0
        self%system_size = 0
        self%n_processes = 0
        self%local_i = 0
        self%local_i_offset = 0
        self%alloc_local = 0
        self%device_n_processes = 0
        self%device_local_i = 0
        self%device_local_i_offset = 0
        self%device_alloc_local = 0
        self%MPI_COMM = MPI_COMM_NULL
        call refresh_layout_topology(self)
    end subroutine layout_destroy

    ! ====================================================================
    ! quop_mpi_layout_t -- Population
    ! ====================================================================

    subroutine layout_set_partitioning(self, local_i, local_i_offset, &
                                       device_local_i, device_local_i_offset, &
                                       error_code)
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int64), intent(in) :: local_i, local_i_offset
        integer(int64), intent(in), optional :: device_local_i, device_local_i_offset
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set partitioning on locked quop_mpi_layout_t")
            return
        end if

        self%local_i = local_i
        self%local_i_offset = local_i_offset

        if (present(device_local_i)) then
            self%device_local_i = device_local_i
        end if
        if (present(device_local_i_offset)) then
            self%device_local_i_offset = device_local_i_offset
        end if
    end subroutine layout_set_partitioning

    subroutine layout_set_MPI_COMM(self, comm, error_code)
        !! Store the root communicator.
        !! Rejects writes on a locked layout.
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int32), intent(in) :: comm
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set MPI_COMM on locked quop_mpi_layout_t")
            return
        end if

        self%MPI_COMM = comm
    end subroutine layout_set_MPI_COMM

    subroutine layout_set_SUBCOMM(self, comm, error_code)
        !! Store the sub-communicator.
        !! Rejects writes on a locked layout.
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int32), intent(in) :: comm
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set SUBCOMM on locked quop_mpi_layout_t")
            return
        end if

        self%SUBCOMM = comm
    end subroutine layout_set_SUBCOMM

    subroutine layout_set_NODECOMM(self, comm, error_code)
        !! Store the node-local communicator.
        !! Rejects writes on a locked layout.
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int32), intent(in) :: comm
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set NODECOMM on locked quop_mpi_layout_t")
            return
        end if

        self%NODECOMM = comm
    end subroutine layout_set_NODECOMM

    subroutine layout_set_DEVCOMM(self, comm, error_code)
        !! Store the device communicator.
        !! Rejects writes on a locked layout.
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int32), intent(in) :: comm
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set DEVCOMM on locked quop_mpi_layout_t")
            return
        end if

        self%DEVCOMM = comm
    end subroutine layout_set_DEVCOMM

    subroutine layout_set_DEVCOMM_NODE(self, comm, error_code)
        !! Store the device-node communicator.
        !! Rejects writes on a locked layout.
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int32), intent(in) :: comm
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set DEVCOMM_NODE on locked quop_mpi_layout_t")
            return
        end if

        self%DEVCOMM_NODE = comm
    end subroutine layout_set_DEVCOMM_NODE

    subroutine layout_set_topology(self, topo, error_code)
        !! Store the GPU topology.
        !! Rejects writes on a locked layout.
        class(quop_mpi_layout_t), intent(inout) :: self
        type(gpu_topology_t), intent(in) :: topo
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set topology on locked quop_mpi_layout_t")
            return
        end if

        self%topology = topo
    end subroutine layout_set_topology

    subroutine layout_set_system_size(self, system_size, error_code)
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int64), intent(in) :: system_size
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set system_size on locked quop_mpi_layout_t")
            return
        end if

        self%system_size = system_size
    end subroutine layout_set_system_size

    subroutine layout_set_n_processes(self, n_processes, error_code)
        !! Set n_processes (active host ranks in SUBCOMM).
        !! Rejects writes on a locked layout.
        !! Values in [0, SUBCOMM size] are accepted; this allows propagators
        !! to request a smaller communicator during negotiation (the negotiate
        !! loop will shrink SUBCOMM accordingly).
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int64), intent(in) :: n_processes
        integer(int32), intent(out) :: error_code
        integer(int32) :: ierr, comm_size

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set n_processes on locked quop_mpi_layout_t")
            return
        end if

        if (self%SUBCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_size(self%SUBCOMM, comm_size, ierr)
            if (n_processes < 0 .or. n_processes > int(comm_size, int64)) then
                error_code = 2
                return
            end if
        else if (n_processes /= 0) then
            error_code = 2
            return
        end if

        self%n_processes = n_processes
    end subroutine layout_set_n_processes

    subroutine layout_set_alloc_local(self, alloc_local, error_code)
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int64), intent(in) :: alloc_local
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set alloc_local on locked quop_mpi_layout_t")
            return
        end if

        self%alloc_local = alloc_local
    end subroutine layout_set_alloc_local

    subroutine layout_set_device_alloc_local(self, device_alloc_local, error_code)
        !! Set device_alloc_local (wavefront device allocation length).
        !! Rejects writes on a locked layout.
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int64), intent(in) :: device_alloc_local
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set device_alloc_local on locked quop_mpi_layout_t")
            return
        end if

        self%device_alloc_local = device_alloc_local
    end subroutine layout_set_device_alloc_local

    subroutine layout_set_device_n_processes(self, device_n_processes, error_code)
        !! Set device_n_processes (active GPU ranks with non-zero device workload).
        !! Rejects writes on a locked layout.
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int64), intent(in) :: device_n_processes
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set device_n_processes on locked quop_mpi_layout_t")
            return
        end if

        self%device_n_processes = device_n_processes
    end subroutine layout_set_device_n_processes

    subroutine layout_build_partition_table(self, error_code)
        !! Build the partition table from an Allgather of local_i on SUBCOMM.
        !! COLLECTIVE over SUBCOMM.
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int32), intent(out) :: error_code
        integer(int32) :: comm_size, ierr, i, local_error
        integer(int64), allocatable :: all_local_i(:)

        error_code = 0
        local_error = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot build partition_table on locked quop_mpi_layout_t")
            local_error = 1
        end if

        if (local_error == 0 .and. self%SUBCOMM == MPI_COMM_NULL) then
            call layout_note_error(error_code, 2, &
                                   "ERROR: cannot build partition_table without a valid SUBCOMM")
            local_error = 2
        end if

        call layout_sync_precondition_error(self%SUBCOMM, local_error, error_code)
        if (local_error /= 0) return

        call MPI_Comm_size(self%SUBCOMM, comm_size, ierr)
        allocate (all_local_i(comm_size))
        call MPI_Allgather(self%local_i, 1, MPI_INTEGER8, &
                           all_local_i, 1, MPI_INTEGER8, self%SUBCOMM, ierr)

        if (allocated(self%partition_table)) deallocate (self%partition_table)
        allocate (self%partition_table(comm_size + 1))
        self%partition_table(1) = 1
        do i = 1, comm_size
            self%partition_table(i + 1) = self%partition_table(i) + all_local_i(i)
        end do

        deallocate (all_local_i)
    end subroutine layout_build_partition_table

    ! ====================================================================
    ! quop_mpi_layout_t -- Validation (collective)
    ! ====================================================================

    subroutine layout_validate(self, system_size, error_code)
        !! Top-level validator. Calls all sub-validators in order.
        !! COLLECTIVE over SUBCOMM (and NODECOMM/DEVCOMM where applicable).
        !!
        !! Returns a bitmask in error_code (0 = all OK).  After calling
        !! every sub-validator, the bitmask is MPI_Allreduced (BOR) over
        !! SUBCOMM so that ALL ranks return the same value.
        class(quop_mpi_layout_t), intent(in) :: self
        integer(int64), intent(in) :: system_size
        integer(int32), intent(out) :: error_code

        integer(int32) :: sub_code, ierr

        error_code = 0

        ! 1. Non-negative fields
        call self%validate_non_negative(sub_code)
        if (sub_code /= 0) error_code = ior(error_code, LAYOUT_ERR_NON_NEGATIVE)

        ! 2. Host completeness: sum(local_i) == system_size
        call self%validate_completeness(system_size, sub_code)
        if (sub_code /= 0) error_code = ior(error_code, LAYOUT_ERR_COMPLETENESS)

        ! 3. Host rank ordering: offsets match cumulative local_i, monotone
        call self%validate_rank_ordering(sub_code)
        if (sub_code /= 0) error_code = ior(error_code, LAYOUT_ERR_RANK_ORDERING)

        ! 4. Host contiguity: partition_table internal consistency
        call self%validate_contiguity(sub_code)
        if (sub_code /= 0) error_code = ior(error_code, LAYOUT_ERR_CONTIGUITY)

        ! 5. Node contiguity: each node's ranks own a contiguous global block
        call self%validate_node_contiguity(sub_code)
        if (sub_code /= 0) error_code = ior(error_code, LAYOUT_ERR_NODE_CONTIGUITY)

        ! 6. Device ordering (DEVCOMM)
        call self%validate_device_ordering(sub_code)
        if (sub_code /= 0) error_code = ior(error_code, LAYOUT_ERR_DEVICE_ORDERING)

        ! 7. Device completeness
        call self%validate_device_completeness(system_size, sub_code)
        if (sub_code /= 0) error_code = ior(error_code, LAYOUT_ERR_DEVICE_COMPLETENESS)

        ! Sync so ALL ranks return the same bitmask.
        ! NOTE: avoid MPI_IN_PLACE -- Cray MPICH 8.1.x can zero the
        ! buffer on size-1 communicators with MPI_BOR + MPI_INTEGER4.
        sub_code = error_code
        call MPI_Allreduce(sub_code, error_code, 1, MPI_INTEGER4, &
                           MPI_BOR, self%SUBCOMM, ierr)
    end subroutine layout_validate

    subroutine layout_validate_non_negative(self, error_code)
        !! Verify scalar layout sizes and offsets are non-negative.
        class(quop_mpi_layout_t), intent(in) :: self
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%system_size < 0) error_code = 1
        if (self%n_processes < 0) error_code = 1
        if (self%local_i < 0) error_code = 1
        if (self%local_i_offset < 0) error_code = 1
        if (self%alloc_local < 0) error_code = 1
        if (self%device_n_processes < 0) error_code = 1
        if (self%device_local_i < 0) error_code = 1
        if (self%device_local_i_offset < 0) error_code = 1
        if (self%device_alloc_local < 0) error_code = 1
    end subroutine layout_validate_non_negative

    subroutine layout_validate_completeness(self, system_size, error_code)
        !! Verify sum(local_i) == system_size across SUBCOMM.
        !! COLLECTIVE over SUBCOMM.
        class(quop_mpi_layout_t), intent(in) :: self
        integer(int64), intent(in) :: system_size
        integer(int32), intent(out) :: error_code
        integer(int64) :: total
        integer(int32) :: ierr, comm_size

        error_code = 0
        call MPI_Comm_size(self%SUBCOMM, comm_size, ierr)
        call MPI_Allreduce(self%local_i, total, 1, MPI_INTEGER8, &
                           MPI_SUM, self%SUBCOMM, ierr)

        if (self%n_processes /= int(comm_size, int64)) error_code = 1
        if (total /= system_size) error_code = 1
        if (self%alloc_local < self%local_i) error_code = 1
    end subroutine layout_validate_completeness

    subroutine layout_validate_rank_ordering(self, error_code)
        !! Verify that host offsets are rank-monotone on SUBCOMM.
        !! COLLECTIVE over SUBCOMM.
        class(quop_mpi_layout_t), intent(in) :: self
        integer(int32), intent(out) :: error_code
        integer(int32) :: comm_size, rank, ierr, i
        integer(int64), allocatable :: all_local_i(:), all_offsets(:)
        integer(int64) :: expected_offset

        error_code = 0
        call MPI_Comm_size(self%SUBCOMM, comm_size, ierr)
        call MPI_Comm_rank(self%SUBCOMM, rank, ierr)
        allocate (all_local_i(comm_size), all_offsets(comm_size))

        call MPI_Allgather(self%local_i, 1, MPI_INTEGER8, &
                           all_local_i, 1, MPI_INTEGER8, self%SUBCOMM, ierr)
        call MPI_Allgather(self%local_i_offset, 1, MPI_INTEGER8, &
                           all_offsets, 1, MPI_INTEGER8, self%SUBCOMM, ierr)

        ! Check: offset[r] == sum(local_i[0..r-1])
        expected_offset = 0
        do i = 1, comm_size
            if (all_offsets(i) /= expected_offset) error_code = 1
            expected_offset = expected_offset + all_local_i(i)
        end do

        ! Strict monotonicity for consecutive non-zero ranks
        do i = 2, comm_size
            if (all_local_i(i) > 0 .and. all_local_i(i - 1) > 0) then
                if (all_offsets(i) <= all_offsets(i - 1)) error_code = 1
            end if
        end do

        deallocate (all_local_i, all_offsets)
    end subroutine layout_validate_rank_ordering

    subroutine layout_validate_contiguity(self, error_code)
        !! Verify partition_table internal consistency with per-rank local_i.
        !! COLLECTIVE over SUBCOMM.
        class(quop_mpi_layout_t), intent(in) :: self
        integer(int32), intent(out) :: error_code
        integer(int32) :: comm_size, ierr, i
        integer(int64), allocatable :: all_local_i(:)

        error_code = 0

        ! Skip if partition_table has not been built yet -- nothing to check.
        if (.not. allocated(self%partition_table)) return

        call MPI_Comm_size(self%SUBCOMM, comm_size, ierr)

        if (size(self%partition_table) /= comm_size + 1) then
            error_code = 1
            return
        end if

        if (self%partition_table(1) /= 1_int64) error_code = 1

        if (self%partition_table(comm_size + 1) /= self%system_size + 1_int64) error_code = 1

        allocate (all_local_i(comm_size))

        call MPI_Allgather(self%local_i, 1, MPI_INTEGER8, &
                           all_local_i, 1, MPI_INTEGER8, self%SUBCOMM, ierr)

        ! Check: partition_table(i+1) - partition_table(i) == local_i(i)
        do i = 1, comm_size
            if (self%partition_table(i + 1) - self%partition_table(i) /= all_local_i(i)) then
                error_code = 1
            end if
        end do

        deallocate (all_local_i)
    end subroutine layout_validate_contiguity

    subroutine layout_validate_node_contiguity(self, error_code)
        !! Verify that SUBCOMM partitions on each node form one contiguous block.
        !! COLLECTIVE over NODECOMM.  No-op if NODECOMM == MPI_COMM_NULL.
        class(quop_mpi_layout_t), intent(in) :: self
        integer(int32), intent(out) :: error_code
        integer(int32) :: node_size, node_rank, ierr
        integer(int64) :: node_lo, node_hi, my_lo, my_hi
        integer(int64) :: sum_local_i, expected_span

        error_code = 0
        if (self%NODECOMM == MPI_COMM_NULL) return

        call MPI_Comm_size(self%NODECOMM, node_size, ierr)
        call MPI_Comm_rank(self%NODECOMM, node_rank, ierr)

        my_lo = self%local_i_offset
        my_hi = self%local_i_offset + self%local_i

        call MPI_Allreduce(my_lo, node_lo, 1, MPI_INTEGER8, MPI_MIN, &
                           self%NODECOMM, ierr)
        call MPI_Allreduce(my_hi, node_hi, 1, MPI_INTEGER8, MPI_MAX, &
                           self%NODECOMM, ierr)
        call MPI_Allreduce(self%local_i, sum_local_i, 1, MPI_INTEGER8, MPI_SUM, &
                           self%NODECOMM, ierr)

        expected_span = node_hi - node_lo

        if (expected_span /= sum_local_i) error_code = 1
    end subroutine layout_validate_node_contiguity

    subroutine layout_validate_device_ordering(self, error_code)
        !! Verify that device offsets are rank-monotone on DEVCOMM.
        !! COLLECTIVE over DEVCOMM.  No-op if DEVCOMM == MPI_COMM_NULL.
        class(quop_mpi_layout_t), intent(in) :: self
        integer(int32), intent(out) :: error_code
        integer(int32) :: comm_size, rank, ierr, i
        integer(int64), allocatable :: all_dev_local_i(:), all_dev_offsets(:)
        integer(int64) :: expected_offset

        error_code = 0
        if (self%DEVCOMM == MPI_COMM_NULL) return

        call MPI_Comm_size(self%DEVCOMM, comm_size, ierr)
        call MPI_Comm_rank(self%DEVCOMM, rank, ierr)
        allocate (all_dev_local_i(comm_size), all_dev_offsets(comm_size))

        call MPI_Allgather(self%device_local_i, 1, MPI_INTEGER8, &
                           all_dev_local_i, 1, MPI_INTEGER8, self%DEVCOMM, ierr)
        call MPI_Allgather(self%device_local_i_offset, 1, MPI_INTEGER8, &
                           all_dev_offsets, 1, MPI_INTEGER8, self%DEVCOMM, ierr)

        expected_offset = 0
        do i = 1, comm_size
            if (all_dev_offsets(i) /= expected_offset) error_code = 1
            expected_offset = expected_offset + all_dev_local_i(i)
        end do

        do i = 2, comm_size
            if (all_dev_local_i(i) > 0 .and. all_dev_local_i(i - 1) > 0) then
                if (all_dev_offsets(i) <= all_dev_offsets(i - 1)) error_code = 1
            end if
        end do

        deallocate (all_dev_local_i, all_dev_offsets)
    end subroutine layout_validate_device_ordering

    subroutine layout_validate_device_completeness(self, system_size, error_code)
        !! Verify sum(device_local_i) == system_size across DEVCOMM.
        !! COLLECTIVE over DEVCOMM.  No-op if DEVCOMM == MPI_COMM_NULL.
        class(quop_mpi_layout_t), intent(in) :: self
        integer(int64), intent(in) :: system_size
        integer(int32), intent(out) :: error_code
        integer(int64) :: total
        integer(int32) :: ierr, active_device_rank, active_device_ranks

        error_code = 0

        if (self%device_alloc_local < self%device_local_i) error_code = 1

        if (self%NODECOMM /= MPI_COMM_NULL) then
            active_device_rank = 0
            if (self%device_local_i > 0) active_device_rank = 1

            call MPI_Allreduce(active_device_rank, active_device_ranks, 1, &
                               MPI_INTEGER4, MPI_SUM, self%NODECOMM, ierr)

            if (self%device_n_processes /= int(active_device_ranks, int64)) error_code = 1
        end if

        if (self%DEVCOMM == MPI_COMM_NULL) return

        call MPI_Allreduce(self%device_local_i, total, 1, MPI_INTEGER8, &
                           MPI_SUM, self%DEVCOMM, ierr)

        if (total /= system_size) error_code = 1
    end subroutine layout_validate_device_completeness

    ! ====================================================================
    ! quop_mpi_layout_t -- Communicator Management
    ! ====================================================================

    subroutine layout_shrink(self, new_size, error_code)
        !! Shrink SUBCOMM to new_size ranks and rebuild a consumable layout.
        !! Ranks 0..new_size-1 stay in, others get SUBCOMM = MPI_COMM_NULL.
        !! Active ranks are repartitioned over the new communicator and the
        !! partition table is rebuilt before return.
        !! For wavefront backend: NODECOMM, DEVCOMM, and DEVCOMM_NODE are
        !! rebuilt from the new SUBCOMM before repartitioning.
        !! COLLECTIVE over the current SUBCOMM.
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int64), intent(in) :: new_size
        integer(int32), intent(out) :: error_code
        integer(int32) :: rank, color, ierr, comm_size, pt_err, bd_err, local_error
        integer(int32) :: old_subcomm, new_subcomm

        error_code = 0
        local_error = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot shrink locked quop_mpi_layout_t")
            local_error = 1
        end if

        if (local_error == 0 .and. self%SUBCOMM == MPI_COMM_NULL) then
            call layout_note_error(error_code, 2, &
                                   "ERROR: cannot shrink without a valid SUBCOMM")
            local_error = 2
        end if

        call layout_sync_precondition_error(self%SUBCOMM, local_error, error_code)
        if (local_error /= 0) return

        call MPI_Comm_size(self%SUBCOMM, comm_size, ierr)
        if (new_size <= 0_int64 .or. new_size > int(comm_size, int64)) then
            local_error = 2
            call layout_sync_precondition_error(self%SUBCOMM, local_error, error_code)
            return
        end if

#ifdef WAVEFRONT_BACKEND
        ! Free device communicators BEFORE the SUBCOMM shrink
        ! (they were derived from the old SUBCOMM).
        if (self%DEVCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%DEVCOMM, ierr)
            self%DEVCOMM = MPI_COMM_NULL
        end if
        if (self%DEVCOMM_NODE /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%DEVCOMM_NODE, ierr)
            self%DEVCOMM_NODE = MPI_COMM_NULL
        end if
#endif
        if (self%NODECOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%NODECOMM, ierr)
            self%NODECOMM = MPI_COMM_NULL
        end if

        call MPI_Comm_rank(self%SUBCOMM, rank, ierr)

        if (rank < int(new_size, int32)) then
            color = 0
        else
            color = MPI_UNDEFINED
        end if

        old_subcomm = self%SUBCOMM
        call MPI_Comm_split(old_subcomm, color, rank, new_subcomm, ierr)
        call MPI_Comm_free(old_subcomm, ierr)

        self%SUBCOMM = new_subcomm

        if (self%SUBCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_size(self%SUBCOMM, comm_size, ierr)
            call MPI_Comm_rank(self%SUBCOMM, rank, ierr)
            self%n_processes = int(comm_size, int64)
            call create_layout_nodecomm(self%SUBCOMM, self%NODECOMM)
            call refresh_layout_topology(self)
#ifdef WAVEFRONT_BACKEND
            if (self%backend_flag == 1) then
                ! Zero host fields; device_block_distribute derives them
                ! bottom-up from device partitioning.
                self%local_i = 0
                self%local_i_offset = 0
                self%alloc_local = 0
                ! Rebuild wavefront communicator hierarchy from the new SUBCOMM.
                call create_devcomm_with_topology(self%SUBCOMM, self%NODECOMM, &
                                                  self%topology, &
                                                  self%DEVCOMM, self%DEVCOMM_NODE)
                call refresh_layout_topology(self)
                call device_block_distribute(self, bd_err)
                if (bd_err /= 0) then
                    error_code = bd_err
                    return
                end if
            else
                call block_distribute(self, comm_size, rank, bd_err)
                if (bd_err /= 0) then
                    error_code = bd_err
                    return
                end if
                self%device_n_processes = 0
                self%device_local_i = 0
                self%device_local_i_offset = 0
                self%device_alloc_local = 0
            end if
#else
            call block_distribute(self, comm_size, rank, bd_err)
            if (bd_err /= 0) then
                error_code = bd_err
                return
            end if
            self%device_n_processes = 0
            self%device_local_i = 0
            self%device_local_i_offset = 0
            self%device_alloc_local = 0
#endif
            call self%build_partition_table(pt_err)
            if (pt_err /= 0) then
                call layout_note_error(error_code, 3, &
                                       "ERROR: shrink failed to rebuild partition_table after communicator resize")
                return
            end if
        else
            self%n_processes = 0
            self%local_i = 0
            self%local_i_offset = 0
            self%alloc_local = 0
            self%device_local_i = 0
            self%device_local_i_offset = 0
            self%device_n_processes = 0
            self%device_alloc_local = 0
            if (allocated(self%partition_table)) then
                deallocate (self%partition_table)
            end if
            call refresh_layout_topology(self)
        end if
    end subroutine layout_shrink

    subroutine layout_filter_active_ranks(self, error_code)
        !! Rebuild SUBCOMM to keep only ranks with non-zero host data.
        !! Active ranks are repartitioned over the filtered communicator and the
        !! partition table is rebuilt before return.
        !! For wavefront backend: NODECOMM, DEVCOMM, and DEVCOMM_NODE are
        !! rebuilt from the filtered SUBCOMM before repartitioning.
        !! COLLECTIVE over the current SUBCOMM.
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int32), intent(out) :: error_code
        integer(int32) :: rank, color, ierr, comm_size, pt_err, bd_err, local_error
        integer(int32) :: active_host_rank, active_host_ranks
        integer(int32) :: old_subcomm, new_subcomm

        error_code = 0
        local_error = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot filter ranks on locked quop_mpi_layout_t")
            local_error = 1
        end if

        if (local_error == 0 .and. self%SUBCOMM == MPI_COMM_NULL) then
            call layout_note_error(error_code, 2, &
                                   "ERROR: cannot filter ranks without a valid SUBCOMM")
            local_error = 2
        end if

        call layout_sync_precondition_error(self%SUBCOMM, local_error, error_code)
        if (local_error /= 0) return

        active_host_rank = merge(1_int32, 0_int32, self%local_i > 0_int64)
        call MPI_Allreduce(active_host_rank, active_host_ranks, 1, &
                           MPI_INTEGER, MPI_SUM, self%SUBCOMM, ierr)
        if (active_host_ranks <= 0) then
            call layout_note_error(error_code, 3, &
                                   "ERROR: cannot filter ranks when all local_i are zero")
            return
        end if

#ifdef WAVEFRONT_BACKEND
        ! Free device communicators BEFORE the SUBCOMM rebuild
        ! (they were derived from the old SUBCOMM).
        if (self%DEVCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%DEVCOMM, ierr)
            self%DEVCOMM = MPI_COMM_NULL
        end if
        if (self%DEVCOMM_NODE /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%DEVCOMM_NODE, ierr)
            self%DEVCOMM_NODE = MPI_COMM_NULL
        end if
#endif
        if (self%NODECOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%NODECOMM, ierr)
            self%NODECOMM = MPI_COMM_NULL
        end if

        call MPI_Comm_rank(self%SUBCOMM, rank, ierr)
        if (self%local_i > 0_int64) then
            color = 0
        else
            color = MPI_UNDEFINED
        end if

        old_subcomm = self%SUBCOMM
        call MPI_Comm_split(old_subcomm, color, rank, new_subcomm, ierr)
        call MPI_Comm_free(old_subcomm, ierr)

        self%SUBCOMM = new_subcomm

        if (self%SUBCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_size(self%SUBCOMM, comm_size, ierr)
            call MPI_Comm_rank(self%SUBCOMM, rank, ierr)
            self%n_processes = int(comm_size, int64)
            call create_layout_nodecomm(self%SUBCOMM, self%NODECOMM)
            call refresh_layout_topology(self)
#ifdef WAVEFRONT_BACKEND
            if (self%backend_flag == 1) then
                ! Zero host fields; device_block_distribute derives them
                ! bottom-up from device partitioning.
                self%local_i = 0
                self%local_i_offset = 0
                self%alloc_local = 0
                ! Rebuild wavefront communicator hierarchy from the filtered SUBCOMM.
                call create_devcomm_with_topology(self%SUBCOMM, self%NODECOMM, &
                                                  self%topology, &
                                                  self%DEVCOMM, self%DEVCOMM_NODE)
                call refresh_layout_topology(self)
                call device_block_distribute(self, bd_err)
                if (bd_err /= 0) then
                    error_code = bd_err
                    return
                end if
            else
                call block_distribute(self, comm_size, rank, bd_err)
                if (bd_err /= 0) then
                    error_code = bd_err
                    return
                end if
                self%device_n_processes = 0
                self%device_local_i = 0
                self%device_local_i_offset = 0
                self%device_alloc_local = 0
            end if
#else
            call block_distribute(self, comm_size, rank, bd_err)
            if (bd_err /= 0) then
                error_code = bd_err
                return
            end if
            self%device_n_processes = 0
            self%device_local_i = 0
            self%device_local_i_offset = 0
            self%device_alloc_local = 0
#endif
            call self%build_partition_table(pt_err)
            if (pt_err /= 0) then
                call layout_note_error(error_code, 4, &
                                       "ERROR: filter_active_ranks failed to rebuild partition_table")
                return
            end if
        else
            self%n_processes = 0
            self%local_i = 0
            self%local_i_offset = 0
            self%alloc_local = 0
            self%device_local_i = 0
            self%device_local_i_offset = 0
            self%device_n_processes = 0
            self%device_alloc_local = 0
            if (allocated(self%partition_table)) then
                deallocate (self%partition_table)
            end if
            call refresh_layout_topology(self)
        end if
    end subroutine layout_filter_active_ranks

    subroutine layout_rebuild_communicators(self, error_code)
        !! Rebuild DEVCOMM/DEVCOMM_NODE for current partitioning.
        !! MPI backend: No-op (NODECOMM is unchanged and device comms are NULL).
        !! Wavefront backend: Free old DEVCOMM/DEVCOMM_NODE and rebuild
        !! based on current data distribution (has_data = device_local_i > 0).
        !! NODECOMM is unchanged (it depends only on SUBCOMM, not data).
        !! COLLECTIVE over SUBCOMM (and NODECOMM when applicable).
        class(quop_mpi_layout_t), intent(inout) :: self
        integer(int32), intent(out) :: error_code
        integer(int32) :: ierr, local_error, dn_size

        error_code = 0
        local_error = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot rebuild communicators on locked quop_mpi_layout_t")
            local_error = 1
        end if

        call layout_sync_precondition_error(self%SUBCOMM, local_error, error_code)
        if (local_error /= 0) return

#ifdef WAVEFRONT_BACKEND
        if (self%backend_flag /= 1) then
            self%device_n_processes = 0
            self%device_local_i = 0
            self%device_local_i_offset = 0
            self%device_alloc_local = 0
            return
        end if

        ! Skip if no NODECOMM (rank excluded)
        if (self%NODECOMM == MPI_COMM_NULL) return

        ! Free old DEVCOMM and DEVCOMM_NODE
        if (self%DEVCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%DEVCOMM, ierr)
            self%DEVCOMM = MPI_COMM_NULL
        end if
        if (self%DEVCOMM_NODE /= MPI_COMM_NULL) then
            call MPI_Comm_free(self%DEVCOMM_NODE, ierr)
            self%DEVCOMM_NODE = MPI_COMM_NULL
        end if

        ! Reset transient device allocation metadata before the next
        ! negotiation pass repopulates it from propagator callbacks.
        self%device_alloc_local = 0

        ! Rebuild based on current partitioning:
        ! A rank joins DEVCOMM only if it has a GPU AND has non-zero local data.
        call create_devcomm_with_data(self%SUBCOMM, self%NODECOMM, &
                                      self%topology, &
                                      self%device_local_i > 0, &
                                      self%DEVCOMM, self%DEVCOMM_NODE)
        call refresh_layout_topology(self)
        dn_size = self%topology%devcomm_node_size
        self%device_n_processes = int(dn_size, int64)
#else
        ! MPI backend: nothing to rebuild.
        continue
#endif
    end subroutine layout_rebuild_communicators

    ! ====================================================================
    ! Top-level entry points
    ! ====================================================================
    ! These subroutines are the Fortran entry points called by Python
    ! across the f2py boundary.  They orchestrate the phases shown in
    ! quop_mpi_layout_flowchart.md.
    ! Stage 1: stubs that perform the MPI-backend-safe subset of work.
    ! Full negotiate loop is deferred to Stage 3+.

    subroutine destroy_topology(topo_ptr)
        !! Free the gpu_topology_t allocated by discover_topology.
        type(c_ptr), intent(inout) :: topo_ptr
        type(gpu_topology_t), pointer :: topo

        if (.not. c_associated(topo_ptr)) return
        call c_f_pointer(topo_ptr, topo)
        deallocate (topo)
        topo_ptr = c_null_ptr
    end subroutine destroy_topology

    subroutine init_explicit_wavefront_layout(ci_ptr, error_code)
        !! Populate NODECOMM and topology on an explicit layout so the
        !! wavefront backend can initialise native contexts without going
        !! through the full negotiate path.
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: error_code

        type(quop_mpi_layout_t), pointer :: ci
        type(gpu_topology_t), pointer :: topo
        type(c_ptr) :: topo_ptr

        error_code = 0
        topo_ptr = c_null_ptr

        if (.not. c_associated(ci_ptr)) then
            error_code = 1
            return
        end if

        call c_f_pointer(ci_ptr, ci)
        if (.not. associated(ci)) then
            error_code = 1
            return
        end if

        if (ci%locked) then
            error_code = 1
            return
        end if

        if (ci%SUBCOMM == MPI_COMM_NULL) then
            error_code = 2
            return
        end if

        ci%backend_flag = 1

        if (ci%NODECOMM == MPI_COMM_NULL) then
            call create_layout_nodecomm(ci%SUBCOMM, ci%NODECOMM)
        end if

        call discover_topology(topo_ptr, ci%SUBCOMM, 1_int32, error_code)
        if (error_code /= 0) return

        call c_f_pointer(topo_ptr, topo)
        ci%topology = topo
        call destroy_topology(topo_ptr)

        ci%DEVCOMM = MPI_COMM_NULL
        ci%DEVCOMM_NODE = MPI_COMM_NULL
        ci%device_n_processes = 0_int64
        ci%device_local_i = 0_int64
        ci%device_local_i_offset = 0_int64
        ci%device_alloc_local = 0_int64

        call refresh_layout_topology(ci)
    end subroutine init_explicit_wavefront_layout

    subroutine get_topology_info(topo_ptr, n_physical_gpus, ranks_per_gpu, node_size)
        !! Return key topology fields for Python-side configuration.
        !! NOT collective -- purely local read.
        type(c_ptr), intent(in) :: topo_ptr
        integer(int32), intent(out) :: n_physical_gpus
        integer(int32), intent(out) :: ranks_per_gpu
        integer(int32), intent(out) :: node_size
        type(gpu_topology_t), pointer :: topo

        call c_f_pointer(topo_ptr, topo)
        n_physical_gpus = topo%n_physical_gpus
        ranks_per_gpu = topo%ranks_per_gpu
        node_size = topo%node_size
    end subroutine get_topology_info

    subroutine get_layout_topology_info(ci_ptr, n_physical_gpus, ranks_per_gpu, node_size)
        !! Return key topology fields from the current layout state.
        !! NOT collective -- purely local read.
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: n_physical_gpus
        integer(int32), intent(out) :: ranks_per_gpu
        integer(int32), intent(out) :: node_size
        type(quop_mpi_layout_t), pointer :: ci

        n_physical_gpus = 0
        ranks_per_gpu = 1
        node_size = 0

        if (.not. c_associated(ci_ptr)) return

        call c_f_pointer(ci_ptr, ci)
        if (.not. associated(ci)) return

        n_physical_gpus = ci%topology%n_physical_gpus
        ranks_per_gpu = ci%topology%ranks_per_gpu
        node_size = ci%topology%node_size
    end subroutine get_layout_topology_info

    subroutine discover_topology(topo_ptr, MPI_COMM, backend_flag, error_code)
        !! Phase 0: Detect node structure and GPU hardware.
        !! Collective on MPI_COMM (ALL ranks).
        !!
        !! Both backends: detect node structure using
        !!   MPI_Comm_split_type(MPI_COMM_TYPE_SHARED) to count nodes,
        !!   assign sequential node IDs, and record per-rank node info.
        !!
        !! MPI backend (backend_flag == 0):
        !!   GPU fields stay at zero/false defaults.
        !!
        !! Wavefront backend: full GPU topology detection (Stage 6).
        !!
        !! Returns topo_ptr -- invariant, read-only for the rest of the
        !! program.
        type(c_ptr), intent(out) :: topo_ptr
        integer(int32), intent(in) :: MPI_COMM
        integer(int32), intent(in) :: backend_flag
        integer(int32), intent(out) :: error_code

        type(gpu_topology_t), pointer :: topo
        integer(int32) :: nodecomm, ierr, alloc_status, free_ierr
        integer(int32) :: mpi_rank, node_rank, node_size
        integer(int32) :: is_leader, my_node_id, total_nodes
        integer(int32) :: proc_name_len

        topo_ptr = c_null_ptr
        error_code = 0
        nodecomm = MPI_COMM_NULL

        allocate (topo, stat=alloc_status)
        if (alloc_status /= 0) then
            error_code = 100
            return
        end if

        ! Ensure deterministic defaults even when backend_flag == 0
        topo%ranks_per_gpu = 0
        topo%binding_mode = 'none'
        topo%visible_device_count = 0
        topo%n_physical_gpus = 0
        topo%my_gpu_index = 0
        topo%assigned_device_id = 0
        topo%rank_within_gpu = 0
        topo%gpu_slot_ordinal = -1
        topo%cpu_numa_node = -1
        topo%rank_within_cpu_numa = 0
        topo%is_gpu_rank = .false.
        topo%node_rank = 0
        topo%node_size = 0
        topo%devcomm_node_size = 0
        topo%n_nodes = 1
        topo%node_id = 0
        topo%hostname = ''
        if (allocated(topo%visible_gpus)) deallocate (topo%visible_gpus)

        ! -- Detect node structure (all backends) --------------------
        ! Create a temporary node-local communicator.
        call MPI_Comm_split_type(MPI_COMM, MPI_COMM_TYPE_SHARED, 0, &
                                 MPI_INFO_NULL, nodecomm, ierr)
        if (ierr /= MPI_SUCCESS) then
            error_code = 101
            deallocate (topo)
            return
        end if

#ifdef WAVEFRONT_BACKEND
        ! -- Wavefront backend: full GPU topology detection -------
        if (backend_flag == 1) then
            call init_gpu_topology(nodecomm, topo)
        end if
#endif

        call MPI_Comm_rank(nodecomm, node_rank, ierr)
        if (ierr /= MPI_SUCCESS) then
            error_code = 102
            goto 900
        end if

        call MPI_Comm_size(nodecomm, node_size, ierr)
        if (ierr /= MPI_SUCCESS) then
            error_code = 103
            goto 900
        end if

        call MPI_Comm_rank(MPI_COMM, mpi_rank, ierr)
        if (ierr /= MPI_SUCCESS) then
            error_code = 104
            goto 900
        end if

        topo%node_rank = node_rank
        topo%node_size = node_size

        ! Assign sequential node IDs: each node leader (node_rank == 0)
        ! gets a unique ID via exclusive prefix sum of leader flags.
        is_leader = merge(1, 0, node_rank == 0)
        call MPI_Exscan(is_leader, my_node_id, 1, MPI_INTEGER, &
                        MPI_SUM, MPI_COMM, ierr)
        if (ierr /= MPI_SUCCESS) then
            error_code = 105
            goto 900
        end if
        if (mpi_rank == 0) my_node_id = 0 ! MPI_Exscan undefined for rank 0

        ! Broadcast the node ID from each leader to all ranks on its node.
        call MPI_Bcast(my_node_id, 1, MPI_INTEGER, 0, nodecomm, ierr)
        if (ierr /= MPI_SUCCESS) then
            error_code = 106
            goto 900
        end if
        topo%node_id = my_node_id

        ! Total number of nodes.
        call MPI_Allreduce(is_leader, total_nodes, 1, MPI_INTEGER, &
                           MPI_SUM, MPI_COMM, ierr)
        if (ierr /= MPI_SUCCESS) then
            error_code = 107
            goto 900
        end if
        topo%n_nodes = total_nodes

        ! Processor/host name (typically hostname) - store as invariant.
        call MPI_Get_processor_name(topo%hostname, proc_name_len, ierr)
        if (ierr /= MPI_SUCCESS) then
            error_code = 108
            goto 900
        end if
        if (proc_name_len < len(topo%hostname)) then
            topo%hostname(proc_name_len + 1:) = ' '
        end if

        ! Free the temporary node communicator.
        call MPI_Comm_free(nodecomm, ierr)
        if (ierr /= MPI_SUCCESS) then
            error_code = 109
            deallocate (topo)
            return
        end if

        topo_ptr = c_loc(topo)
        return

900     continue
        if (nodecomm /= MPI_COMM_NULL) then
            call MPI_Comm_free(nodecomm, free_ierr)
        end if
        deallocate (topo)
    end subroutine discover_topology

    subroutine split_workers(split_ptr, MPI_COMM, topo_ptr, &
                             n_jacobian_workers, backend_flag, &
                             worker_id, status)
        !! Phase 0b: Split MPI_COMM into balanced worker groups.
        !! Collective on MPI_COMM (ALL ranks).
        !!
        !! Degenerate case (n_jacobian_workers == 1):
        !!   SUBCOMM = MPI_Comm_dup(MPI_COMM), worker_id = 0.
        !!
        !! General case (n_workers <= n_nodes):
        !!   Node-aligned split: whole nodes are assigned to each worker
        !!   to minimise inter-node communication over the resulting
        !!   SUBCOMMs.  Uses floor/ceil node assignment when n_nodes is
        !!   not evenly divisible by n_workers.
        !!
        !! Fallback (n_workers > n_nodes):
        !!   Rank-based round-robin (more workers than nodes requires
        !!   splitting intra-node).
        !!
        !! Creates split_info_t containing:
        !!   SUBCOMM, JACCOMM (MPI_COMM_NULL), worker_id, n_workers.
        !!
        !! status == 0: success.
        type(c_ptr), intent(out) :: split_ptr
        integer(int32), intent(in) :: MPI_COMM
        type(c_ptr), intent(in) :: topo_ptr
        integer(int32), intent(in) :: n_jacobian_workers
        integer(int32), intent(in) :: backend_flag
        integer(int32), intent(out) :: worker_id
        integer(int32), intent(out) :: status

        type(split_info_t), pointer :: si
        type(gpu_topology_t), pointer :: topo
        integer(int32) :: nprocs, rank, ierr
        integer(int32) :: color
        integer(int32) :: n_nodes, my_node_id
        integer(int32) :: nodes_per_worker, node_remainder
        integer(int32) :: ranks_per_worker, rank_remainder
        integer(int32) :: subcomm_rank
        integer(int32) :: my_node_nw, my_node_w0
        integer(int32) :: my_node_active_gpu_ranks
        integer(int32) :: my_gpu_ordinal, remaining_workers, progress
        integer(int32) :: i, other_node_id, other_is_leader, other_active_gpu_ranks
        integer(int32) :: local_error, global_error
        integer(int32) :: gather_info(3)
        integer(int32), allocatable :: all_worker_info(:), node_active_gpu_ranks(:), workers_per_node(:)

        status = 0
        allocate (si)

        call MPI_Comm_size(MPI_COMM, nprocs, ierr)
        call MPI_Comm_rank(MPI_COMM, rank, ierr)

        si%n_workers = n_jacobian_workers
        si%MPI_COMM = MPI_COMM ! store parent comm for negotiate()

        if (n_jacobian_workers <= 0 .or. n_jacobian_workers > nprocs) then
            status = 1
            si%SUBCOMM = MPI_COMM_NULL
            si%worker_id = -1
            split_ptr = c_loc(si)
            worker_id = si%worker_id
            return
        end if

        if (n_jacobian_workers == 1) then
            ! Degenerate case: single worker group
            call MPI_Comm_dup(MPI_COMM, si%SUBCOMM, ierr)
            si%worker_id = 0
        else
            ! Read node topology from discover_topology() result
            call c_f_pointer(topo_ptr, topo)
            n_nodes = topo%n_nodes
            my_node_id = topo%node_id

            if (n_jacobian_workers <= n_nodes) then
                ! -- Node-aligned split ------------------------------
                ! Assign whole nodes to each worker to minimise
                ! inter-node communication on the resulting SUBCOMMs.
                ! First `node_remainder` workers get one extra node.
                nodes_per_worker = n_nodes / n_jacobian_workers
                node_remainder = mod(n_nodes, n_jacobian_workers)

                if (my_node_id < node_remainder * (nodes_per_worker + 1)) then
                    color = my_node_id / (nodes_per_worker + 1)
                else
                    color = node_remainder + &
                            (my_node_id - node_remainder * (nodes_per_worker + 1)) &
                            / nodes_per_worker
                end if
            else
#ifdef WAVEFRONT_BACKEND
                ! -- GPU-aware intra-node split ----------------------
                ! Distribute workers across nodes according to the number
                ! of launched GPU-capable ranks on each node, then assign
                ! those active GPU ranks evenly to workers.
                !
                ! QUOP_RANKS_PER_GPU limits how many launched ranks may
                ! share a GPU, but it must not create empty workers from
                ! theoretical slot capacity that has not actually been
                ! launched into this MPI job.
                !
                ! Heterogeneous GPU topology (nodes with different GPU
                ! counts) is rejected by Python create_workers().
                allocate (all_worker_info(3 * nprocs))
                allocate (node_active_gpu_ranks(n_nodes))
                allocate (workers_per_node(n_nodes))
                node_active_gpu_ranks = 0
                workers_per_node = 0

                gather_info(1) = my_node_id
                gather_info(2) = merge(1_int32, 0_int32, topo%node_rank == 0)
                gather_info(3) = topo%devcomm_node_size
                call MPI_Allgather(gather_info, 3, MPI_INTEGER, all_worker_info, 3, MPI_INTEGER, MPI_COMM, ierr)

                do i = 0, nprocs - 1
                    other_node_id = all_worker_info(3 * i + 1)
                    other_is_leader = all_worker_info(3 * i + 2)
                    other_active_gpu_ranks = all_worker_info(3 * i + 3)
                    if (other_node_id >= 0 .and. other_node_id < n_nodes) then
                        if (other_is_leader /= 0) then
                            node_active_gpu_ranks(other_node_id + 1) = other_active_gpu_ranks
                        end if
                    end if
                end do

                ! Give each populated node one worker, then hand out the
                ! remaining workers round-robin up to the number of active
                ! GPU ranks on that node.
                remaining_workers = n_jacobian_workers
                do i = 1, n_nodes
                    if (node_active_gpu_ranks(i) > 0 .and. remaining_workers > 0) then
                        workers_per_node(i) = 1
                        remaining_workers = remaining_workers - 1
                    end if
                end do

                do while (remaining_workers > 0)
                    progress = 0
                    do i = 1, n_nodes
                        if (workers_per_node(i) < node_active_gpu_ranks(i)) then
                            workers_per_node(i) = workers_per_node(i) + 1
                            remaining_workers = remaining_workers - 1
                            progress = 1
                            if (remaining_workers == 0) exit
                        end if
                    end do
                    if (progress == 0) exit
                end do

                my_node_active_gpu_ranks = topo%devcomm_node_size
                my_node_nw = workers_per_node(my_node_id + 1)

                if (remaining_workers > 0 .or. (my_node_nw > 0 .and. my_node_active_gpu_ranks <= 0) .or. &
                    (topo%is_gpu_rank .and. (topo%gpu_slot_ordinal < 0 .or. &
                     topo%gpu_slot_ordinal >= my_node_active_gpu_ranks))) then
                    local_error = 1
                else
                    local_error = 0
                end if
                call MPI_Allreduce(local_error, global_error, 1, MPI_INTEGER, MPI_MAX, MPI_COMM, ierr)
                if (global_error /= 0) then
                    deallocate (all_worker_info, node_active_gpu_ranks, workers_per_node)
                    status = 1
                    si%SUBCOMM = MPI_COMM_NULL
                    si%worker_id = -1
                    split_ptr = c_loc(si)
                    worker_id = si%worker_id
                    return
                end if

                my_node_w0 = 0
                do i = 1, my_node_id
                    my_node_w0 = my_node_w0 + workers_per_node(i)
                end do

                if (topo%is_gpu_rank) then
                    ! gpu_topology owns the dense per-node GPU slot order.
                    my_gpu_ordinal = topo%gpu_slot_ordinal
                    color = my_node_w0 + my_gpu_ordinal * my_node_nw / my_node_active_gpu_ranks
                else
                    ! Non-GPU rank: round-robin among this node's workers.
                    color = my_node_w0 + mod(topo%node_rank, my_node_nw)
                end if

                deallocate (all_worker_info, node_active_gpu_ranks, workers_per_node)
#else
                ! -- Rank-based fallback (MPI backend) ---------------
                ! More workers than nodes: must split intra-node.
                ranks_per_worker = nprocs / n_jacobian_workers
                rank_remainder = mod(nprocs, n_jacobian_workers)

                if (rank < rank_remainder * (ranks_per_worker + 1)) then
                    color = rank / (ranks_per_worker + 1)
                else
                    color = rank_remainder + &
                            (rank - rank_remainder * (ranks_per_worker + 1)) &
                            / ranks_per_worker
                end if
#endif
            end if

            si%worker_id = color
            call MPI_Comm_split(MPI_COMM, color, rank, si%SUBCOMM, ierr)
        end if

        ! Create ROOTCOMM connecting rank 0 of each pre-negotiate SUBCOMM.
        ! This is collective on MPI_COMM (all ranks call MPI_Comm_split).
        ! Gives swarm immediate access to a ROOTCOMM before negotiate().
        ! Post-negotiate code will recreate this to remain valid.
        call MPI_Comm_rank(si%SUBCOMM, subcomm_rank, ierr)
        if (subcomm_rank == 0) then
            color = 0
        else
            color = MPI_UNDEFINED
        end if
        call MPI_Comm_split(MPI_COMM, color, rank, si%ROOTCOMM, ierr)

        split_ptr = c_loc(si)
        worker_id = si%worker_id
    end subroutine split_workers

    subroutine negotiate(layout_ptr, split_ptr, topo_ptr, &
                         system_size, backend_flag, &
                         n_propagators, propagator_ptrs, &
                         callback_ptrs, status)
        !! Phases 1-5: CREATE layout -> NEGOTIATE loop -> FINALISE ->
        !! VALIDATE -> LOCK.
        !! Collective on the per-worker SUBCOMM (from split_ptr).
        !!
        !! Takes ownership of SUBCOMM from split_ptr (sets split_ptr's
        !! SUBCOMM to MPI_COMM_NULL).
        !!
        !! The negotiate loop (Phase 2) iterates over propagators, calling
        !! each one's max_comm_size callback with the layout.  Propagators
        !! may lower ci%n_processes (requesting a smaller communicator) or
        !! override ci%local_i/local_i_offset (e.g. FFTW distribution).
        !! The loop filters/shrinks SUBCOMM and re-distributes until stable.
        !!
        !! status ==    0: success, rank is active, layout locked.
        !! status ==   -1: rank excluded during negotiate (SUBCOMM shrunk/filtered).
        !! status ==    1: system_size <= 0.
        !! status ==    3: failed to converge.
        !! status ==    4: shrink finalization failed during negotiate.
        !! status ==    6: communicator rebuild failed during negotiate.
        !! status ==    7: zero-host-data rank filtering failed during negotiate.
        !! status == 1000 + code: propagator max_comm_size callback failed
        !!     with recoverable status code.
        type(c_ptr), intent(out) :: layout_ptr
        type(c_ptr), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: topo_ptr
        integer(int64), intent(in) :: system_size
        integer(int32), intent(in) :: backend_flag
        integer(int32), intent(in) :: n_propagators
        integer(int64), dimension(n_propagators), intent(in) :: propagator_ptrs
        integer(int64), dimension(n_propagators), intent(in) :: callback_ptrs
        integer(int32), intent(out) :: status

        type(split_info_t), pointer :: si
        type(gpu_topology_t), pointer :: topo
        type(quop_mpi_layout_t), pointer :: ci
        integer(int32) :: comm_size, rank, ierr, i
        integer(int32) :: shrink_err, rebuild_err, filter_err
        integer(int32) :: callback_err, synced_callback_err
        integer(int32) :: bd_err, pt_err, lock_err
        integer(int64) :: prev_n_procs, prev_local_i, prev_offset

        ! For callback dispatch
        type(c_funptr) :: cb_funptr
        procedure(negotiate_callback_iface), pointer :: callback
        type(c_ptr) :: prop_cptr

        ! Stability check
        integer(int32) :: local_stable_int, global_stable_int
        integer(int32) :: local_filter_int, global_filter_int
        integer(int32) :: local_rebuild_int, global_rebuild_int
        logical :: global_stable
        logical :: restart_loop

        integer(int32), parameter :: MAX_ITERATIONS = 100
        integer(int32) :: iteration

        status = 0
        layout_ptr = c_null_ptr

        ! Recover split_info and topology
        call c_f_pointer(split_ptr, si)
        call c_f_pointer(topo_ptr, topo)

        ! Validate system_size
        if (system_size <= 0) then
            status = 1
            return
        end if

        ! Phase 1: CREATE
        allocate (ci)

        ! Store parent communicator + invariants (for MPI_Abort and dumps)
        ci%MPI_COMM = si%MPI_COMM
        ci%backend_flag = backend_flag
        ci%system_size = system_size

        ! Copy topology invariants (needed even for excluded ranks)
        ci%topology = topo

        ! Take ownership of SUBCOMM from split_info
        ci%SUBCOMM = si%SUBCOMM
        si%SUBCOMM = MPI_COMM_NULL

        ! If SUBCOMM is null, this rank was excluded
        if (ci%SUBCOMM == MPI_COMM_NULL) then
            call refresh_layout_topology(ci)
            status = -1
            layout_ptr = c_loc(ci)
            return
        end if

        call MPI_Comm_size(ci%SUBCOMM, comm_size, ierr)
        call MPI_Comm_rank(ci%SUBCOMM, rank, ierr)
        ci%n_processes = int(comm_size, int64)
        ! ci%system_size and ci%topology already set above

#ifdef WAVEFRONT_BACKEND
        ! Create a node-local communicator for all active ranks.
        call create_layout_nodecomm(ci%SUBCOMM, ci%NODECOMM)
        call refresh_layout_topology(ci)

        ! Wavefront backend: create per-worker device communicator hierarchy.
        ! DEVCOMM and DEVCOMM_NODE are derived from SUBCOMM/NODECOMM and the
        ! hw-intrinsic topology detected by discover_topology().
        if (backend_flag == 1) then
            call create_devcomm_with_topology(ci%SUBCOMM, ci%NODECOMM, &
                                              ci%topology, &
                                              ci%DEVCOMM, ci%DEVCOMM_NODE)
        end if
#else
        ! MPI backend: create a node-local communicator for all active ranks.
        call create_layout_nodecomm(ci%SUBCOMM, ci%NODECOMM)
#endif
        call refresh_layout_topology(ci)
        ci%device_n_processes = int(ci%topology%devcomm_node_size, int64)

        ! Phase 2: NEGOTIATE loop
        ! Block distribute initial partitioning
#ifdef WAVEFRONT_BACKEND
        call redistribute_current_layout(bd_err)
        if (bd_err /= 0) then
            status = 5
            layout_ptr = c_loc(ci)
            return
        end if
#else
        call block_distribute(ci, comm_size, rank, bd_err)
        if (bd_err /= 0) then
            status = 5
            layout_ptr = c_loc(ci)
            return
        end if
#endif

        iteration = 0
        negotiate_loop: do while (iteration < MAX_ITERATIONS)

            ! Save current state for stability check
            prev_n_procs = ci%n_processes
            prev_local_i = ci%local_i
            prev_offset = ci%local_i_offset

            ! Query each propagator via its callback
            do i = 1, n_propagators
                cb_funptr = transfer(callback_ptrs(i), cb_funptr)
                call c_f_procpointer(cb_funptr, callback)
                prop_cptr = transfer(propagator_ptrs(i), prop_cptr)
                call callback(prop_cptr, c_loc(ci), callback_err)
                synced_callback_err = callback_err
                call MPI_Allreduce(callback_err, synced_callback_err, 1, &
                                   MPI_INTEGER4, MPI_MAX, ci%SUBCOMM, ierr)
                if (synced_callback_err /= 0) then
                    status = 1000 + synced_callback_err
                    layout_ptr = c_loc(ci)
                    return
                end if
            end do

            call reconcile_post_callbacks(prev_n_procs, restart_loop)
            if (status /= 0) then
                layout_ptr = c_loc(ci)
                return
            end if
            if (restart_loop) then
                iteration = iteration + 1
                cycle negotiate_loop
            end if

            ! Collective stability check: did any rank see changes?
            if (ci%n_processes == prev_n_procs .and. &
                ci%local_i == prev_local_i .and. &
                ci%local_i_offset == prev_offset) then
                local_stable_int = 1
            else
                local_stable_int = 0
            end if
            call MPI_Allreduce(local_stable_int, global_stable_int, &
                               1, MPI_INTEGER, MPI_MIN, ci%SUBCOMM, ierr)
            global_stable = (global_stable_int == 1)

            if (global_stable) then
                ! Double-check: one more pass to confirm
                prev_n_procs = ci%n_processes
                prev_local_i = ci%local_i
                prev_offset = ci%local_i_offset

                do i = 1, n_propagators
                    cb_funptr = transfer(callback_ptrs(i), cb_funptr)
                    call c_f_procpointer(cb_funptr, callback)
                    prop_cptr = transfer(propagator_ptrs(i), prop_cptr)
                    call callback(prop_cptr, c_loc(ci), callback_err)
                    synced_callback_err = callback_err
                    call MPI_Allreduce(callback_err, synced_callback_err, 1, &
                                       MPI_INTEGER4, MPI_MAX, ci%SUBCOMM, ierr)
                    if (synced_callback_err /= 0) then
                        status = 1000 + synced_callback_err
                        layout_ptr = c_loc(ci)
                        return
                    end if
                end do

                call reconcile_post_callbacks(prev_n_procs, restart_loop)
                if (status /= 0) then
                    layout_ptr = c_loc(ci)
                    return
                end if
                if (restart_loop) then
                    iteration = iteration + 1
                    cycle negotiate_loop
                end if

                if (ci%n_processes == prev_n_procs .and. &
                    ci%local_i == prev_local_i .and. &
                    ci%local_i_offset == prev_offset) then
                    local_stable_int = 1
                else
                    local_stable_int = 0
                end if
                call MPI_Allreduce(local_stable_int, global_stable_int, &
                                   1, MPI_INTEGER, MPI_MIN, ci%SUBCOMM, ierr)

                if (global_stable_int == 1) exit negotiate_loop
            end if

            iteration = iteration + 1
        end do negotiate_loop

        if (iteration >= MAX_ITERATIONS) then
            status = 3
            layout_ptr = c_loc(ci)
            return
        end if

        ! Set alloc_local (default: same as local_i; propagators may
        ! have updated it during max_comm_size, e.g. FFTW padding)
        if (ci%alloc_local < ci%local_i) then
            ci%alloc_local = ci%local_i
        end if

        ! Phase 3: FINALISE
        call ci%build_partition_table(pt_err)
        if (pt_err /= 0) then
            status = 200 + pt_err
            layout_ptr = c_loc(ci)
            return
        end if

        ! Note: Device partitioning (device_local_i, device_local_i_offset,
        ! device_alloc_local, device_n_processes) and communicators (DEVCOMM,
        ! DEVCOMM_NODE) are already correct from the negotiate loop.
        ! device_block_distribute sets the default (block over DEVCOMM),
        ! and propagator callbacks (e.g., circulant/SHAFFT) may override
        ! with their own distribution, deriving host fields bottom-up.

        ! Phase 4: VALIDATE
        block
            integer(int32) :: val_err
            call ci%validate(system_size, val_err)
            if (val_err /= 0) then
                status = 100 + val_err
                layout_ptr = c_loc(ci)
                return
            end if
        end block

        ! Phase 5: LOCK
        call ci%lock(lock_err)
        if (lock_err /= 0) then
            status = 300 + lock_err
            layout_ptr = c_loc(ci)
            return
        end if

        layout_ptr = c_loc(ci)
    contains

        subroutine redistribute_current_layout(error_code)
            integer(int32), intent(out) :: error_code

            error_code = 0
#ifdef WAVEFRONT_BACKEND
            if (backend_flag == 1) then
                ! Zero host fields; device_block_distribute derives them
                ! bottom-up from device partitioning.
                ci%local_i = 0
                ci%local_i_offset = 0
                ci%alloc_local = 0
                call device_block_distribute(ci, error_code)
                return
            end if
#endif
            call block_distribute(ci, comm_size, rank, error_code)
            if (error_code /= 0) return
            ci%device_n_processes = 0
            ci%device_local_i = 0
            ci%device_local_i_offset = 0
            ci%device_alloc_local = 0
        end subroutine redistribute_current_layout

        subroutine reconcile_post_callbacks(prev_process_count, restart)
            integer(int64), intent(in) :: prev_process_count
            logical, intent(out) :: restart

            restart = .false.

            if (host_membership_changed()) then
                call ci%filter_active_ranks(filter_err)
                if (filter_err /= 0) then
                    status = 7
                    return
                end if
                if (ci%SUBCOMM == MPI_COMM_NULL) then
                    status = -1
                    return
                end if
                ! filter_active_ranks already redistributed internally;
                ! just sync the negotiate-local comm_size / rank variables.
                call MPI_Comm_size(ci%SUBCOMM, comm_size, ierr)
                call MPI_Comm_rank(ci%SUBCOMM, rank, ierr)
                restart = .true.
                return
            end if

            if (ci%n_processes < prev_process_count) then
                call ci%shrink(ci%n_processes, shrink_err)
                if (shrink_err /= 0) then
                    status = 4
                    return
                end if
                if (ci%SUBCOMM == MPI_COMM_NULL) then
                    status = -1
                    return
                end if
                ! shrink already redistributed internally;
                ! just sync the negotiate-local comm_size / rank variables.
                call MPI_Comm_size(ci%SUBCOMM, comm_size, ierr)
                call MPI_Comm_rank(ci%SUBCOMM, rank, ierr)
                restart = .true.
                return
            end if

#ifdef WAVEFRONT_BACKEND
            if (backend_flag == 1 .and. device_membership_changed()) then
                call ci%rebuild_communicators(rebuild_err)
                if (rebuild_err /= 0) then
                    status = 6
                    return
                end if
                restart = .true.
            end if
#endif
        end subroutine reconcile_post_callbacks

        logical function host_membership_changed()
            host_membership_changed = .false.

            local_filter_int = 0
            if (ci%local_i == 0_int64) then
                local_filter_int = 1
            end if
            call MPI_Allreduce(local_filter_int, global_filter_int, &
                               1, MPI_INTEGER, MPI_MAX, ci%SUBCOMM, ierr)
            host_membership_changed = (global_filter_int /= 0)
        end function host_membership_changed

#ifdef WAVEFRONT_BACKEND
        logical function device_membership_changed()
            device_membership_changed = .false.

            local_rebuild_int = 0
            if ((ci%device_local_i > 0_int64) .neqv. (ci%DEVCOMM /= MPI_COMM_NULL)) then
                local_rebuild_int = 1
            end if
            call MPI_Allreduce(local_rebuild_int, global_rebuild_int, &
                               1, MPI_INTEGER, MPI_MAX, ci%SUBCOMM, ierr)
            device_membership_changed = (global_rebuild_int /= 0)
        end function device_membership_changed
#endif
    end subroutine negotiate

    ! ====================================================================
    ! negotiate -- helper: block distribution
    ! ====================================================================

    subroutine block_distribute(ci, comm_size, rank, error_code)
        !! Compute simple block distribution and set on ci.
        !! NOT collective -- purely local arithmetic.
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(in) :: comm_size, rank
        integer(int32), intent(out) :: error_code
        integer(int64) :: base_size, remainder, li, li_off

        error_code = 0
        base_size = ci%system_size / int(comm_size, int64)
        remainder = mod(ci%system_size, int(comm_size, int64))
        if (rank < int(remainder, int32)) then
            li = base_size + 1
            li_off = int(rank, int64) * li
        else
            li = base_size
            li_off = int(rank, int64) * li + remainder
        end if
        call ci%set_partitioning(li, li_off, error_code=error_code)
        if (error_code /= 0) return
        ci%alloc_local = li
    end subroutine block_distribute

#ifdef WAVEFRONT_BACKEND
    subroutine sync_layout_from_device_partition(ci, device_local_i, device_local_i_offset, error_code)
        !! Derive host-level fields bottom-up from a device partition and
        !! update the layout via setters. Intended for negotiate-time use
        !! (layout must not be locked).
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int64), intent(in) :: device_local_i, device_local_i_offset
        integer(int32), intent(out) :: error_code

        integer(int32) :: ierr, has_data, node_ranks_with_data
        integer(int64) :: DEVCOMM_NODE_total_local_i, DEVCOMM_NODE_rank_0_offset
        integer(int64) :: NODECOMM_local_i, NODECOMM_local_i_offset

        error_code = 0

        call DEVCOMM_NODE_layout_from_DEVCOMM(device_local_i, device_local_i_offset, &
                                              ci%DEVCOMM_NODE, &
                                              ci%DEVCOMM, &
                                              DEVCOMM_NODE_total_local_i, &
                                              DEVCOMM_NODE_rank_0_offset)

        call NODECOMM_layout_from_DEVCOMM_NODE(DEVCOMM_NODE_total_local_i, &
                                               DEVCOMM_NODE_rank_0_offset, &
                                               ci%DEVCOMM_NODE, &
                                               ci%NODECOMM, &
                                               NODECOMM_local_i, &
                                               NODECOMM_local_i_offset, &
                                               ci%SUBCOMM, &
                                               ci%topology%node_id, &
                                               ci%topology%n_nodes, &
                                               active_device_local_i=device_local_i, &
                                               cpu_numa_node=ci%topology%cpu_numa_node)

        call ci%set_partitioning(NODECOMM_local_i, &
                                 DEVCOMM_NODE_rank_0_offset + NODECOMM_local_i_offset, &
                                 device_local_i, device_local_i_offset, error_code)
        if (error_code /= 0) return

        if (device_local_i > 0_int64) then
            has_data = 1
        else
            has_data = 0
        end if
        call MPI_Allreduce(has_data, node_ranks_with_data, 1, MPI_INTEGER, &
                           MPI_SUM, ci%NODECOMM, ierr)
        call ci%set_device_n_processes(int(node_ranks_with_data, int64), error_code)
    end subroutine sync_layout_from_device_partition

    subroutine device_block_distribute(ci, error_code)
        !! Block-distribute system_size over DEVCOMM ranks, then derive
        !! host-level fields bottom-up via DEVCOMM_NODE and NODECOMM.
        !! COLLECTIVE over SUBCOMM (and sub-communicators).
        type(quop_mpi_layout_t), intent(inout) :: ci
        integer(int32), intent(out) :: error_code

        integer(int32) :: ierr, devcomm_size, devcomm_rank
        integer(int64) :: base_size, remainder
        integer(int64) :: dev_li, dev_li_off

        error_code = 0

        ! Block-distribute over DEVCOMM
        dev_li = 0
        dev_li_off = 0
        if (ci%DEVCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_size(ci%DEVCOMM, devcomm_size, ierr)
            call MPI_Comm_rank(ci%DEVCOMM, devcomm_rank, ierr)

            base_size = ci%system_size / int(devcomm_size, int64)
            remainder = mod(ci%system_size, int(devcomm_size, int64))
            if (devcomm_rank < int(remainder, int32)) then
                dev_li = base_size + 1
                dev_li_off = int(devcomm_rank, int64) * dev_li
            else
                dev_li = base_size
                dev_li_off = int(devcomm_rank, int64) * dev_li + remainder
            end if
        end if

        ! Set device fields
        ci%device_alloc_local = dev_li

        call sync_layout_from_device_partition(ci, dev_li, dev_li_off, error_code)
        if (error_code /= 0) return
        ci%alloc_local = ci%local_i

    end subroutine device_block_distribute
#endif

    subroutine create_jaccomm(MPI_COMM, split_ptr, layout_ptr)
        !! Post-negotiate: Build JACCOMM for parallel Jacobian evaluation.
        !! Collective on MPI_COMM (ALL ranks must call).
        !!
        !! JACCOMM membership:
        !!   - root of subcomm 0 (the optimizer rank, worker_id == 0)
        !!   - ALL ranks of worker subcomms (worker_id > 0)
        !! Non-root ranks of subcomm 0 and inactive ranks get MPI_COMM_NULL.
        !!
        !! Stores JACCOMM into split_info_t.
        integer(int32), intent(in) :: MPI_COMM
        type(c_ptr), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: layout_ptr

        type(split_info_t), pointer :: si
        type(quop_mpi_layout_t), pointer :: ci
        integer(int32) :: mpi_rank, subcomm_rank, color, ierr

        call c_f_pointer(split_ptr, si)
        call MPI_Comm_rank(MPI_COMM, mpi_rank, ierr)

        ! Free any prior JACCOMM to prevent leaks on repeated calls
        if (si%JACCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(si%JACCOMM, ierr)
            si%JACCOMM = MPI_COMM_NULL
        end if

        ! Guard: if negotiate returned an error (status /= 0), layout_ptr
        ! may be c_null_ptr.  Treat as inactive rank.
        if (.not. c_associated(layout_ptr)) then
            color = MPI_UNDEFINED
            call MPI_Comm_split(MPI_COMM, color, mpi_rank, si%JACCOMM, ierr)
            return
        end if

        call c_f_pointer(layout_ptr, ci)

        ! Guard against dangling layout pointer (already destroyed):
        ! a destroyed layout has MPI_COMM == MPI_COMM_NULL.
        if (ci%MPI_COMM == MPI_COMM_NULL .or. ci%SUBCOMM == MPI_COMM_NULL) then
            color = MPI_UNDEFINED
            call MPI_Comm_split(MPI_COMM, color, mpi_rank, si%JACCOMM, ierr)
            return
        end if

        if (si%worker_id > 0) then
            ! Worker subcomm: ALL ranks participate in Jacobian evaluation
            color = 0
        else if (si%worker_id == 0) then
            ! Optimizer subcomm: only rank 0 (the optimizer) joins JACCOMM
            call MPI_Comm_rank(ci%SUBCOMM, subcomm_rank, ierr)
            if (subcomm_rank == 0) then
                color = 0
            else
                color = MPI_UNDEFINED
            end if
        else
            ! Inactive rank (excluded during negotiate)
            color = MPI_UNDEFINED
        end if

        call MPI_Comm_split(MPI_COMM, color, mpi_rank, si%JACCOMM, ierr)
    end subroutine create_jaccomm

    subroutine create_rootcomm(MPI_COMM, split_ptr, layout_ptr)
        !! Post-negotiate: Build ROOTCOMM connecting rank 0 of every
        !! post-negotiate SUBCOMM (i.e. the subcomm "leaders").
        !! Collective on MPI_COMM (ALL ranks must call).
        !!
        !! Leader ranks get color = 0.
        !! All other ranks (including inactive) get MPI_UNDEFINED.
        !! Stores ROOTCOMM into split_info_t.
        integer(int32), intent(in) :: MPI_COMM
        type(c_ptr), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: layout_ptr

        type(split_info_t), pointer :: si
        type(quop_mpi_layout_t), pointer :: ci
        integer(int32) :: mpi_rank, subcomm_rank, color, ierr

        call c_f_pointer(split_ptr, si)
        call MPI_Comm_rank(MPI_COMM, mpi_rank, ierr)

        ! Free any prior ROOTCOMM to prevent leaks on repeated calls
        if (si%ROOTCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(si%ROOTCOMM, ierr)
            si%ROOTCOMM = MPI_COMM_NULL
        end if

        ! Guard: if negotiate returned an error (status /= 0), layout_ptr
        ! may be c_null_ptr.  Treat as inactive rank.
        if (.not. c_associated(layout_ptr)) then
            color = MPI_UNDEFINED
            call MPI_Comm_split(MPI_COMM, color, mpi_rank, si%ROOTCOMM, ierr)
            return
        end if

        call c_f_pointer(layout_ptr, ci)

        ! Guard against dangling layout pointer (already destroyed):
        ! a destroyed layout has MPI_COMM == MPI_COMM_NULL.
        if (ci%MPI_COMM == MPI_COMM_NULL .or. ci%SUBCOMM == MPI_COMM_NULL) then
            color = MPI_UNDEFINED
            call MPI_Comm_split(MPI_COMM, color, mpi_rank, si%ROOTCOMM, ierr)
            return
        end if

        ! Determine if this rank is the leader of its SUBCOMM
        if (ci%SUBCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_rank(ci%SUBCOMM, subcomm_rank, ierr)
            if (subcomm_rank == 0) then
                color = 0
            else
                color = MPI_UNDEFINED
            end if
        else
            ! Inactive rank (excluded during negotiate)
            color = MPI_UNDEFINED
        end if

        call MPI_Comm_split(MPI_COMM, color, mpi_rank, si%ROOTCOMM, ierr)
    end subroutine create_rootcomm

    subroutine create_split_from_subcomm(si_out, MPI_COMM, SUBCOMM, &
                                         worker_id, n_workers)
        !! Create a split_info_t from an existing SUBCOMM.
        !! SUBCOMM is MPI_Comm_dup'd so negotiate() can take ownership
        !! without invalidating the caller's handle.
        type(split_info_t), pointer, intent(out) :: si_out
        integer(int32), intent(in) :: MPI_COMM
        integer(int32), intent(in) :: SUBCOMM
        integer(int32), intent(in) :: worker_id
        integer(int32), intent(in) :: n_workers

        integer(int32) :: ierr

        allocate (si_out)
        si_out%MPI_COMM = MPI_COMM
        si_out%worker_id = worker_id
        si_out%n_workers = n_workers
        si_out%JACCOMM = MPI_COMM_NULL
        si_out%ROOTCOMM = MPI_COMM_NULL

        if (SUBCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_dup(SUBCOMM, si_out%SUBCOMM, ierr)
        else
            si_out%SUBCOMM = MPI_COMM_NULL
        end if
    end subroutine create_split_from_subcomm

    ! ====================================================================
    ! dump_comm_info  -- pure-Fortran diagnostic dump
    ! ====================================================================
    !
    ! Module-level convenience entry point.  Operates on a c_ptr handle
    ! so that the f2py wrapper can call it directly.

    subroutine dump_comm_info(ci_ptr, phase, phase_len)
        !! Check QUOP_DUMP_COMM_INFO env var, gather layout data on
        !! rank 0 of MPI_COMM, and write a human-readable text file.
        !! COLLECTIVE over MPI_COMM.
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(in) :: phase_len
        character(len=phase_len), intent(in) :: phase

        type(quop_mpi_layout_t), pointer :: ci

        if (.not. c_associated(ci_ptr)) return
        call c_f_pointer(ci_ptr, ci)
        call ci%dump(phase)
    end subroutine dump_comm_info

    ! ====================================================================
    ! layout_dump_comm_info  -- type-bound implementation
    ! ====================================================================

    subroutine layout_dump_comm_info(self, phase)
        !! Gather all quop_mpi_layout_t data to rank 0 of MPI_COMM and
        !! write a human-readable text file.
        !!
        !! Triggered by the QUOP_DUMP_COMM_INFO environment variable:
        !!   "1"   -> dump to CWD
        !!   <dir> -> dump to that directory (created if needed)
        !!   unset -> no-op (zero overhead)
        !!
        !! COLLECTIVE over MPI_COMM.
        class(quop_mpi_layout_t), intent(in) :: self
        character(len=*), intent(in)         :: phase

        ! -- Per-rank record for MPI_Gather --
        ! Packed into 10 x int64 + 19 x int32, plus binding mode and hostname strings.
        integer, parameter :: N_I64 = 10, N_I32 = 19, BIND_MODE_MAXLEN = 16
        integer(int64) :: send_i64(N_I64)
        integer(int32) :: send_i32(N_I32)
        integer(int64), allocatable :: recv_i64(:, :)
        integer(int32), allocatable :: recv_i32(:, :)
        integer, parameter :: PROC_NAME_MAXLEN = MPI_MAX_PROCESSOR_NAME
        character(len=BIND_MODE_MAXLEN) :: send_binding_mode
        character(len=BIND_MODE_MAXLEN), allocatable :: recv_binding_mode(:)
        character(len=PROC_NAME_MAXLEN) :: send_proc_name
        character(len=PROC_NAME_MAXLEN), allocatable :: recv_proc_name(:)

        character(len=512) :: env_val, output_dir, filename
        integer(int32) :: env_len, env_stat
        integer(int32) :: mpi_rank, mpi_size, ierr
        integer(int32) :: sc_rank, sc_size, nc_rank, nc_size
        integer(int32) :: dc_rank, dc_size, dn_rank, dn_size
        integer(int32) :: funit, i, ref_idx, active_sc_size
        integer(int32) :: date_values(8)
        character(len=15)  :: timestamp
        character(len=1024) :: filepath
        logical :: dir_exists
        logical :: dump_enabled, env_is_logical
        logical :: header_locked

        ! -- 1. Check environment variable --------------------------
        call GET_ENVIRONMENT_VARIABLE("QUOP_DUMP_COMM_INFO", &
                                      env_val, env_len, env_stat)
        if (env_stat /= 0 .or. env_len == 0) return ! unset -> no-op

        output_dir = ''
        call parse_logical_token(trim(adjustl(env_val)), dump_enabled, env_is_logical)
        if (env_is_logical) then
            if (.not. dump_enabled) return
            output_dir = '.'
        else
            output_dir = trim(env_val)
        end if

        ! -- 2. Early exit if communicator is invalid ---------------
        if (self%MPI_COMM == MPI_COMM_NULL) return

        call MPI_Comm_rank(self%MPI_COMM, mpi_rank, ierr)
        call MPI_Comm_size(self%MPI_COMM, mpi_size, ierr)

        ! -- 3. Pack per-rank data ----------------------------------
        ! int64 fields
        send_i64(1) = self%local_i
        send_i64(2) = self%local_i_offset
        send_i64(3) = self%alloc_local
        send_i64(4) = self%device_local_i
        send_i64(5) = self%device_local_i_offset
        send_i64(6) = self%device_alloc_local
        send_i64(7) = self%device_n_processes
        send_i64(8) = self%system_size
        send_i64(9) = self%n_processes
        if (self%locked) then
            send_i64(10) = 1
        else
            send_i64(10) = 0
        end if

        ! int32: communicator rank/size pairs + GPU topology
        call safe_comm_rank_size(self%SUBCOMM, sc_rank, sc_size)
        call safe_comm_rank_size(self%NODECOMM, nc_rank, nc_size)
        call safe_comm_rank_size(self%DEVCOMM, dc_rank, dc_size)
        call safe_comm_rank_size(self%DEVCOMM_NODE, dn_rank, dn_size)

        send_i32(1) = sc_rank
        send_i32(2) = sc_size
        send_i32(3) = nc_rank
        send_i32(4) = nc_size
        send_i32(5) = dc_rank
        send_i32(6) = dc_size
        send_i32(7) = dn_rank
        send_i32(8) = dn_size
        send_i32(9) = self%topology%n_physical_gpus
        send_i32(10) = self%topology%visible_device_count
        send_i32(11) = self%topology%assigned_device_id
        if (self%DEVCOMM /= MPI_COMM_NULL) then
            send_i32(12) = 1
        else
            send_i32(12) = 0
        end if
        send_i32(13) = self%topology%n_nodes
        send_i32(14) = self%topology%node_id
        send_i32(15) = self%topology%node_size
        send_i32(16) = self%topology%my_gpu_index
        send_i32(17) = self%topology%cpu_numa_node
        send_i32(18) = self%topology%rank_within_cpu_numa
        send_i32(19) = self%topology%rank_within_gpu

        send_binding_mode = self%topology%binding_mode
        ! Processor/host name (stored as invariant in topology)
        send_proc_name = self%topology%hostname

        ! -- 4. Gather on rank 0 -----------------------------------
        if (mpi_rank == 0) then
            allocate (recv_i64(N_I64, mpi_size))
            allocate (recv_i32(N_I32, mpi_size))
            allocate (recv_binding_mode(mpi_size))
            allocate (recv_proc_name(mpi_size))
        else
            allocate (recv_i64(1, 1)) ! dummy
            allocate (recv_i32(1, 1))
            allocate (recv_binding_mode(1))
            allocate (recv_proc_name(1))
        end if

        call MPI_Gather(send_i64, N_I64, MPI_INTEGER8, &
                        recv_i64, N_I64, MPI_INTEGER8, 0, self%MPI_COMM, ierr)
        call MPI_Gather(send_i32, N_I32, MPI_INTEGER4, &
                        recv_i32, N_I32, MPI_INTEGER4, 0, self%MPI_COMM, ierr)
        call MPI_Gather(send_binding_mode, BIND_MODE_MAXLEN, MPI_CHARACTER, &
                        recv_binding_mode, BIND_MODE_MAXLEN, MPI_CHARACTER, 0, self%MPI_COMM, ierr)
        call MPI_Gather(send_proc_name, PROC_NAME_MAXLEN, MPI_CHARACTER, &
                        recv_proc_name, PROC_NAME_MAXLEN, MPI_CHARACTER, 0, self%MPI_COMM, ierr)

        ! -- 5. Rank 0 writes the file -----------------------------
        if (mpi_rank == 0) then
            ref_idx = 0
            do i = 1, mpi_size
                if (recv_i32(1, i) == 0) then
                    ref_idx = i
                    exit
                end if
            end do
            if (ref_idx == 0) then
                do i = 1, mpi_size
                    if (recv_i32(1, i) >= 0) then
                        ref_idx = i
                        exit
                    end if
                end do
            end if

            active_sc_size = 0
            header_locked = .false.
            if (ref_idx > 0) then
                active_sc_size = recv_i32(2, ref_idx)
                header_locked = (recv_i64(10, ref_idx) /= 0)
            end if

            ! Build timestamp: YYYYMMDD_HHMMSS
            call DATE_AND_TIME(VALUES=date_values)
            write (timestamp, '(I4.4,I2.2,I2.2,"_",I2.2,I2.2,I2.2)') &
                date_values(1), date_values(2), date_values(3), &
                date_values(5), date_values(6), date_values(7)

            write (filename, '(A,A,A,A)') &
                'quop_comm_info_', trim(phase), '_', trim(timestamp)
            filename = trim(filename)//'.txt'

            ! Ensure output directory exists (use mkdir -p via EXECUTE_COMMAND_LINE)
            if (trim(output_dir) /= '.') then
                inquire (file=trim(output_dir)//'/.', exist=dir_exists)
                if (.not. dir_exists) then
                    call EXECUTE_COMMAND_LINE('mkdir -p '//trim(output_dir), &
                                              exitstat=ierr)
                end if
            end if

            write (filepath, '(A,A,A)') trim(output_dir), '/', trim(filename)

            open (newunit=funit, file=trim(filepath), status='replace', &
                  action='write', iostat=ierr)
            if (ierr /= 0) then
                write (error_unit, '(A,A)') "WARNING: dump_comm_info: could not open ", &
                    trim(filepath)
                deallocate (recv_i64, recv_i32, recv_binding_mode, recv_proc_name)
                return
            end if

            ! -- Header ---------------------------------------------
            write (funit, '(A,A,A)') &
                'QuOp_MPI quop_mpi_layout_t dump (', trim(phase), ')'
            write (funit, '(A)') repeat('=', 70)
            write (funit, '(A,I0)') 'system_size     = ', self%system_size
            write (funit, '(A,I0)') 'n_processes     = ', active_sc_size
            write (funit, '(A,I0)') 'MPI_COMM size   = ', mpi_size
            if (ref_idx > 0) then
                write (funit, '(A,I0)') 'active topology n_nodes = ', recv_i32(13, ref_idx)
            else
                write (funit, '(A,I0)') 'active topology n_nodes = ', 0
            end if
            if (header_locked) then
                write (funit, '(A)') 'locked          = True'
            else
                write (funit, '(A)') 'locked          = False'
            end if
            write (funit, '(A)') ''

            ! -- Per-rank table -------------------------------------
            write (funit, '(A)') 'Per-rank data (ordered by MPI_COMM rank):'
            ! Header row
            write (funit, '(A)') &
                '  Rank          li      li_off       alloc        d_li       d_alc       d_off'// &
                '    SC_r  SC_s  NC_r  NC_s  DC_r  DC_s  DN_r  DN_s   GPU  phys  gpu?  cpuNm'// &
                '   rCpuN   rGpu   node  mode        hostname'
            write (funit, '(A)') repeat('-', 290)

            ! Data rows
            do i = 1, mpi_size
                write (funit, '(I6,2X,I11,2X,I11,2X,I11,2X,I11,2X,I11,2X,I11,2X,'// &
                       'I5,2X,I5,2X,I5,2X,I5,2X,I5,2X,I5,2X,I5,2X,I5,2X,'// &
                       'I5,2X,I5,2X,I5,2X,I6,2X,I7,2X,I6,2X,I6,2X,A10,2X,A)') &
                    i - 1, &
                    recv_i64(1, i), recv_i64(2, i), recv_i64(3, i), & ! li, li_off, alloc
                    recv_i64(4, i), recv_i64(6, i), recv_i64(5, i), & ! d_li, d_alc, d_off
                    recv_i32(1, i), recv_i32(2, i), & ! SC_r, SC_s
                    recv_i32(3, i), recv_i32(4, i), & ! NC_r, NC_s
                    recv_i32(5, i), recv_i32(6, i), & ! DC_r, DC_s
                    recv_i32(7, i), recv_i32(8, i), & ! DN_r, DN_s
                    recv_i32(11, i), recv_i32(16, i), recv_i32(12, i), & ! GPU, phys, gpu?
                    recv_i32(17, i), recv_i32(18, i), recv_i32(19, i), & ! cpu_numa, rank_within_cpu_numa, rank_within_gpu
                    recv_i32(14, i), adjustl(recv_binding_mode(i)), & ! node_id, binding mode
                    trim(recv_proc_name(i)) ! hostname
            end do
            write (funit, '(A)') ''

            ! -- Partition table (locked phase only) ----------------
            if (trim(phase) == 'locked' .and. active_sc_size > 0) then
                block
                    integer(int32) :: active_rank, pt_idx
                    integer(int64), allocatable :: active_local_i(:), partition_table(:)

                    allocate (active_local_i(active_sc_size))
                    active_local_i = 0_int64
                    do i = 1, mpi_size
                        active_rank = recv_i32(1, i)
                        if (active_rank >= 0 .and. active_rank < active_sc_size) then
                            active_local_i(active_rank + 1) = recv_i64(1, i)
                        end if
                    end do

                    allocate (partition_table(active_sc_size + 1))
                    partition_table(1) = 1_int64
                    do pt_idx = 1, active_sc_size
                        partition_table(pt_idx + 1) = partition_table(pt_idx) + active_local_i(pt_idx)
                    end do

                    write (funit, '(A)') 'Partition table (1-based cumulative):'
                    write (funit, '(A)', advance='no') '  ['
                    do i = 1, size(partition_table)
                        if (i > 1) write (funit, '(A)', advance='no') ', '
                        write (funit, '(I0)', advance='no') partition_table(i)
                    end do
                    write (funit, '(A)') ']'
                    write (funit, '(A)') ''

                    deallocate (active_local_i, partition_table)
                end block
            end if

            ! -- Footer / column legend -----------------------------
            write (funit, '(A)') repeat('=', 70)
            write (funit, '(A)') 'Column legend:'
            write (funit, '(A)') '  Rank          : MPI_COMM rank (0-based)'
            write (funit, '(A)') '  li            : local_i (host elements on this rank)'
            write (funit, '(A)') '  li_off        : local_i_offset (host global start index)'
            write (funit, '(A)') '  alloc         : alloc_local (host allocation length; >= li)'
            write (funit, '(A)') '  d_li          : device_local_i (device elements on this rank)'
            write (funit, '(A)') '  d_alc         : device_alloc_local (device allocation length; >= d_li)'
            write (funit, '(A)') '  d_off         : device_local_i_offset (device global start index)'
            write (funit, '(A)') '  SC_r/SC_s     : SUBCOMM rank/size'
            write (funit, '(A)') '  NC_r/NC_s     : NODECOMM rank/size'
            write (funit, '(A)') '  DC_r/DC_s     : DEVCOMM rank/size'
            write (funit, '(A)') '  DN_r/DN_s     : DEVCOMM_NODE rank/size'
            write (funit, '(A)') '  GPU           : assigned_device_id (GPU device id for this rank)'
            write (funit, '(A)') '  phys          : my_gpu_index (physical GPU index on this node)'
            write (funit, '(A)') '  gpu?          : current DEVCOMM membership (1 if DEVCOMM /= MPI_COMM_NULL)'
            write (funit, '(A)') '  cpuNm         : cpu_numa_node (best NUMA match for this rank''s CPU affinity)'
            write (funit, '(A)') '  rCpuN         : rank_within_cpu_numa (lower NODECOMM ranks on same CPU NUMA node)'
            write (funit, '(A)') '  rGpu          : rank_within_gpu (lower NODECOMM ranks sharing this GPU)'
            write (funit, '(A)') '  node          : active node_id (0-based within current SUBCOMM; -1 if excluded)'
            write (funit, '(A)') '  mode          : GPU binding mode used by topology assignment'
            write (funit, '(A)') '  hostname      : MPI processor name (often the hostname)'
            write (funit, '(A)') '  Note: *_r=-1 and *_s=0 indicates MPI_COMM_NULL.'
            write (funit, '(A)') ''

            ! -- Per-node summary (topology invariants) -------------
            block
                integer(int32) :: total_nodes, nid, r
                integer(int32), allocatable :: ranks_on_node(:), active_on_node(:)
                integer(int32), allocatable :: node_sz(:), ngpu(:), vdev_min(:), vdev_max(:)
                integer(int64), allocatable :: d_np(:)
                character(len=PROC_NAME_MAXLEN), allocatable :: host(:)
                logical, allocatable :: host_mixed(:)
                character(len=PROC_NAME_MAXLEN) :: this_host

                total_nodes = maxval(recv_i32(14, 1:mpi_size)) + 1
                if (total_nodes < 1) total_nodes = 1

                allocate (ranks_on_node(total_nodes))
                allocate (active_on_node(total_nodes))
                allocate (node_sz(total_nodes))
                allocate (ngpu(total_nodes))
                allocate (vdev_min(total_nodes))
                allocate (vdev_max(total_nodes))
                allocate (d_np(total_nodes))
                allocate (host(total_nodes))
                allocate (host_mixed(total_nodes))

                ranks_on_node = 0
                active_on_node = 0
                node_sz = -1
                ngpu = -1
                vdev_min = huge(0_int32)
                vdev_max = -huge(0_int32)
                d_np = 0_int64
                host = ''
                host_mixed = .false.

                do r = 1, mpi_size
                    nid = recv_i32(14, r) + 1 ! 0-based -> 1-based
                    if (nid < 1 .or. nid > total_nodes) cycle

                    ranks_on_node(nid) = ranks_on_node(nid) + 1
                    if (recv_i32(1, r) >= 0) active_on_node(nid) = active_on_node(nid) + 1

                    if (node_sz(nid) < 0) node_sz(nid) = recv_i32(15, r)
                    if (ngpu(nid) < 0) ngpu(nid) = recv_i32(9, r)
                    vdev_min(nid) = min(vdev_min(nid), recv_i32(10, r))
                    vdev_max(nid) = max(vdev_max(nid), recv_i32(10, r))
                    d_np(nid) = max(d_np(nid), recv_i64(7, r))

                    this_host = recv_proc_name(r)
                    if (len_trim(host(nid)) == 0) then
                        host(nid) = this_host
                    else if (trim(host(nid)) /= trim(this_host)) then
                        host_mixed(nid) = .true.
                    end if
                end do

                write (funit, '(A)') 'Per-node summary (discover_topology invariants):'
                write (funit, '(A)') &
                    '  node  ranks  actv  node_sz  ngpu  vdev_min  vdev_max  d_np        hostname'
                write (funit, '(A)') repeat('-', 120)
                do nid = 1, total_nodes
                    if (host_mixed(nid)) host(nid) = '<mixed>'
                    write (funit, '(I6,2X,I5,2X,I4,2X,I7,2X,I5,2X,I8,2X,I8,2X,I10,2X,A)') &
                        nid - 1, ranks_on_node(nid), active_on_node(nid), node_sz(nid), &
                        ngpu(nid), vdev_min(nid), vdev_max(nid), d_np(nid), trim(host(nid))
                end do
                write (funit, '(A)') ''
            end block

            close (funit)
        end if

        deallocate (recv_i64, recv_i32, recv_binding_mode, recv_proc_name)

    contains

        subroutine safe_comm_rank_size(comm, r, s)
            integer(int32), intent(in)  :: comm
            integer(int32), intent(out) :: r, s
            integer(int32) :: local_ierr
            if (comm /= MPI_COMM_NULL) then
                call MPI_Comm_rank(comm, r, local_ierr)
                call MPI_Comm_size(comm, s, local_ierr)
            else
                r = -1
                s = 0
            end if
        end subroutine safe_comm_rank_size

        subroutine parse_logical_token(raw_value, logical_value, is_logical)
            character(len=*), intent(in) :: raw_value
            logical, intent(out) :: logical_value
            logical, intent(out) :: is_logical

            character(len=len(raw_value)) :: normalized
            integer :: j, code

            normalized = trim(adjustl(raw_value))
            do j = 1, len_trim(normalized)
                code = iachar(normalized(j:j))
                if (code >= iachar('A') .and. code <= iachar('Z')) then
                    normalized(j:j) = achar(code + 32)
                end if
            end do

            select case (trim(normalized))
            case ('1', 'true', 'yes', 'on')
                logical_value = .true.
                is_logical = .true.
            case ('0', 'false', 'no', 'off')
                logical_value = .false.
                is_logical = .true.
            case default
                logical_value = .false.
                is_logical = .false.
            end select
        end subroutine parse_logical_token

    end subroutine layout_dump_comm_info

    ! ====================================================================
    ! quop_mpi_layout_t -- Getters
    ! ====================================================================

    pure integer(int64) function layout_get_system_size(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_system_size = self%system_size
    end function layout_get_system_size

    pure integer(int64) function layout_get_n_processes(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_n_processes = self%n_processes
    end function layout_get_n_processes

    pure integer(int64) function layout_get_local_i(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_local_i = self%local_i
    end function layout_get_local_i

    pure integer(int64) function layout_get_local_i_offset(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_local_i_offset = self%local_i_offset
    end function layout_get_local_i_offset

    pure integer(int64) function layout_get_alloc_local(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_alloc_local = self%alloc_local
    end function layout_get_alloc_local

    pure integer(int64) function layout_get_device_local_i(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_device_local_i = self%device_local_i
    end function layout_get_device_local_i

    pure integer(int64) function layout_get_device_local_i_offset(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_device_local_i_offset = self%device_local_i_offset
    end function layout_get_device_local_i_offset

    pure integer(int64) function layout_get_device_alloc_local(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_device_alloc_local = self%device_alloc_local
    end function layout_get_device_alloc_local

    pure integer(int64) function layout_get_device_n_processes(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_device_n_processes = self%device_n_processes
    end function layout_get_device_n_processes

    pure integer(int32) function layout_get_SUBCOMM(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_SUBCOMM = self%SUBCOMM
    end function layout_get_SUBCOMM

    pure integer(int32) function layout_get_NODECOMM(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_NODECOMM = self%NODECOMM
    end function layout_get_NODECOMM

    pure integer(int32) function layout_get_DEVCOMM(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_DEVCOMM = self%DEVCOMM
    end function layout_get_DEVCOMM

    pure integer(int32) function layout_get_DEVCOMM_NODE(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_DEVCOMM_NODE = self%DEVCOMM_NODE
    end function layout_get_DEVCOMM_NODE

    pure integer(int32) function layout_get_MPI_COMM(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_MPI_COMM = self%MPI_COMM
    end function layout_get_MPI_COMM

    function layout_get_partition_table(self) result(tbl)
        !! Return a pointer to the partition table, or null if not allocated.
        class(quop_mpi_layout_t), target, intent(in) :: self
        integer(int64), pointer :: tbl(:)
        if (allocated(self%partition_table)) then
            tbl => self%partition_table
        else
            tbl => null()
        end if
    end function layout_get_partition_table

    pure function layout_get_topology(self) result(topo)
        class(quop_mpi_layout_t), intent(in) :: self
        type(gpu_topology_t) :: topo
        topo = self%topology
    end function layout_get_topology

    pure logical function layout_get_requires_device_work_buffer(self)
        class(quop_mpi_layout_t), intent(in) :: self
        layout_get_requires_device_work_buffer = self%requires_device_work_buffer
    end function layout_get_requires_device_work_buffer

    ! ====================================================================
    ! quop_mpi_layout_t -- Additional Setters
    ! ====================================================================

    subroutine layout_set_requires_device_work_buffer(self, flag, error_code)
        !! Set requires_device_work_buffer flag.
        !! Rejects writes on a locked layout.
        class(quop_mpi_layout_t), intent(inout) :: self
        logical, intent(in) :: flag
        integer(int32), intent(out) :: error_code

        error_code = 0
        if (self%locked) then
            call layout_note_error(error_code, 1, &
                                   "ERROR: cannot set requires_device_work_buffer on locked quop_mpi_layout_t")
            return
        end if

        self%requires_device_work_buffer = flag
    end subroutine layout_set_requires_device_work_buffer

end module comm_info_module
