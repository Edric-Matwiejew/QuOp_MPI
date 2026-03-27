! comm_info_wrapper.f90
!
! f2py-facing C-interop wrappers for quop_mpi_layout_t.
! Follows the same opaque c_ptr pattern as context_wrapper.f90 and
! propagator_wrapper.f90.

module comm_info_wrapper

    use iso_fortran_env, only: real64, int32, int64
    use iso_c_binding, only: c_loc, c_f_pointer, c_ptr, c_null_ptr, c_associated
    use MPI, only: MPI_SUCCESS, MPI_Comm_dup, MPI_Comm_size, MPI_Comm_free
    use comm_info_module, only: quop_mpi_layout_t, split_info_t, &
                                discover_topology_impl => discover_topology, &
                                destroy_topology_impl => destroy_topology, &
                                get_topology_info_impl => get_topology_info, &
                                get_layout_topology_info_impl => get_layout_topology_info, &
                                split_workers_impl => split_workers, &
                                negotiate_impl => negotiate, &
                                create_jaccomm_impl => create_jaccomm, &
                                create_rootcomm_impl => create_rootcomm, &
                                create_split_from_subcomm_impl => create_split_from_subcomm, &
                                dump_comm_info_impl => dump_comm_info

    implicit none
    public

contains

    ! -- Creation / destruction --------------------------------------

    subroutine create(ci_ptr, MPI_COMM, error_code)
        !! Phase 1 transition shim.
        !! Allocate a quop_mpi_layout_t and store the root communicator.
        !! SUBCOMM = MPI_Comm_dup(MPI_COMM).
        !f2py integer(int64), intent(out) :: ci_ptr
        !f2py integer(int32), intent(in)  :: MPI_COMM
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(out) :: ci_ptr
        integer(int32), intent(in) :: MPI_COMM
        integer(int32), intent(out) :: error_code

        type(quop_mpi_layout_t), pointer :: ci
        integer(int32) :: ierr, comm_size, dup_comm, alloc_status

        ci_ptr = c_null_ptr
        error_code = 0

        allocate (ci, stat=alloc_status)
        if (alloc_status /= 0) then
            error_code = 100
            return
        end if

        call ci%set_MPI_COMM(MPI_COMM, error_code)
        if (error_code /= 0) then
            deallocate (ci)
            return
        end if
        call MPI_Comm_dup(MPI_COMM, dup_comm, ierr)
        if (ierr /= MPI_SUCCESS) then
            error_code = 101
            deallocate (ci)
            return
        end if

        call ci%set_SUBCOMM(dup_comm, error_code)
        if (error_code /= 0) then
            call MPI_Comm_free(dup_comm, ierr)
            deallocate (ci)
            return
        end if
        call MPI_Comm_size(ci%get_SUBCOMM(), comm_size, ierr)
        if (ierr /= MPI_SUCCESS) then
            error_code = 102
            call MPI_Comm_free(dup_comm, ierr)
            deallocate (ci)
            return
        end if

        call ci%set_n_processes(int(comm_size, int64), error_code)
        if (error_code /= 0) then
            call MPI_Comm_free(dup_comm, ierr)
            deallocate (ci)
            return
        end if
        ci_ptr = c_loc(ci)
    end subroutine create

    subroutine destroy(ci_ptr)
        !! Free all owned communicators and deallocate.
        !! MPI_COMM is NOT freed -- the caller owns it.
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        call ci%destroy()
        deallocate (ci)
    end subroutine destroy

    ! -- Lock / unlock -----------------------------------------------

    subroutine lock(ci_ptr, error_code)
        !f2py integer(int64), intent(in) :: ci_ptr
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: error_code
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        call ci%lock(error_code)
    end subroutine lock

    subroutine unlock(ci_ptr, error_code)
        !f2py integer(int64), intent(in) :: ci_ptr
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: error_code
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        call ci%unlock(error_code)
    end subroutine unlock

    subroutine is_locked(ci_ptr, locked_flag)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: locked_flag
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        if (ci%is_locked()) then
            locked_flag = 1
        else
            locked_flag = 0
        end if
    end subroutine is_locked

    ! -- Scalar field accessors --------------------------------------

    subroutine get_local_i(ci_ptr, val)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(out) :: val
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        val = ci%get_local_i()
    end subroutine get_local_i

    subroutine get_local_i_offset(ci_ptr, val)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(out) :: val
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        val = ci%get_local_i_offset()
    end subroutine get_local_i_offset

    subroutine get_n_processes(ci_ptr, val)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(out) :: val
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        val = ci%get_n_processes()
    end subroutine get_n_processes

    subroutine get_system_size(ci_ptr, val)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(out) :: val
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        val = ci%get_system_size()
    end subroutine get_system_size

    subroutine get_alloc_local(ci_ptr, val)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(out) :: val
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        val = ci%get_alloc_local()
    end subroutine get_alloc_local

    subroutine get_device_alloc_local(ci_ptr, val)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(out) :: val
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        val = ci%get_device_alloc_local()
    end subroutine get_device_alloc_local

    subroutine get_device_local_i(ci_ptr, val)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(out) :: val
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        val = ci%get_device_local_i()
    end subroutine get_device_local_i

    subroutine get_device_local_i_offset(ci_ptr, val)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(out) :: val
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        val = ci%get_device_local_i_offset()
    end subroutine get_device_local_i_offset

    subroutine get_device_n_processes(ci_ptr, val)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(out) :: val
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        val = ci%get_device_n_processes()
    end subroutine get_device_n_processes

    ! -- Partition table (array accessor) ----------------------------

    subroutine get_partition_table_size(ci_ptr, n)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: n
        type(quop_mpi_layout_t), pointer :: ci
        integer(int64), pointer :: pt(:)

        call c_f_pointer(ci_ptr, ci)
        pt => ci%get_partition_table()
        if (associated(pt)) then
            n = size(pt)
        else
            n = 0
        end if
    end subroutine get_partition_table_size

    subroutine get_partition_table(ci_ptr, n, table)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(in) :: n
        integer(int64), dimension(n), intent(out) :: table
        type(quop_mpi_layout_t), pointer :: ci
        integer(int64), pointer :: pt(:)

        call c_f_pointer(ci_ptr, ci)
        pt => ci%get_partition_table()
        table(:) = pt(:)
    end subroutine get_partition_table

    ! -- Communicator handles (as Fortran int32 for py2f/f2py) -------

    subroutine get_subcomm(ci_ptr, comm_handle)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: comm_handle
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        comm_handle = ci%get_SUBCOMM()
    end subroutine get_subcomm

    subroutine get_mpi_comm(ci_ptr, comm_handle)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: comm_handle
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        comm_handle = ci%get_MPI_COMM()
    end subroutine get_mpi_comm

    subroutine get_nodecomm(ci_ptr, comm_handle)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: comm_handle
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        comm_handle = ci%get_NODECOMM()
    end subroutine get_nodecomm

    subroutine get_devcomm(ci_ptr, comm_handle)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: comm_handle
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        comm_handle = ci%get_DEVCOMM()
    end subroutine get_devcomm

    subroutine get_devcomm_node(ci_ptr, comm_handle)
        !f2py integer(int64), intent(in) :: ci_ptr
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: comm_handle
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        comm_handle = ci%get_DEVCOMM_NODE()
    end subroutine get_devcomm_node

    ! -- Set partitioning (individual fields) ------------------------

    subroutine set_partitioning(ci_ptr, local_i, local_i_offset, error_code)
        !f2py integer(int64), intent(in) :: ci_ptr
        !f2py integer(int64), intent(in) :: local_i
        !f2py integer(int64), intent(in) :: local_i_offset
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(in) :: local_i, local_i_offset
        integer(int32), intent(out) :: error_code
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        call ci%set_partitioning(local_i, local_i_offset, error_code=error_code)
    end subroutine set_partitioning

    subroutine set_system_size(ci_ptr, system_size, error_code)
        !f2py integer(int64), intent(in) :: ci_ptr
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(in) :: system_size
        integer(int32), intent(out) :: error_code
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        call ci%set_system_size(system_size, error_code)
    end subroutine set_system_size

    subroutine set_n_processes(ci_ptr, n_processes, error_code)
        !f2py integer(int64), intent(in) :: ci_ptr
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(in) :: n_processes
        integer(int32), intent(out) :: error_code
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        call ci%set_n_processes(n_processes, error_code)
    end subroutine set_n_processes

    subroutine set_alloc_local(ci_ptr, alloc_local, error_code)
        !f2py integer(int64), intent(in) :: ci_ptr
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(in) :: alloc_local
        integer(int32), intent(out) :: error_code
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        call ci%set_alloc_local(alloc_local, error_code)
    end subroutine set_alloc_local

    ! -- Build partition table (collective) --------------------------

    subroutine build_partition_table(ci_ptr, error_code)
        !f2py integer(int64), intent(in) :: ci_ptr
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: error_code
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        call ci%build_partition_table(error_code)
    end subroutine build_partition_table

    ! -- Validate (collective) ---------------------------------------

    subroutine validate(ci_ptr, system_size, error_code)
        !f2py integer(int64), intent(in)  :: ci_ptr
        !f2py integer(int64), intent(in)  :: system_size
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(in) :: system_size
        integer(int32), intent(out) :: error_code
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        call ci%validate(system_size, error_code)
    end subroutine validate

    ! -- Communicator management -------------------------------------

    subroutine shrink(ci_ptr, new_size, error_code)
        !f2py integer(int64), intent(in)  :: ci_ptr
        !f2py integer(int64), intent(in)  :: new_size
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(in) :: ci_ptr
        integer(int64), intent(in) :: new_size
        integer(int32), intent(out) :: error_code
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        call ci%shrink(new_size, error_code)
    end subroutine shrink

    subroutine rebuild_communicators(ci_ptr, error_code)
        !f2py integer(int64), intent(in) :: ci_ptr
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: error_code
        type(quop_mpi_layout_t), pointer :: ci

        call c_f_pointer(ci_ptr, ci)
        call ci%rebuild_communicators(error_code)
    end subroutine rebuild_communicators

    ! -- Top-level entry points --------------------------------------

    subroutine wrapper_discover_topology(topo_ptr, MPI_COMM, backend_flag, error_code)
        !! Phase 0: Detect node/GPU topology.
        !! Collective on MPI_COMM.
        !f2py integer(int64), intent(out) :: topo_ptr
        !f2py integer(int32), intent(in)  :: MPI_COMM
        !f2py integer(int32), intent(in)  :: backend_flag
        !f2py integer(int32), intent(out) :: error_code
        type(c_ptr), intent(out) :: topo_ptr
        integer(int32), intent(in) :: MPI_COMM
        integer(int32), intent(in) :: backend_flag
        integer(int32), intent(out) :: error_code

        call discover_topology_impl(topo_ptr, MPI_COMM, backend_flag, error_code)
    end subroutine wrapper_discover_topology

    subroutine wrapper_split_workers(split_ptr, MPI_COMM, topo_ptr, &
                                     n_jacobian_workers, backend_flag, &
                                     worker_id, status)
        !! Phase 0b: Split MPI_COMM into worker groups.
        !! Collective on MPI_COMM.
        !f2py integer(int64), intent(out) :: split_ptr
        !f2py integer(int32), intent(in)  :: MPI_COMM
        !f2py integer(int64), intent(in)  :: topo_ptr
        !f2py integer(int32), intent(in)  :: n_jacobian_workers
        !f2py integer(int32), intent(in)  :: backend_flag
        !f2py integer(int32), intent(out) :: worker_id
        !f2py integer(int32), intent(out) :: status
        type(c_ptr), intent(out) :: split_ptr
        integer(int32), intent(in) :: MPI_COMM
        type(c_ptr), intent(in) :: topo_ptr
        integer(int32), intent(in) :: n_jacobian_workers
        integer(int32), intent(in) :: backend_flag
        integer(int32), intent(out) :: worker_id
        integer(int32), intent(out) :: status

        call split_workers_impl(split_ptr, MPI_COMM, topo_ptr, &
                                n_jacobian_workers, backend_flag, &
                                worker_id, status)
    end subroutine wrapper_split_workers

    subroutine wrapper_negotiate(layout_ptr, split_ptr, topo_ptr, &
                                 system_size, backend_flag, &
                                 n_propagators, propagator_ptrs, &
                                 n_callbacks, callback_ptrs, status)
        !! Phases 1-5: CREATE -> NEGOTIATE -> FINALISE -> VALIDATE -> LOCK.
        !! Collective on per-worker SUBCOMM.
        !! n_callbacks must equal n_propagators (separate for f2py compat).
        !f2py integer(int64), intent(out) :: layout_ptr
        !f2py integer(int64), intent(in)  :: split_ptr
        !f2py integer(int64), intent(in)  :: topo_ptr
        !f2py integer(int64), intent(in)  :: system_size
        !f2py integer(int32), intent(in)  :: backend_flag
        !f2py integer(int32), intent(hide), depend(propagator_ptrs) :: n_propagators = len(propagator_ptrs)
        !f2py integer(int64), intent(in)  :: propagator_ptrs(n_propagators)
        !f2py integer(int32), intent(hide), depend(callback_ptrs) :: n_callbacks = len(callback_ptrs)
        !f2py integer(int64), intent(in)  :: callback_ptrs(n_callbacks)
        !f2py integer(int32), intent(out) :: status
        type(c_ptr), intent(out) :: layout_ptr
        type(c_ptr), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: topo_ptr
        integer(int64), intent(in) :: system_size
        integer(int32), intent(in) :: backend_flag
        integer(int32), intent(in) :: n_propagators
        integer(int64), dimension(n_propagators), intent(in) :: propagator_ptrs
        integer(int32), intent(in) :: n_callbacks
        integer(int64), dimension(n_callbacks), intent(in) :: callback_ptrs
        integer(int32), intent(out) :: status

        call negotiate_impl(layout_ptr, split_ptr, topo_ptr, &
                            system_size, backend_flag, &
                            n_propagators, propagator_ptrs, &
                            callback_ptrs, status)
    end subroutine wrapper_negotiate

    subroutine wrapper_create_jaccomm(MPI_COMM, split_ptr, layout_ptr)
        !! Post-negotiate: Build JACCOMM from SUBCOMM rank-0 leaders.
        !! Collective on MPI_COMM.
        !f2py integer(int32), intent(in) :: MPI_COMM
        !f2py integer(int64), intent(in) :: split_ptr
        !f2py integer(int64), intent(in) :: layout_ptr
        integer(int32), intent(in) :: MPI_COMM
        type(c_ptr), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: layout_ptr

        call create_jaccomm_impl(MPI_COMM, split_ptr, layout_ptr)
    end subroutine wrapper_create_jaccomm

    subroutine wrapper_create_rootcomm(MPI_COMM, split_ptr, layout_ptr)
        !! Post-negotiate: Build ROOTCOMM from SUBCOMM rank-0 leaders.
        !! Collective on MPI_COMM.
        !f2py integer(int32), intent(in) :: MPI_COMM
        !f2py integer(int64), intent(in) :: split_ptr
        !f2py integer(int64), intent(in) :: layout_ptr
        integer(int32), intent(in) :: MPI_COMM
        type(c_ptr), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: layout_ptr

        call create_rootcomm_impl(MPI_COMM, split_ptr, layout_ptr)
    end subroutine wrapper_create_rootcomm

    ! -- Topology lifecycle ------------------------------------------

    subroutine wrapper_destroy_topology(topo_ptr_in, topo_ptr_out)
        !! Free the gpu_topology_t allocated by discover_topology.
        !! Returns topo_ptr_out = 0 so Python can zero its handle.
        !f2py integer(int64), intent(in)  :: topo_ptr_in
        !f2py integer(int64), intent(out) :: topo_ptr_out
        integer(int64), intent(in)  :: topo_ptr_in
        integer(int64), intent(out) :: topo_ptr_out
        type(c_ptr) :: local_ptr

        local_ptr = transfer(topo_ptr_in, local_ptr)
        call destroy_topology_impl(local_ptr)
        topo_ptr_out = 0_int64
    end subroutine wrapper_destroy_topology

    ! -- split_info_t creation from existing SUBCOMM -------------------

    subroutine create_split_from_subcomm(split_ptr, MPI_COMM, SUBCOMM, &
                                         worker_id, n_workers)
        !! Bridge helper:  create a split_info_t from an existing SUBCOMM
        !! (already created by Python subcomms).
        !!
        !! SUBCOMM is MPI_Comm_dup'd so that negotiate() can take ownership
        !! of the duplicate without invalidating the Python-side handle.
        !!
        !! Call destroy_split() when done.
        !f2py integer(int64), intent(out) :: split_ptr
        !f2py integer(int32), intent(in)  :: MPI_COMM
        !f2py integer(int32), intent(in)  :: SUBCOMM
        !f2py integer(int32), intent(in)  :: worker_id
        !f2py integer(int32), intent(in)  :: n_workers
        type(c_ptr), intent(out) :: split_ptr
        integer(int32), intent(in) :: MPI_COMM
        integer(int32), intent(in) :: SUBCOMM
        integer(int32), intent(in) :: worker_id
        integer(int32), intent(in) :: n_workers

        type(split_info_t), pointer :: si

        call create_split_from_subcomm_impl(si, MPI_COMM, SUBCOMM, &
                                            worker_id, n_workers)
        split_ptr = c_loc(si)
    end subroutine create_split_from_subcomm

    ! -- split_info_t accessors / lifecycle ---------------------------

    subroutine destroy_split(split_ptr)
        !! Destroy the split_info_t (frees SUBCOMM and JACCOMM if owned).
        !f2py integer(int64), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: split_ptr
        type(split_info_t), pointer :: si

        if (.not. c_associated(split_ptr)) return
        call c_f_pointer(split_ptr, si)
        call si%destroy()
        deallocate (si)
    end subroutine destroy_split

    subroutine get_jaccomm(split_ptr, comm_handle)
        !! Return the JACCOMM handle from split_info_t.
        !f2py integer(int64), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: split_ptr
        integer(int32), intent(out) :: comm_handle
        type(split_info_t), pointer :: si

        call c_f_pointer(split_ptr, si)
        comm_handle = si%JACCOMM
    end subroutine get_jaccomm

    subroutine get_rootcomm(split_ptr, comm_handle)
        !! Return the ROOTCOMM handle from split_info_t.
        !f2py integer(int64), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: split_ptr
        integer(int32), intent(out) :: comm_handle
        type(split_info_t), pointer :: si

        call c_f_pointer(split_ptr, si)
        comm_handle = si%ROOTCOMM
    end subroutine get_rootcomm

    subroutine get_worker_id(split_ptr, wid)
        !! Return the worker_id from split_info_t.
        !f2py integer(int64), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: split_ptr
        integer(int32), intent(out) :: wid
        type(split_info_t), pointer :: si

        call c_f_pointer(split_ptr, si)
        wid = si%worker_id
    end subroutine get_worker_id

    subroutine get_n_workers(split_ptr, nw)
        !! Return the n_workers from split_info_t.
        !f2py integer(int64), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: split_ptr
        integer(int32), intent(out) :: nw
        type(split_info_t), pointer :: si

        call c_f_pointer(split_ptr, si)
        nw = si%n_workers
    end subroutine get_n_workers

    subroutine get_split_subcomm(split_ptr, comm_handle)
        !! Return the SUBCOMM handle from split_info_t.
        !f2py integer(int64), intent(in) :: split_ptr
        type(c_ptr), intent(in) :: split_ptr
        integer(int32), intent(out) :: comm_handle
        type(split_info_t), pointer :: si

        call c_f_pointer(split_ptr, si)
        comm_handle = si%SUBCOMM
    end subroutine get_split_subcomm

    ! -- Diagnostic dump -----------------------------------------------

    subroutine wrapper_dump_comm_info(ci_ptr, phase)
        !! Fortran-native dump of quop_mpi_layout_t diagnostic info.
        !! Reads QUOP_DUMP_COMM_INFO env var, gathers data, rank 0 writes.
        !! COLLECTIVE over MPI_COMM.
        !f2py integer(int64), intent(in) :: ci_ptr
        !f2py character(len=*), intent(in) :: phase
        type(c_ptr), intent(in) :: ci_ptr
        character(len=*), intent(in) :: phase

        type(quop_mpi_layout_t), pointer :: ci

        if (.not. c_associated(ci_ptr)) return
        call c_f_pointer(ci_ptr, ci)
        call ci%dump(phase)
    end subroutine wrapper_dump_comm_info

    subroutine wrapper_get_topology_info(topo_ptr, n_physical_gpus, &
                                         ranks_per_gpu, node_size)
        !! Return key topology fields for Python-side configuration.
        !! NOT collective -- purely local read.
        !f2py integer(int64), intent(in)  :: topo_ptr
        !f2py integer(int32), intent(out) :: n_physical_gpus
        !f2py integer(int32), intent(out) :: ranks_per_gpu
        !f2py integer(int32), intent(out) :: node_size
        type(c_ptr), intent(in) :: topo_ptr
        integer(int32), intent(out) :: n_physical_gpus
        integer(int32), intent(out) :: ranks_per_gpu
        integer(int32), intent(out) :: node_size

        call get_topology_info_impl(topo_ptr, n_physical_gpus, &
                                    ranks_per_gpu, node_size)
    end subroutine wrapper_get_topology_info

    subroutine wrapper_get_layout_topology_info(ci_ptr, n_physical_gpus, &
                                                ranks_per_gpu, node_size)
        !! Return current topology fields from a live quop_mpi_layout_t.
        !! NOT collective -- purely local read.
        !f2py integer(int64), intent(in)  :: ci_ptr
        !f2py integer(int32), intent(out) :: n_physical_gpus
        !f2py integer(int32), intent(out) :: ranks_per_gpu
        !f2py integer(int32), intent(out) :: node_size
        type(c_ptr), intent(in) :: ci_ptr
        integer(int32), intent(out) :: n_physical_gpus
        integer(int32), intent(out) :: ranks_per_gpu
        integer(int32), intent(out) :: node_size

        call get_layout_topology_info_impl(ci_ptr, n_physical_gpus, &
                                           ranks_per_gpu, node_size)
    end subroutine wrapper_get_layout_topology_info

end module comm_info_wrapper
