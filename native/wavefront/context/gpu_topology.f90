module gpu_topology
    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64, error_unit
    use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_int, c_null_char
    use mpi
    use hipfort
    use hipfort_check
    use numa_detect, only: get_task_cpu_numa_node, assign_gpus_round_robin_by_numa
    implicit none

    private

    public :: visible_gpu_info_t, gpu_topology_t, init_gpu_topology

    type :: visible_gpu_info_t
        integer(int32) :: device_id = -1
        integer(int32) :: physical_gpu_index = -1
        integer(int32) :: numa_node = -1
        character(len=16) :: pci_bus_id = ''
    end type visible_gpu_info_t

    !> Derived type containing all GPU topology information for a node.
    !! This is computed once during context setup and cached for later use.
    type :: gpu_topology_t
        ! Configuration (read once from environment)
        integer(int32) :: ranks_per_gpu !< From QUOP_RANKS_PER_GPU (default: 1)
        character(len=16) :: binding_mode !< 'auto', 'explicit', 'numa', or 'sequential'
        character(len=16) :: binding_strategy = 'none' !< actual strategy fired: 'explicit'/'numa'/'sequential'/'none'

        ! Detected topology (computed once)
        integer(int32) :: visible_device_count !< hipGetDeviceCount result for this rank
        integer(int32) :: n_physical_gpus !< Unique GPUs on this node (from PCI bus IDs)
        integer(int32) :: my_gpu_index !< Which physical GPU I'm bound to (0-based)
        integer(int32) :: assigned_device_id !< Device ID to pass to hipSetDevice
        integer(int32) :: rank_within_gpu !< My rank among those sharing my GPU (0-based)
        integer(int32) :: gpu_slot_ordinal = -1 !< Dense ordinal of active GPU ranks on this node, ordered by physical GPU then NODECOMM rank
        type(visible_gpu_info_t), allocatable :: visible_gpus(:) !< Per-visible-device metadata
        integer(int32) :: cpu_numa_node = -1 !< NUMA node for this rank's CPU affinity, or -1
        integer(int32) :: rank_within_cpu_numa = 0 !< Ranks with lower node_rank on the same CPU NUMA node
        logical :: is_gpu_rank !< Am I assigned a topology-defined GPU rank on this node?

        ! Node info (for reference)
        integer(int32) :: node_rank !< Rank within NODECOMM
        integer(int32) :: node_size !< Size of NODECOMM
        integer(int32) :: devcomm_node_size !< GPU ranks on this node
        character(len=MPI_MAX_PROCESSOR_NAME) :: hostname = '' !< Processor/host name (MPI_Get_processor_name)

        ! Global node topology (populated by discover_topology)
        integer(int32) :: n_nodes = 1 !< Total compute nodes
        integer(int32) :: node_id = 0 !< 0-based sequential node index
    end type gpu_topology_t

contains

    !> Initialize GPU topology by detecting visible devices, gathering PCI bus IDs,
    !! and determining DEVCOMM membership for each rank.
    !!
    !! This routine uses a fixed set of NODECOMM collectives to discover
    !! GPU visibility, CPU NUMA placement, and GPU-rank membership.
    !!
    !! @param[in]  NODECOMM        Node-local communicator (from MPI_Comm_split_type)
    !! @param[out] topology        Populated gpu_topology_t structure
    !! @param[in]  suppress_warnings (optional) If true, suppress informational warnings
    subroutine init_gpu_topology(NODECOMM, topology, suppress_warnings)
        integer(int32), intent(in) :: NODECOMM
        type(gpu_topology_t), intent(out) :: topology
        logical, intent(in), optional :: suppress_warnings

        character(len=64) :: env_val
        character(len=16), allocatable :: physical_gpu_pci_bus_ids(:)
        integer(int32), allocatable :: rank_numa_nodes(:), vis_counts(:), vis_displs(:)
        integer(int32), allocatable :: all_visible_physical(:), gpu_numa_by_physical(:)
        integer(int32), allocatable :: assigned_physical_indices(:)
        logical, allocatable :: rank_visible_physical(:, :), assigned_is_gpu_rank(:)
        integer(int32) :: ierr, i, d, g, r, total_visible_devices, gpu_rank_count
        integer(int32) :: global_rank, hostname_len
        logical :: debug_communicators, emit_warnings
        logical :: env_is_set, env_is_valid, all_ranks_single_visible

        emit_warnings = .true.
        if (present(suppress_warnings)) then
            emit_warnings = .not. suppress_warnings
        end if

        call MPI_Comm_rank(NODECOMM, topology%node_rank, ierr)
        call MPI_Comm_size(NODECOMM, topology%node_size, ierr)
        call MPI_Get_processor_name(topology%hostname, hostname_len, ierr)

        ! ===================================================================
        ! Step 1: Read environment variables (once)
        ! ===================================================================
        topology%ranks_per_gpu = 1
        call get_environment_variable('QUOP_RANKS_PER_GPU', env_val)
        if (len_trim(env_val) > 0) then
            read (env_val, *, iostat=ierr) topology%ranks_per_gpu
            if (ierr /= 0) then
                if (topology%node_rank == 0) then
                    write (error_unit, '(A,A,A)') &
                        'ERROR: QUOP_RANKS_PER_GPU must be a positive integer, got "', &
                        trim(adjustl(env_val)), '".'
                end if
                call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
            end if
        end if

        if (topology%ranks_per_gpu < 1) then
            if (topology%node_rank == 0) then
                write (error_unit, '(A,I0)') &
                    'ERROR: QUOP_RANKS_PER_GPU must be >= 1, got ', topology%ranks_per_gpu
            end if
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        end if

        call get_environment_variable('QUOP_GPU_BINDING_MODE', env_val)
        if (len_trim(env_val) > 0) then
            topology%binding_mode = trim(adjustl(env_val))
        else
            topology%binding_mode = 'auto'
        end if
        call lowercase_inplace(topology%binding_mode)
        if (.not. is_valid_binding_mode(topology%binding_mode)) then
            if (topology%node_rank == 0) then
                write (error_unit, '(A,A,A)') &
                    'ERROR: QUOP_GPU_BINDING_MODE must be one of auto, explicit, numa, sequential; got "', &
                    trim(topology%binding_mode), '".'
            end if
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        end if

        call read_env_flag('QUOP_DEBUG_BACKEND', .false., debug_communicators, &
                           env_is_set, env_is_valid, env_val)
        if (.not. env_is_valid .and. topology%node_rank == 0) then
            write (error_unit, '(A,A,A)') &
                'WARNING: QUOP_DEBUG_BACKEND has unrecognised value "', &
                trim(env_val), '". Using 0.'
        end if

        ! ===================================================================
        ! Step 2: Query visible devices (once per rank)
        ! ===================================================================
        call hipCheck(hipGetDeviceCount(topology%visible_device_count))
        if (topology%visible_device_count == 0) then
            if (topology%node_rank == 0) then
                write (error_unit, '(A)') "ERROR: no GPU devices are visible to any rank on this node."
            end if
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        end if

        ! ===================================================================
        ! Step 3: Discover physical GPUs and node-wide visibility
        ! ===================================================================
        block
            integer(int32) :: s
            character(len=16), target :: dev_pci_id
            character(len=16), allocatable :: my_pci_ids(:), all_gathered(:)
            integer(int32), allocatable :: char_counts(:), char_displs(:)
            integer(int32), allocatable :: my_visible_physical(:), local_gpu_numa_by_physical(:)
            logical :: already_seen

            allocate (topology%visible_gpus(topology%visible_device_count))
            allocate (my_pci_ids(topology%visible_device_count))
            do d = 0, topology%visible_device_count - 1
                call get_device_pci_bus_id(d, dev_pci_id)
                my_pci_ids(d + 1) = dev_pci_id
                topology%visible_gpus(d + 1)%device_id = d
                topology%visible_gpus(d + 1)%pci_bus_id = dev_pci_id
                call read_gpu_numa_node(dev_pci_id, topology%visible_gpus(d + 1)%numa_node)
            end do

            allocate (vis_counts(topology%node_size))
            allocate (vis_displs(topology%node_size))
            call MPI_Allgather(topology%visible_device_count, 1, MPI_INTEGER, &
                               vis_counts, 1, MPI_INTEGER, NODECOMM, ierr)

            vis_displs(1) = 0
            do d = 2, topology%node_size
                vis_displs(d) = vis_displs(d - 1) + vis_counts(d - 1)
            end do
            total_visible_devices = sum(vis_counts)

            allocate (char_counts(topology%node_size))
            allocate (char_displs(topology%node_size))
            char_counts = vis_counts * 16
            char_displs = vis_displs * 16

            allocate (all_gathered(total_visible_devices))
            call MPI_Gatherv(my_pci_ids, topology%visible_device_count * 16, MPI_CHARACTER, &
                             all_gathered, char_counts, char_displs, MPI_CHARACTER, &
                             0, NODECOMM, ierr)

            if (topology%node_rank == 0) then
                allocate (physical_gpu_pci_bus_ids(total_visible_devices))
                topology%n_physical_gpus = 0
                do d = 1, total_visible_devices
                    already_seen = .false.
                    do s = 1, topology%n_physical_gpus
                        if (all_gathered(d) == physical_gpu_pci_bus_ids(s)) then
                            already_seen = .true.
                            exit
                        end if
                    end do
                    if (.not. already_seen) then
                        topology%n_physical_gpus = topology%n_physical_gpus + 1
                        physical_gpu_pci_bus_ids(topology%n_physical_gpus) = all_gathered(d)
                    end if
                end do
            else
                allocate (physical_gpu_pci_bus_ids(1))
            end if

            call MPI_Bcast(topology%n_physical_gpus, 1, MPI_INTEGER, 0, NODECOMM, ierr)
            if (topology%node_rank /= 0) then
                deallocate (physical_gpu_pci_bus_ids)
                allocate (physical_gpu_pci_bus_ids(topology%n_physical_gpus))
            end if
            call MPI_Bcast(physical_gpu_pci_bus_ids, topology%n_physical_gpus * 16, MPI_CHARACTER, &
                           0, NODECOMM, ierr)

            allocate (my_visible_physical(topology%visible_device_count))
            do d = 1, topology%visible_device_count
                topology%visible_gpus(d)%physical_gpu_index = find_pci_bus_id_index( &
                                              topology%visible_gpus(d)%pci_bus_id, &
                                              physical_gpu_pci_bus_ids, topology%n_physical_gpus)
                my_visible_physical(d) = topology%visible_gpus(d)%physical_gpu_index
            end do

            allocate (all_visible_physical(total_visible_devices))
            call MPI_Allgatherv(my_visible_physical, topology%visible_device_count, MPI_INTEGER, &
                                all_visible_physical, vis_counts, vis_displs, MPI_INTEGER, NODECOMM, ierr)

            allocate (rank_visible_physical(topology%node_size, topology%n_physical_gpus))
            rank_visible_physical = .false.
            do r = 1, topology%node_size
                do d = vis_displs(r) + 1, vis_displs(r) + vis_counts(r)
                    if (all_visible_physical(d) >= 0) then
                        rank_visible_physical(r, all_visible_physical(d) + 1) = .true.
                    end if
                end do
            end do

            allocate (local_gpu_numa_by_physical(topology%n_physical_gpus))
            local_gpu_numa_by_physical = -1
            do d = 1, topology%visible_device_count
                g = topology%visible_gpus(d)%physical_gpu_index
                if (g >= 0) then
                    local_gpu_numa_by_physical(g + 1) = max(local_gpu_numa_by_physical(g + 1), &
                                                            topology%visible_gpus(d)%numa_node)
                end if
            end do

            allocate (gpu_numa_by_physical(topology%n_physical_gpus))
            call MPI_Allreduce(local_gpu_numa_by_physical, gpu_numa_by_physical, topology%n_physical_gpus, &
                               MPI_INTEGER, MPI_MAX, NODECOMM, ierr)

            deallocate (my_pci_ids, all_gathered, char_counts, char_displs)
            deallocate (my_visible_physical, local_gpu_numa_by_physical)
        end block

        ! ===================================================================
        ! Step 4: Gather CPU NUMA topology and validate binding mode
        ! ===================================================================
        call get_task_cpu_numa_node(topology%cpu_numa_node)
        allocate (rank_numa_nodes(topology%node_size))
        call MPI_Allgather(topology%cpu_numa_node, 1, MPI_INTEGER, &
                           rank_numa_nodes, 1, MPI_INTEGER, NODECOMM, ierr)

        topology%rank_within_cpu_numa = 0
        if (topology%cpu_numa_node >= 0) then
            do i = 1, topology%node_rank
                if (rank_numa_nodes(i) == topology%cpu_numa_node) then
                    topology%rank_within_cpu_numa = topology%rank_within_cpu_numa + 1
                end if
            end do
        end if

        all_ranks_single_visible = all_ranks_see_exactly_one_gpu(rank_visible_physical)
        if (topology%binding_mode == 'explicit' .and. .not. all_ranks_single_visible) then
            if (topology%node_rank == 0) then
                write (error_unit, '(A)') &
                    'ERROR: QUOP_GPU_BINDING_MODE=explicit requires each rank to see exactly one GPU.'
                write (error_unit, '(A)') &
                    'ERROR: use --gpu-bind=closest or set QUOP_GPU_BINDING_MODE=numa/sequential.'
            end if
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        end if

        ! ===================================================================
        ! Step 5: Determine DEVCOMM membership and device assignment
        ! ===================================================================
        allocate (assigned_physical_indices(topology%node_size))
        allocate (assigned_is_gpu_rank(topology%node_size))
        call compute_gpu_assignment(topology, rank_numa_nodes, rank_visible_physical, gpu_numa_by_physical, &
                                    assigned_physical_indices, assigned_is_gpu_rank)

        if (topology%my_gpu_index >= 0 .and. topology%assigned_device_id < 0) then
            write (error_unit, '(A,I0,A,A,A,I0,A)') &
                'ERROR: NODECOMM rank ', topology%node_rank, ' on ', trim(topology%hostname), &
                ' was assigned invisible physical GPU ', topology%my_gpu_index, '.'
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        end if

        topology%rank_within_gpu = 0
        if (topology%my_gpu_index >= 0) then
            do i = 1, topology%node_rank
                if (assigned_physical_indices(i) == topology%my_gpu_index) then
                    topology%rank_within_gpu = topology%rank_within_gpu + 1
                end if
            end do
        end if

        topology%gpu_slot_ordinal = -1
        if (topology%is_gpu_rank) then
            block
                integer(int32), allocatable :: active_gpu_counts(:)

                allocate (active_gpu_counts(topology%n_physical_gpus))
                active_gpu_counts = 0

                do i = 1, topology%node_size
                    if (.not. assigned_is_gpu_rank(i)) cycle
                    g = assigned_physical_indices(i)
                    if (g >= 0) then
                        active_gpu_counts(g + 1) = active_gpu_counts(g + 1) + 1
                    end if
                end do

                topology%gpu_slot_ordinal = topology%rank_within_gpu
                do i = 1, topology%my_gpu_index
                    topology%gpu_slot_ordinal = topology%gpu_slot_ordinal + active_gpu_counts(i)
                end do

                deallocate (active_gpu_counts)
            end block
        end if

        ! ===================================================================
        ! Step 6: Count GPU ranks on node (single reduction)
        ! ===================================================================
        if (topology%is_gpu_rank) then
            gpu_rank_count = 1
        else
            gpu_rank_count = 0
        end if
        call MPI_Allreduce(gpu_rank_count, topology%devcomm_node_size, 1, &
                           MPI_INTEGER, MPI_SUM, NODECOMM, ierr)

        ! ===================================================================
        ! Step 7: Debug output if requested
        ! ===================================================================
        if (debug_communicators) then
            call MPI_Comm_rank(MPI_COMM_WORLD, global_rank, ierr)
            write (error_unit, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,L1,A,L1,A,A,A,A)') &
                "DEBUG [Rank ", global_rank, &
                "]: visible_devices=", topology%visible_device_count, &
                ", n_physical_gpus=", topology%n_physical_gpus, &
                ", my_gpu_index=", topology%my_gpu_index, &
                ", gpu_slot_ordinal=", topology%gpu_slot_ordinal, &
                ", cpu_numa=", topology%cpu_numa_node, &
                ", assigned_device=", topology%assigned_device_id, &
                ", is_gpu_rank=", topology%is_gpu_rank, &
                ", mode=", trim(topology%binding_mode), &
                ", strategy=", trim(topology%binding_strategy)
        end if

        if ((topology%binding_mode == 'auto' .or. topology%binding_mode == 'numa') .and. &
            .not. (all_ranks_single_visible .and. topology%n_physical_gpus > 1) .and. &
            numa_info_available(rank_numa_nodes, gpu_numa_by_physical)) then
            if (emit_warnings .and. topology%node_rank == 0) then
                block
                    integer(int32) :: gpu_numa
                    logical :: has_any_rank, has_local_rank

                    do g = 0, topology%n_physical_gpus - 1
                        gpu_numa = gpu_numa_by_physical(g + 1)
                        if (gpu_numa < 0) cycle

                        has_any_rank = .false.
                        has_local_rank = .false.
                        do r = 1, topology%node_size
                            if (assigned_is_gpu_rank(r) .and. assigned_physical_indices(r) == g) then
                                has_any_rank = .true.
                                if (rank_numa_nodes(r) == gpu_numa) then
                                    has_local_rank = .true.
                                    exit
                                end if
                            end if
                        end do

                        if (has_any_rank .and. .not. has_local_rank) then
                            write (error_unit, '(A,A,A,A,A)') &
                                "WARNING: GPU ", trim(physical_gpu_pci_bus_ids(g + 1)), &
                                " on ", trim(topology%hostname), &
                                " is not assigned to any NUMA-local rank; consider --gpu-bind=closest."
                        end if
                    end do
                end block
            end if
        end if

        deallocate (rank_numa_nodes, vis_counts, vis_displs, all_visible_physical)
        deallocate (gpu_numa_by_physical, assigned_physical_indices, physical_gpu_pci_bus_ids)
        deallocate (rank_visible_physical, assigned_is_gpu_rank)

    end subroutine init_gpu_topology

    !> Compute GPU assignment in physical-GPU space, then map the local rank
    !! back to a rank-local HIP device id.
    subroutine compute_gpu_assignment(topology, rank_numa_nodes, rank_visible_physical, gpu_numa_by_physical, &
                                      assigned_physical_indices, assigned_is_gpu_rank)
        type(gpu_topology_t), intent(inout) :: topology
        integer(int32), intent(in) :: rank_numa_nodes(:), gpu_numa_by_physical(:)
        logical, intent(in) :: rank_visible_physical(:, :)
        integer(int32), intent(out) :: assigned_physical_indices(:)
        logical, intent(out) :: assigned_is_gpu_rank(:)

        integer(int32) :: r
        logical :: external_binding_detected

        assigned_physical_indices = -1
        assigned_is_gpu_rank = .false.
        topology%my_gpu_index = -1
        topology%assigned_device_id = -1
        topology%is_gpu_rank = .false.

        external_binding_detected = all_ranks_see_exactly_one_gpu(rank_visible_physical) .and. &
                                    topology%n_physical_gpus > 1

        select case (trim(topology%binding_mode))
        case ('explicit')
            topology%binding_strategy = 'explicit'
            call apply_explicit_assignment()

        case ('sequential')
            topology%binding_strategy = 'sequential'
            call apply_sequential_assignment()

        case ('numa')
            if (external_binding_detected) then
                topology%binding_strategy = 'explicit'
                call apply_explicit_assignment()
            else if (numa_info_available(rank_numa_nodes, gpu_numa_by_physical)) then
                topology%binding_strategy = 'numa'
                call apply_numa_assignment()
            else
                topology%binding_strategy = 'sequential'
                call apply_sequential_assignment()
            end if

        case default ! 'auto'
            if (external_binding_detected) then
                topology%binding_strategy = 'explicit'
                call apply_explicit_assignment()
            else if (numa_info_available(rank_numa_nodes, gpu_numa_by_physical)) then
                topology%binding_strategy = 'numa'
                call apply_numa_assignment()
            else
                topology%binding_strategy = 'sequential'
                call apply_sequential_assignment()
            end if
        end select

    contains

        subroutine apply_explicit_assignment()
            integer(int32) :: g, prior_matches, i

            do r = 1, size(rank_numa_nodes)
                g = get_only_visible_physical_gpu(rank_visible_physical(r, :))
                assigned_physical_indices(r) = g

                prior_matches = 0
                if (g >= 0) then
                    do i = 1, r - 1
                        if (assigned_physical_indices(i) == g) prior_matches = prior_matches + 1
                    end do
                end if
                assigned_is_gpu_rank(r) = (g >= 0 .and. prior_matches < topology%ranks_per_gpu)
            end do

            call apply_local_assignment()
        end subroutine apply_explicit_assignment

        subroutine apply_sequential_assignment()
            integer(int32), allocatable :: gpu_loads(:)
            integer(int32) :: g, target_gpu

            allocate (gpu_loads(topology%n_physical_gpus))
            gpu_loads = 0

            do r = 1, size(rank_numa_nodes)
                target_gpu = mod((r - 1) / topology%ranks_per_gpu, topology%n_physical_gpus)
                g = find_visible_gpu_with_capacity(rank_visible_physical(r, :), target_gpu, gpu_loads)
                if (g >= 0) then
                    assigned_physical_indices(r) = g
                    assigned_is_gpu_rank(r) = .true.
                    gpu_loads(g + 1) = gpu_loads(g + 1) + 1
                else
                    assigned_physical_indices(r) = find_visible_gpu_from_target(rank_visible_physical(r, :), target_gpu)
                end if
            end do

            deallocate (gpu_loads)
            call apply_local_assignment()
        end subroutine apply_sequential_assignment

        subroutine apply_numa_assignment()
            integer(int32) :: g, target_gpu

            call assign_gpus_round_robin_by_numa(rank_numa_nodes, rank_visible_physical, gpu_numa_by_physical, &
                                                 topology%ranks_per_gpu, assigned_physical_indices, &
                                                 assigned_is_gpu_rank)

            do r = 1, size(rank_numa_nodes)
                if (assigned_physical_indices(r) >= 0) cycle
                target_gpu = mod((r - 1) / topology%ranks_per_gpu, topology%n_physical_gpus)
                g = find_visible_gpu_from_target(rank_visible_physical(r, :), target_gpu)
                assigned_physical_indices(r) = g
            end do

            call apply_local_assignment()
        end subroutine apply_numa_assignment

        subroutine apply_local_assignment()
            topology%my_gpu_index = assigned_physical_indices(topology%node_rank + 1)
            topology%is_gpu_rank = assigned_is_gpu_rank(topology%node_rank + 1)
            if (topology%my_gpu_index >= 0) then
                topology%assigned_device_id = find_local_device_id_for_physical(topology, topology%my_gpu_index)
            else
                topology%assigned_device_id = -1
            end if
        end subroutine apply_local_assignment

        integer(int32) function find_visible_gpu_with_capacity(rank_visibility, start_gpu, gpu_loads)
            logical, intent(in) :: rank_visibility(:)
            integer(int32), intent(in) :: start_gpu, gpu_loads(:)
            integer(int32) :: offset, g

            find_visible_gpu_with_capacity = -1
            do offset = 0, topology%n_physical_gpus - 1
                g = mod(start_gpu + offset, topology%n_physical_gpus)
                if (.not. rank_visibility(g + 1)) cycle
                if (gpu_loads(g + 1) >= topology%ranks_per_gpu) cycle
                find_visible_gpu_with_capacity = g
                return
            end do
        end function find_visible_gpu_with_capacity

        integer(int32) function find_visible_gpu_from_target(rank_visibility, start_gpu)
            logical, intent(in) :: rank_visibility(:)
            integer(int32), intent(in) :: start_gpu
            integer(int32) :: offset, g

            find_visible_gpu_from_target = -1
            do offset = 0, topology%n_physical_gpus - 1
                g = mod(start_gpu + offset, topology%n_physical_gpus)
                if (.not. rank_visibility(g + 1)) cycle
                find_visible_gpu_from_target = g
                return
            end do
        end function find_visible_gpu_from_target

    end subroutine compute_gpu_assignment

    subroutine get_device_pci_bus_id(device_id, pci_bus_id)
        integer(int32), intent(in) :: device_id
        character(len=16), intent(out) :: pci_bus_id

        character(len=16), target :: raw_pci_bus_id

        raw_pci_bus_id = ''
        call hipCheck(hipDeviceGetPCIBusId(c_loc(raw_pci_bus_id), &
                                           int(len(raw_pci_bus_id), c_int), &
                                           int(device_id, c_int)))
        pci_bus_id = clean_c_string(raw_pci_bus_id)
        call lowercase_inplace(pci_bus_id)
    end subroutine get_device_pci_bus_id

    subroutine read_gpu_numa_node(pci_bus_id, numa_node)
        character(len=*), intent(in) :: pci_bus_id
        integer(int32), intent(out) :: numa_node

        character(len=256) :: path
        logical :: exists
        logical :: ok

        write (path, '(A,A,A)') '/sys/bus/pci/devices/', trim(pci_bus_id), '/numa_node'
        inquire (file=trim(path), exist=exists)
        if (.not. exists) then
            numa_node = -1
            return
        end if

        call read_integer_file(trim(path), numa_node, ok)
        if (.not. ok) numa_node = -1
    end subroutine read_gpu_numa_node

    subroutine read_integer_file(path, value, ok)
        character(len=*), intent(in) :: path
        integer(int32), intent(out) :: value
        logical, intent(out) :: ok

        integer :: unit, ios

        value = -1
        ok = .false.

        open (newunit=unit, file=trim(path), status='old', action='read', iostat=ios)
        if (ios /= 0) return

        read (unit, *, iostat=ios) value
        close (unit)
        ok = (ios == 0)
    end subroutine read_integer_file

    integer(int32) function find_pci_bus_id_index(target_pci_bus_id, pci_bus_ids, n_ids)
        character(len=*), intent(in) :: target_pci_bus_id
        character(len=16), intent(in) :: pci_bus_ids(:)
        integer(int32), intent(in) :: n_ids

        integer(int32) :: i

        find_pci_bus_id_index = -1
        do i = 1, n_ids
            if (target_pci_bus_id == pci_bus_ids(i)) then
                find_pci_bus_id_index = i - 1
                return
            end if
        end do
    end function find_pci_bus_id_index

    character(len=16) function clean_c_string(raw)
        character(len=*), intent(in) :: raw

        integer :: i

        clean_c_string = ' '
        do i = 1, min(len(raw), len(clean_c_string))
            if (raw(i:i) == c_null_char) exit
            clean_c_string(i:i) = raw(i:i)
        end do
        clean_c_string = adjustl(clean_c_string)
    end function clean_c_string

    subroutine lowercase_inplace(str)
        character(len=*), intent(inout) :: str

        integer :: i, code

        do i = 1, len_trim(str)
            code = iachar(str(i:i))
            if (code >= iachar('A') .and. code <= iachar('Z')) then
                str(i:i) = achar(code + 32)
            end if
        end do
    end subroutine lowercase_inplace

    subroutine read_env_flag(name, default_value, value, env_is_set, env_is_valid, raw_value)
        character(len=*), intent(in) :: name
        logical, intent(in) :: default_value
        logical, intent(out) :: value
        logical, intent(out) :: env_is_set
        logical, intent(out) :: env_is_valid
        character(len=*), intent(out) :: raw_value

        raw_value = ''
        call get_environment_variable(name, raw_value)
        raw_value = trim(adjustl(raw_value))
        env_is_set = len_trim(raw_value) > 0

        if (.not. env_is_set) then
            value = default_value
            env_is_valid = .true.
            return
        end if

        call lowercase_inplace(raw_value)
        select case (trim(raw_value))
        case ('1', 'true', 'yes', 'on')
            value = .true.
            env_is_valid = .true.
        case ('0', 'false', 'no', 'off')
            value = .false.
            env_is_valid = .true.
        case default
            value = default_value
            env_is_valid = .false.
        end select
    end subroutine read_env_flag

    logical function is_valid_binding_mode(binding_mode)
        character(len=*), intent(in) :: binding_mode

        select case (trim(binding_mode))
        case ('auto', 'explicit', 'numa', 'sequential')
            is_valid_binding_mode = .true.
        case default
            is_valid_binding_mode = .false.
        end select
    end function is_valid_binding_mode

    !> Check if NUMA topology information is available.
    !! Returns true if at least one rank has a known CPU NUMA node and at least
    !! one physical GPU has a known NUMA node, so NUMA-aware assignment is
    !! meaningful.
    logical function numa_info_available(rank_numa_nodes, gpu_numa_nodes)
        integer(int32), intent(in) :: rank_numa_nodes(:), gpu_numa_nodes(:)

        integer(int32) :: i
        logical :: have_cpu_numa, have_gpu_numa

        numa_info_available = .false.

        have_cpu_numa = .false.
        do i = 1, size(rank_numa_nodes)
            if (rank_numa_nodes(i) >= 0) then
                have_cpu_numa = .true.
                exit
            end if
        end do
        if (.not. have_cpu_numa) return

        have_gpu_numa = .false.
        do i = 1, size(gpu_numa_nodes)
            if (gpu_numa_nodes(i) >= 0) then
                have_gpu_numa = .true.
                exit
            end if
        end do

        numa_info_available = have_gpu_numa
    end function numa_info_available

    logical function all_ranks_see_exactly_one_gpu(rank_visible_physical)
        logical, intent(in) :: rank_visible_physical(:, :)

        integer(int32) :: r

        all_ranks_see_exactly_one_gpu = .true.
        do r = 1, size(rank_visible_physical, 1)
            if (count(rank_visible_physical(r, :)) /= 1) then
                all_ranks_see_exactly_one_gpu = .false.
                return
            end if
        end do
    end function all_ranks_see_exactly_one_gpu

    integer(int32) function get_only_visible_physical_gpu(rank_visibility)
        logical, intent(in) :: rank_visibility(:)

        integer(int32) :: i

        get_only_visible_physical_gpu = -1
        if (count(rank_visibility) /= 1) return

        do i = 1, size(rank_visibility)
            if (rank_visibility(i)) then
                get_only_visible_physical_gpu = i - 1
                return
            end if
        end do
    end function get_only_visible_physical_gpu

    integer(int32) function find_local_device_id_for_physical(topology, physical_gpu)
        type(gpu_topology_t), intent(in) :: topology
        integer(int32), intent(in) :: physical_gpu

        integer(int32) :: i

        find_local_device_id_for_physical = -1
        if (.not. allocated(topology%visible_gpus)) return

        do i = 1, topology%visible_device_count
            if (topology%visible_gpus(i)%physical_gpu_index == physical_gpu) then
                find_local_device_id_for_physical = topology%visible_gpus(i)%device_id
                return
            end if
        end do
    end function find_local_device_id_for_physical

end module gpu_topology
