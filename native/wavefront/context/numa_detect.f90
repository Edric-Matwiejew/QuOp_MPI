module numa_detect
    use, intrinsic :: iso_fortran_env, only: int32
    use, intrinsic :: iso_c_binding, only: c_associated, c_int, c_null_ptr, c_ptr
    implicit none

    private

    public :: libnuma_available
    public :: get_task_cpu_numa_node
    public :: assign_gpus_round_robin_by_numa

    interface
        integer(c_int) function c_numa_available() bind(c, name='numa_available')
            import :: c_int
        end function c_numa_available

        integer(c_int) function c_numa_num_configured_nodes() bind(c, name='numa_num_configured_nodes')
            import :: c_int
        end function c_numa_num_configured_nodes

        integer(c_int) function c_numa_num_configured_cpus() bind(c, name='numa_num_configured_cpus')
            import :: c_int
        end function c_numa_num_configured_cpus

        type(c_ptr) function c_numa_allocate_cpumask() bind(c, name='numa_allocate_cpumask')
            import :: c_ptr
        end function c_numa_allocate_cpumask

        subroutine c_numa_bitmask_free(mask) bind(c, name='numa_bitmask_free')
            import :: c_ptr
            type(c_ptr), value :: mask
        end subroutine c_numa_bitmask_free

        integer(c_int) function c_numa_sched_getaffinity(pid, mask) bind(c, name='numa_sched_getaffinity')
            import :: c_int, c_ptr
            integer(c_int), value :: pid
            type(c_ptr), value :: mask
        end function c_numa_sched_getaffinity

        integer(c_int) function c_numa_node_to_cpus(node, mask) bind(c, name='numa_node_to_cpus')
            import :: c_int, c_ptr
            integer(c_int), value :: node
            type(c_ptr), value :: mask
        end function c_numa_node_to_cpus

        integer(c_int) function c_numa_bitmask_isbitset(mask, bit) bind(c, name='numa_bitmask_isbitset')
            import :: c_int, c_ptr
            type(c_ptr), value :: mask
            integer(c_int), value :: bit
        end function c_numa_bitmask_isbitset
    end interface

contains

    logical function libnuma_available()
        libnuma_available = (c_numa_available() >= 0)
    end function libnuma_available

    subroutine get_task_cpu_numa_node(cpu_numa_node, overlap_cpus, success)
        integer(int32), intent(out) :: cpu_numa_node
        integer(int32), intent(out), optional :: overlap_cpus
        logical, intent(out), optional :: success

        integer(c_int) :: n_nodes, n_cpus
        integer(c_int) :: node, cpu, ierr
        integer(int32) :: best_overlap, current_overlap
        type(c_ptr) :: task_mask, node_mask
        logical :: ok

        cpu_numa_node = -1
        best_overlap = 0
        ok = .false.
        task_mask = c_null_ptr
        node_mask = c_null_ptr

        if (.not. libnuma_available()) then
            call set_optional_outputs()
            return
        end if

        n_nodes = c_numa_num_configured_nodes()
        n_cpus = c_numa_num_configured_cpus()
        if (n_nodes <= 0_c_int .or. n_cpus <= 0_c_int) then
            call set_optional_outputs()
            return
        end if

        task_mask = c_numa_allocate_cpumask()
        node_mask = c_numa_allocate_cpumask()
        if (.not. c_associated(task_mask) .or. .not. c_associated(node_mask)) then
            call free_masks(task_mask, node_mask)
            call set_optional_outputs()
            return
        end if

        ! numa_sched_getaffinity() returns the size in bytes of the cpumask
        ! on success (a positive value) and -1 on failure -- it does NOT
        ! return 0 on success.  Treat any negative return as an error.
        ierr = c_numa_sched_getaffinity(0_c_int, task_mask)
        if (ierr < 0_c_int) then
            call free_masks(task_mask, node_mask)
            call set_optional_outputs()
            return
        end if

        do node = 0_c_int, n_nodes - 1_c_int
            ! numa_node_to_cpus() returns 0 on success, -1 on error.
            ierr = c_numa_node_to_cpus(node, node_mask)
            if (ierr < 0_c_int) cycle

            current_overlap = 0
            do cpu = 0_c_int, n_cpus - 1_c_int
                if (c_numa_bitmask_isbitset(task_mask, cpu) /= 0_c_int .and. &
                    c_numa_bitmask_isbitset(node_mask, cpu) /= 0_c_int) then
                    current_overlap = current_overlap + 1
                end if
            end do

            if (current_overlap > best_overlap) then
                best_overlap = current_overlap
                cpu_numa_node = int(node, int32)
                ok = .true.
            end if
        end do

        call free_masks(task_mask, node_mask)

        if (present(overlap_cpus)) overlap_cpus = best_overlap
        if (present(success)) success = ok

    contains

        subroutine free_masks(mask_a, mask_b)
            type(c_ptr), intent(in) :: mask_a, mask_b

            if (c_associated(mask_a)) call c_numa_bitmask_free(mask_a)
            if (c_associated(mask_b)) call c_numa_bitmask_free(mask_b)
        end subroutine free_masks

        subroutine set_optional_outputs()
            if (present(overlap_cpus)) overlap_cpus = 0
            if (present(success)) success = .false.
        end subroutine set_optional_outputs

    end subroutine get_task_cpu_numa_node

    subroutine assign_gpus_round_robin_by_numa(rank_numa_nodes, rank_visible_gpus, gpu_numa_nodes, &
                                               ranks_per_gpu, assigned_physical_indices, is_gpu_rank)
        integer(int32), intent(in) :: rank_numa_nodes(:)
        logical, intent(in) :: rank_visible_gpus(:, :)
        integer(int32), intent(in) :: gpu_numa_nodes(:)
        integer(int32), intent(in) :: ranks_per_gpu
        integer(int32), intent(out) :: assigned_physical_indices(:)
        logical, intent(out) :: is_gpu_rank(:)

        integer(int32), allocatable :: group_nodes(:), gpu_loads(:), next_rank_pos(:), next_gpu_pos(:)
        integer(int32) :: n_gpus, n_groups, group_idx, rank_idx, gpu_idx
        logical :: progress

        assigned_physical_indices = -1
        is_gpu_rank = .false.

        if (size(assigned_physical_indices) /= size(rank_numa_nodes) .or. &
            size(is_gpu_rank) /= size(rank_numa_nodes)) then
            return
        end if

        if (size(rank_visible_gpus, 1) /= size(rank_numa_nodes)) return
        if (size(rank_visible_gpus, 2) /= size(gpu_numa_nodes)) return

        n_gpus = size(gpu_numa_nodes)
        if (n_gpus == 0) return

        call build_numa_groups(rank_numa_nodes, group_nodes)
        n_groups = size(group_nodes)
        if (n_groups == 0) return

        allocate (gpu_loads(n_gpus))
        allocate (next_rank_pos(n_groups))
        allocate (next_gpu_pos(n_groups))
        gpu_loads = 0
        next_rank_pos = 1
        next_gpu_pos = 1

        do
            progress = .false.
            do group_idx = 1, n_groups
                rank_idx = find_next_group_rank(rank_numa_nodes, assigned_physical_indices, group_nodes(group_idx), &
                                                next_rank_pos(group_idx))
                if (rank_idx == 0) cycle

                gpu_idx = find_candidate_gpu(rank_visible_gpus(rank_idx, :), gpu_numa_nodes, gpu_loads, &
                                             max(1_int32, ranks_per_gpu), group_nodes(group_idx), &
                                             next_gpu_pos(group_idx), .true.)
                if (gpu_idx == 0) then
                    gpu_idx = find_candidate_gpu(rank_visible_gpus(rank_idx, :), gpu_numa_nodes, gpu_loads, &
                                                 max(1_int32, ranks_per_gpu), group_nodes(group_idx), &
                                                 next_gpu_pos(group_idx), .false.)
                end if
                if (gpu_idx == 0) cycle

                assigned_physical_indices(rank_idx) = gpu_idx - 1
                is_gpu_rank(rank_idx) = .true.
                gpu_loads(gpu_idx) = gpu_loads(gpu_idx) + 1
                next_rank_pos(group_idx) = rank_idx + 1
                next_gpu_pos(group_idx) = merge(1_int32, gpu_idx + 1, gpu_idx == n_gpus)
                progress = .true.
            end do

            if (.not. progress) exit
        end do

        deallocate (group_nodes, gpu_loads, next_rank_pos, next_gpu_pos)

    contains

        subroutine build_numa_groups(rank_nodes, group_nodes)
            integer(int32), intent(in) :: rank_nodes(:)
            integer(int32), allocatable, intent(out) :: group_nodes(:)

            integer(int32), allocatable :: tmp_nodes(:)
            integer(int32) :: i, j, n_groups
            logical :: seen, have_unknown

            allocate (tmp_nodes(size(rank_nodes)))
            n_groups = 0
            have_unknown = .false.

            do i = 1, size(rank_nodes)
                if (rank_nodes(i) < 0) then
                    have_unknown = .true.
                    cycle
                end if

                seen = .false.
                do j = 1, n_groups
                    if (tmp_nodes(j) == rank_nodes(i)) then
                        seen = .true.
                        exit
                    end if
                end do
                if (seen) cycle

                n_groups = n_groups + 1
                tmp_nodes(n_groups) = rank_nodes(i)
            end do

            call sort_i32_inplace(tmp_nodes, n_groups)
            if (have_unknown) then
                n_groups = n_groups + 1
                tmp_nodes(n_groups) = -1
            end if

            allocate (group_nodes(n_groups))
            if (n_groups > 0) group_nodes = tmp_nodes(:n_groups)
            deallocate (tmp_nodes)
        end subroutine build_numa_groups

        integer(int32) function find_next_group_rank(rank_nodes, assigned_ids, group_node, start_pos)
            integer(int32), intent(in) :: rank_nodes(:), assigned_ids(:), group_node, start_pos
            integer(int32) :: i

            find_next_group_rank = 0
            do i = max(1_int32, start_pos), size(rank_nodes)
                if (assigned_ids(i) /= -1) cycle
                if (group_node == -1) then
                    if (rank_nodes(i) < 0) then
                        find_next_group_rank = i
                        return
                    end if
                else if (rank_nodes(i) == group_node) then
                    find_next_group_rank = i
                    return
                end if
            end do
        end function find_next_group_rank

        integer(int32) function find_candidate_gpu(rank_visibility, gpu_nodes, gpu_loads, capacity, group_node, &
                                                   start_pos, prefer_local)
            logical, intent(in) :: rank_visibility(:)
            integer(int32), intent(in) :: gpu_nodes(:), gpu_loads(:), capacity, group_node, start_pos
            logical, intent(in) :: prefer_local

            integer(int32) :: offset, idx, n_gpus

            n_gpus = size(gpu_nodes)
            find_candidate_gpu = 0
            if (n_gpus == 0) return

            do offset = 0, n_gpus - 1
                idx = 1 + mod(start_pos - 1 + offset, n_gpus)
                if (.not. rank_visibility(idx)) cycle
                if (gpu_loads(idx) >= capacity) cycle
                if (prefer_local .and. group_node >= 0 .and. gpu_nodes(idx) /= group_node) cycle
                find_candidate_gpu = idx
                return
            end do
        end function find_candidate_gpu

        subroutine sort_i32_inplace(array, n)
            integer(int32), intent(inout) :: array(:)
            integer(int32), intent(in) :: n

            integer(int32) :: i, j, tmp

            do i = 2, n
                tmp = array(i)
                j = i - 1
                do while (j >= 1 .and. array(j) > tmp)
                    array(j + 1) = array(j)
                    j = j - 1
                end do
                array(j + 1) = tmp
            end do
        end subroutine sort_i32_inplace

    end subroutine assign_gpus_round_robin_by_numa

end module numa_detect
