program numa_demo
    use, intrinsic :: iso_fortran_env, only: int32
    use, intrinsic :: iso_c_binding, only: c_int, c_loc, c_null_char
    use mpi
    use hipfort
    use hipfort_check
    use communicators, only: create_NODECOMM
    implicit none

    integer, parameter :: MAX_CPU_ID = 8191
    integer, parameter :: MAX_NUMA_NODES = 256

    integer(int32) :: ierr, COMM, NODECOMM
    integer(int32) :: comm_rank, comm_size
    integer(int32) :: node_rank, node_size
    integer(int32) :: visible_device_count
    integer(int32) :: cpu_numa_node, overlap_count
    integer(int32) :: sequential_device, recommended_device, current_device
    integer(int32) :: ranks_per_gpu, rank_within_numa, i, d, name_len
    integer(int32), allocatable :: gpu_numa_nodes(:), all_cpu_numa_nodes(:)
    character(len=16), allocatable :: gpu_bdfs(:)
    character(len=512) :: cpu_affinity
    character(len=MPI_MAX_PROCESSOR_NAME) :: hostname
    logical :: have_affinity

    call MPI_Init(ierr)
    COMM = MPI_COMM_WORLD
    call MPI_Comm_rank(COMM, comm_rank, ierr)
    call MPI_Comm_size(COMM, comm_size, ierr)
    call MPI_Get_processor_name(hostname, name_len, ierr)

    call create_NODECOMM(COMM, NODECOMM)
    call MPI_Comm_rank(NODECOMM, node_rank, ierr)
    call MPI_Comm_size(NODECOMM, node_size, ierr)

    ranks_per_gpu = read_env_int('QUOP_RANKS_PER_GPU', 1)

    call hipCheck(hipGetDeviceCount(visible_device_count))
    allocate (gpu_numa_nodes(visible_device_count))
    allocate (gpu_bdfs(visible_device_count))
    do d = 0, visible_device_count - 1
        call get_device_bdf(d, gpu_bdfs(d + 1))
        call read_gpu_numa_node(gpu_bdfs(d + 1), gpu_numa_nodes(d + 1))
    end do

    call read_cpu_affinity(cpu_affinity, have_affinity)
    call find_best_numa_node_for_cpulist(cpu_affinity, cpu_numa_node, overlap_count)

    allocate (all_cpu_numa_nodes(node_size))
    call MPI_Allgather(cpu_numa_node, 1, MPI_INTEGER, all_cpu_numa_nodes, 1, MPI_INTEGER, &
                       NODECOMM, ierr)

    rank_within_numa = 0
    if (cpu_numa_node >= 0) then
        do i = 1, node_rank
            if (all_cpu_numa_nodes(i) == cpu_numa_node) rank_within_numa = rank_within_numa + 1
        end do
    end if

    sequential_device = mod(node_rank / max(1_int32, ranks_per_gpu), visible_device_count)
    call choose_numa_local_device(cpu_numa_node, rank_within_numa, ranks_per_gpu, gpu_numa_nodes, &
                                  sequential_device, recommended_device)

    call hipCheck(hipSetDevice(recommended_device))
    call hipCheck(hipGetDevice(current_device))

    if (comm_rank == 0) then
        write (*, *) "========================================"
        write (*, *) " NUMA-Aware GPU Binding Demo"
        write (*, *) " Running with", comm_size, "MPI processes"
        write (*, *) "========================================"
        write (*, *) "This demo shows a NUMA-aware fallback when no external GPU binding is used."
        write (*, *) ""
    end if

    do i = 0, comm_size - 1
        call MPI_Barrier(COMM, ierr)
        if (comm_rank /= i) cycle

        write (*, '(A)') "----------------------------------------"
        write (*, '(A,I0,A,I0,A,A)') "Rank ", comm_rank, " (node rank ", node_rank, &
            ") on ", trim(hostname)
        write (*, '(A,I0)') "  visible_device_count: ", visible_device_count
        write (*, '(A,A)') "  cpu_affinity: ", trim(cpu_affinity)
        write (*, '(A,I0,A,I0)') "  detected_cpu_numa_node: ", cpu_numa_node, &
            " overlap_cpus: ", overlap_count
        write (*, '(A,I0)') "  rank_within_numa: ", rank_within_numa
        do d = 1, visible_device_count
            write (*, '(A,I0,A,A,A,I0)') "  gpu[", d - 1, "] bdf=", trim(gpu_bdfs(d)), &
                " numa_node=", gpu_numa_nodes(d)
        end do
        write (*, '(A,I0)') "  sequential_device: ", sequential_device
        write (*, '(A,I0)') "  recommended_device: ", recommended_device
        write (*, '(A,I0)') "  hipGetDevice(): ", current_device
        if (.not. have_affinity) then
            write (*, '(A)') "  note: failed to read Cpus_allowed_list; fallback was used"
        else if (cpu_numa_node < 0) then
            write (*, '(A)') "  note: no NUMA node overlap detected; fallback was used"
        else if (recommended_device == sequential_device) then
            write (*, '(A)') "  note: NUMA-aware choice matches sequential assignment"
        else
            write (*, '(A)') "  note: NUMA-aware choice differs from sequential assignment"
        end if
    end do
    call MPI_Barrier(COMM, ierr)

    deallocate (all_cpu_numa_nodes, gpu_numa_nodes, gpu_bdfs)
    call MPI_Comm_free(NODECOMM, ierr)
    call MPI_Finalize(ierr)

contains

    integer(int32) function read_env_int(name, default_value)
        character(len=*), intent(in) :: name
        integer(int32), intent(in) :: default_value

        character(len=64) :: env_val
        integer(int32) :: ios

        call get_environment_variable(name, env_val)
        if (len_trim(env_val) == 0) then
            read_env_int = default_value
            return
        end if

        read (env_val, *, iostat=ios) read_env_int
        if (ios /= 0 .or. read_env_int < 1) read_env_int = default_value
    end function read_env_int

    subroutine get_device_bdf(device_id, bdf)
        integer(int32), intent(in) :: device_id
        character(len=16), intent(out) :: bdf

        character(len=16), target :: raw_bdf

        raw_bdf = ''
        call hipCheck(hipDeviceGetPCIBusId(c_loc(raw_bdf), int(len(raw_bdf), c_int), &
                                           int(device_id, c_int)))
        bdf = clean_c_string(raw_bdf)
        call lowercase_inplace(bdf)
    end subroutine get_device_bdf

    subroutine read_gpu_numa_node(bdf, numa_node)
        character(len=*), intent(in) :: bdf
        integer(int32), intent(out) :: numa_node

        character(len=256) :: path
        logical :: exists
        logical :: ok

        write (path, '(A,A,A)') '/sys/bus/pci/devices/', trim(bdf), '/numa_node'
        inquire (file=trim(path), exist=exists)
        if (.not. exists) then
            numa_node = -1
            return
        end if

        call read_integer_file(trim(path), numa_node, ok)
        if (.not. ok) numa_node = -1
    end subroutine read_gpu_numa_node

    subroutine read_cpu_affinity(cpu_affinity, found)
        character(len=*), intent(out) :: cpu_affinity
        logical, intent(out) :: found

        integer :: unit, ios, colon_pos
        character(len=512) :: line

        cpu_affinity = ''
        found = .false.

        open (newunit=unit, file='/proc/self/status', status='old', action='read', iostat=ios)
        if (ios /= 0) return

        do
            read (unit, '(A)', iostat=ios) line
            if (ios /= 0) exit
            if (index(adjustl(line), 'Cpus_allowed_list:') == 1) then
                colon_pos = index(line, ':')
                if (colon_pos > 0 .and. colon_pos < len(line)) then
                    cpu_affinity = adjustl(line(colon_pos + 1:))
                    found = (len_trim(cpu_affinity) > 0)
                end if
                exit
            end if
        end do

        close (unit)
    end subroutine read_cpu_affinity

    subroutine find_best_numa_node_for_cpulist(cpu_affinity, numa_node, best_overlap)
        character(len=*), intent(in) :: cpu_affinity
        integer(int32), intent(out) :: numa_node
        integer(int32), intent(out) :: best_overlap

        integer(int32) :: node, overlap
        character(len=256) :: path
        character(len=512) :: node_cpulist
        logical :: exists, ok

        numa_node = -1
        best_overlap = 0
        if (len_trim(cpu_affinity) == 0) return

        do node = 0, MAX_NUMA_NODES - 1
            write (path, '(A,I0,A)') '/sys/devices/system/node/node', node, '/cpulist'
            inquire (file=trim(path), exist=exists)
            if (.not. exists) cycle

            call read_first_line(trim(path), node_cpulist, ok)
            if (.not. ok) cycle

            overlap = count_cpu_overlap(cpu_affinity, node_cpulist)
            if (overlap > best_overlap) then
                best_overlap = overlap
                numa_node = node
            end if
        end do
    end subroutine find_best_numa_node_for_cpulist

    subroutine choose_numa_local_device(cpu_numa_node, rank_within_numa, ranks_per_gpu, &
                                        gpu_numa_nodes, sequential_device, recommended_device)
        integer(int32), intent(in) :: cpu_numa_node, rank_within_numa, ranks_per_gpu
        integer(int32), intent(in) :: gpu_numa_nodes(:)
        integer(int32), intent(in) :: sequential_device
        integer(int32), intent(out) :: recommended_device

        integer(int32) :: i, n_matches, slot
        integer(int32), allocatable :: matching_devices(:)

        recommended_device = sequential_device
        if (cpu_numa_node < 0) return

        allocate (matching_devices(size(gpu_numa_nodes)))
        n_matches = 0
        do i = 1, size(gpu_numa_nodes)
            if (gpu_numa_nodes(i) == cpu_numa_node) then
                n_matches = n_matches + 1
                matching_devices(n_matches) = i - 1
            end if
        end do

        if (n_matches > 0) then
            slot = mod(rank_within_numa / max(1_int32, ranks_per_gpu), n_matches) + 1
            recommended_device = matching_devices(slot)
        end if

        deallocate (matching_devices)
    end subroutine choose_numa_local_device

    subroutine read_first_line(path, line, ok)
        character(len=*), intent(in) :: path
        character(len=*), intent(out) :: line
        logical, intent(out) :: ok

        integer :: unit, ios

        line = ''
        ok = .false.

        open (newunit=unit, file=trim(path), status='old', action='read', iostat=ios)
        if (ios /= 0) return

        read (unit, '(A)', iostat=ios) line
        close (unit)
        ok = (ios == 0)
    end subroutine read_first_line

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

    integer(int32) function count_cpu_overlap(list_a, list_b)
        character(len=*), intent(in) :: list_a, list_b

        logical :: mask_a(0:MAX_CPU_ID), mask_b(0:MAX_CPU_ID)

        call parse_cpulist(list_a, mask_a)
        call parse_cpulist(list_b, mask_b)
        count_cpu_overlap = count(mask_a .and. mask_b)
    end function count_cpu_overlap

    subroutine parse_cpulist(cpulist, mask)
        character(len=*), intent(in) :: cpulist
        logical, intent(out) :: mask(0:MAX_CPU_ID)

        integer :: start_pos, end_pos, comma_pos, list_len
        character(len=128) :: token

        mask = .false.
        list_len = len_trim(cpulist)
        if (list_len == 0) return

        start_pos = 1
        do while (start_pos <= list_len)
            comma_pos = index(cpulist(start_pos:list_len), ',')
            if (comma_pos == 0) then
                end_pos = list_len
            else
                end_pos = start_pos + comma_pos - 2
            end if

            token = adjustl(cpulist(start_pos:end_pos))
            call apply_cpu_token(trim(token), mask)

            if (comma_pos == 0) exit
            start_pos = end_pos + 2
        end do
    end subroutine parse_cpulist

    subroutine apply_cpu_token(token, mask)
        character(len=*), intent(in) :: token
        logical, intent(inout) :: mask(0:MAX_CPU_ID)

        integer :: dash_pos, colon_pos, ios
        integer :: start_cpu, end_cpu, stride, cpu
        character(len=128) :: lhs, rhs

        if (len_trim(token) == 0) return

        dash_pos = index(token, '-')
        colon_pos = index(token, ':')

        if (dash_pos == 0) then
            read (token, *, iostat=ios) start_cpu
            if (ios == 0 .and. start_cpu >= 0 .and. start_cpu <= MAX_CPU_ID) then
                mask(start_cpu) = .true.
            end if
            return
        end if

        lhs = token(:dash_pos - 1)
        if (colon_pos == 0) then
            rhs = token(dash_pos + 1:)
            stride = 1
        else
            rhs = token(dash_pos + 1:colon_pos - 1)
            read (token(colon_pos + 1:), *, iostat=ios) stride
            if (ios /= 0 .or. stride < 1) stride = 1
        end if

        read (lhs, *, iostat=ios) start_cpu
        if (ios /= 0) return
        read (rhs, *, iostat=ios) end_cpu
        if (ios /= 0) return

        start_cpu = max(0, start_cpu)
        end_cpu = min(MAX_CPU_ID, end_cpu)
        if (end_cpu < start_cpu) return

        do cpu = start_cpu, end_cpu, stride
            mask(cpu) = .true.
        end do
    end subroutine apply_cpu_token

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

end program numa_demo
