!> GPU Transfer Module
!>
!> This module provides abstracted GPU memory transfer operations with two
!> implementation paths:
!>
!> 1. GPU-aware MPI (when compiled with QUOP_GPU_AWARE_MPI=ON):
!>    - Uses MPI directly with device pointers
!>    - Requires a GPU-aware MPI implementation (e.g., Cray MPICH with GTL)
!>    - Highest performance path
!>
!> 2. Host-staged fallback (default):
!>    - Copies device memory to host, uses regular MPI, copies back to device
!>    - Does NOT require GPU-aware MPI
!>
!> Preprocessor options:
!>   QUOP_GPU_AWARE_MPI  - Use GPU-aware MPI
!>
module gpu_transfer

    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64, error_unit
    use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_size_t, c_int, c_f_pointer, c_null_ptr
    use MPI
    use hipfort
    use hipfort_check

    implicit none

    private

    public :: gpu_allgatherv_dtoh
    public :: gpu_allscatterv_htod

contains

#ifdef QUOP_GPU_AWARE_MPI
    subroutine mpi_irecv_from_ptr(buf_ptr, count, mpi_type, element_size, source, tag, comm, request, ierr)
        type(c_ptr), value, intent(in) :: buf_ptr
        integer(int32), intent(in) :: count, source, tag, comm
        integer, intent(in) :: mpi_type
        integer(c_size_t), intent(in) :: element_size
        integer, intent(out) :: request
        integer(c_int), intent(out) :: ierr

        complex(real64), pointer :: zseg(:)
        real(real64), pointer :: rseg(:)
        character, pointer :: bseg(:)
        integer :: n_bytes

        select case (mpi_type)
        case (MPI_DOUBLE_COMPLEX)
            call c_f_pointer(buf_ptr, zseg, [int(count)])
            call MPI_Irecv(zseg, count, mpi_type, source, tag, comm, request, ierr)
        case (MPI_DOUBLE)
            call c_f_pointer(buf_ptr, rseg, [int(count)])
            call MPI_Irecv(rseg, count, mpi_type, source, tag, comm, request, ierr)
        case default
            n_bytes = int(count) * int(element_size)
            call c_f_pointer(buf_ptr, bseg, [n_bytes])
            call MPI_Irecv(bseg, count, mpi_type, source, tag, comm, request, ierr)
        end select
    end subroutine mpi_irecv_from_ptr

    subroutine mpi_isend_from_ptr(buf_ptr, count, mpi_type, element_size, dest, tag, comm, request, ierr)
        type(c_ptr), value, intent(in) :: buf_ptr
        integer(int32), intent(in) :: count, dest, tag, comm
        integer, intent(in) :: mpi_type
        integer(c_size_t), intent(in) :: element_size
        integer, intent(out) :: request
        integer(c_int), intent(out) :: ierr

        complex(real64), pointer :: zseg(:)
        real(real64), pointer :: rseg(:)
        character, pointer :: bseg(:)
        integer :: n_bytes

        select case (mpi_type)
        case (MPI_DOUBLE_COMPLEX)
            call c_f_pointer(buf_ptr, zseg, [int(count)])
            call MPI_Isend(zseg, count, mpi_type, dest, tag, comm, request, ierr)
        case (MPI_DOUBLE)
            call c_f_pointer(buf_ptr, rseg, [int(count)])
            call MPI_Isend(rseg, count, mpi_type, dest, tag, comm, request, ierr)
        case default
            n_bytes = int(count) * int(element_size)
            call c_f_pointer(buf_ptr, bseg, [n_bytes])
            call MPI_Isend(bseg, count, mpi_type, dest, tag, comm, request, ierr)
        end select
    end subroutine mpi_isend_from_ptr
#endif

    type(c_ptr) function ptr_with_byte_offset(base_ptr, byte_offset)
        type(c_ptr), value, intent(in) :: base_ptr
        integer(c_size_t), intent(in) :: byte_offset

        ptr_with_byte_offset = transfer(transfer(base_ptr, 1_c_size_t) + byte_offset, c_null_ptr)
    end function ptr_with_byte_offset

    subroutine get_mpi_element_size(mpi_type, element_size)
        integer, intent(in) :: mpi_type
        integer(c_size_t), intent(out) :: element_size

        integer(c_int) :: ierr
        integer :: type_size

        call MPI_Type_size(mpi_type, type_size, ierr)
        element_size = int(type_size, c_size_t)
    end subroutine get_mpi_element_size

    subroutine create_byte_element_type(element_size, elem_type)
        integer(c_size_t), intent(in) :: element_size
        integer, intent(out) :: elem_type

        integer(c_int) :: ierr

        call MPI_Type_contiguous(int(element_size), MPI_BYTE, elem_type, ierr)
        call MPI_Type_commit(elem_type, ierr)
    end subroutine create_byte_element_type

    subroutine sync_device_if_nonempty(count)
        integer(int64), intent(in) :: count

        if (count > 0) then
            call hipCheck(hipDeviceSynchronize())
        end if
    end subroutine sync_device_if_nonempty

    subroutine copy_local_overlap(src_ptr, src_start, dst_ptr, dst_start, overlap_start, &
                                  overlap_end, element_size, copy_kind)
        type(c_ptr), value, intent(in) :: src_ptr, dst_ptr
        integer(int64), intent(in) :: src_start, dst_start, overlap_start, overlap_end
        integer(c_size_t), intent(in) :: element_size
        integer, intent(in) :: copy_kind

        integer(int64) :: copy_count
        integer(c_size_t) :: src_byte_offset, dst_byte_offset, byte_count

        if (overlap_end <= overlap_start) return

        copy_count = overlap_end - overlap_start
        byte_count = int(copy_count, c_size_t) * element_size
        src_byte_offset = int(overlap_start - src_start, c_size_t) * element_size
        dst_byte_offset = int(overlap_start - dst_start, c_size_t) * element_size

        call hipCheck(hipMemcpy( &
                      ptr_with_byte_offset(dst_ptr, dst_byte_offset), &
                      ptr_with_byte_offset(src_ptr, src_byte_offset), &
                      byte_count, copy_kind))
    end subroutine copy_local_overlap

    subroutine build_transfer_schedule(src_counts, dst_counts, src_displs, dst_displs, &
                                       rank, comm, send_cnts, send_displs, recv_cnts, recv_displs)
        integer(int64), intent(in) :: src_counts(:), dst_counts(:)
        integer(int64), intent(in) :: src_displs(:), dst_displs(:)
        integer, intent(in) :: rank
        integer(int32), intent(in) :: comm
        integer, intent(out) :: send_cnts(:), send_displs(:), recv_cnts(:), recv_displs(:)

        integer :: i
        integer(int64) :: src_start, src_end, dst_start, dst_end
        integer(int64) :: overlap_start, overlap_end

        src_start = src_displs(rank + 1)
        src_end = src_start + src_counts(rank + 1)
        do i = 1, size(send_cnts)
            dst_start = dst_displs(i)
            dst_end = dst_start + dst_counts(i)
            overlap_start = max(src_start, dst_start)
            overlap_end = min(src_end, dst_end)
            if (overlap_end > overlap_start) then
                send_cnts(i) = safe_int32(overlap_end - overlap_start, comm)
                send_displs(i) = safe_int32(overlap_start - src_start, comm)
            else
                send_cnts(i) = 0
                send_displs(i) = 0
            end if
        end do

        dst_start = dst_displs(rank + 1)
        dst_end = dst_start + dst_counts(rank + 1)
        do i = 1, size(recv_cnts)
            src_start = src_displs(i)
            src_end = src_start + src_counts(i)
            overlap_start = max(src_start, dst_start)
            overlap_end = min(src_end, dst_end)
            if (overlap_end > overlap_start) then
                recv_cnts(i) = safe_int32(overlap_end - overlap_start, comm)
                recv_displs(i) = safe_int32(overlap_start - dst_start, comm)
            else
                recv_cnts(i) = 0
                recv_displs(i) = 0
            end if
        end do
    end subroutine build_transfer_schedule

#ifdef QUOP_GPU_AWARE_MPI
    subroutine exchange_nonlocal_segments(src_ptr, dst_ptr, send_cnts, send_displs, recv_cnts, &
                                          recv_displs, mpi_type, element_size, rank, comm)
        type(c_ptr), value, intent(in) :: src_ptr, dst_ptr
        integer, intent(in) :: send_cnts(:), send_displs(:), recv_cnts(:), recv_displs(:)
        integer, intent(in) :: mpi_type, rank
        integer(c_size_t), intent(in) :: element_size
        integer(int32), intent(in) :: comm

        integer(c_int) :: ierr
        integer :: i, partner, n_req
        integer, allocatable :: requests(:)
        type(c_ptr) :: ptr_tmp

        allocate (requests(2 * (size(send_cnts) - 1)))

        n_req = 0
        do i = 1, size(recv_cnts)
            partner = i - 1
            if (partner == rank .or. recv_cnts(i) == 0) cycle
            n_req = n_req + 1
            ptr_tmp = ptr_with_byte_offset(dst_ptr, int(recv_displs(i), c_size_t) * element_size)
            call mpi_irecv_from_ptr(ptr_tmp, recv_cnts(i), mpi_type, element_size, &
                                    partner, rank, comm, requests(n_req), ierr)
        end do

        do i = 1, size(send_cnts)
            partner = i - 1
            if (partner == rank .or. send_cnts(i) == 0) cycle
            n_req = n_req + 1
            ptr_tmp = ptr_with_byte_offset(src_ptr, int(send_displs(i), c_size_t) * element_size)
            call mpi_isend_from_ptr(ptr_tmp, send_cnts(i), mpi_type, element_size, &
                                    partner, partner, comm, requests(n_req), ierr)
        end do

        if (n_req > 0) then
            call MPI_Waitall(n_req, requests(1:n_req), MPI_STATUSES_IGNORE, ierr)
        end if

        deallocate (requests)
    end subroutine exchange_nonlocal_segments
#endif

    !> Gather data from device memory across ranks to host memory on all ranks
    subroutine gpu_allgatherv_dtoh(dev_counts, host_counts, dev_displs, &
                                   host_displs, dev_ptr, host_ptr, mpi_type, &
                                   NODECOMM)
        integer(int64), dimension(:), target, intent(in) :: dev_counts, host_counts
        integer(int64), dimension(:), target, intent(in) :: dev_displs, host_displs
        type(c_ptr), value, intent(in) :: dev_ptr
        type(c_ptr), value, intent(in) :: host_ptr
        integer, intent(in) :: mpi_type
        integer(int32), intent(in) :: NODECOMM

#ifdef QUOP_GPU_AWARE_MPI
        ! GPU-aware MPI path: use MPI directly with device pointers
        call gpu_allgatherv_dtoh_mpi(dev_counts, host_counts, dev_displs, &
                                     host_displs, dev_ptr, host_ptr, mpi_type, &
                                     NODECOMM)
#else
        ! Host-staged fallback: copy device->host, MPI redistribute, done
        call gpu_allgatherv_dtoh_staged(dev_counts, host_counts, &
                                        dev_displs, host_displs, dev_ptr, &
                                        host_ptr, mpi_type, NODECOMM)
#endif

    end subroutine gpu_allgatherv_dtoh

    !> Scatter data from host memory on all ranks to device memory
    subroutine gpu_allscatterv_htod(host_counts, dev_counts, host_displs, &
                                    dev_displs, host_ptr, dev_ptr, mpi_type, &
                                    NODECOMM)
        integer(int64), dimension(:), target, intent(in) :: host_counts, dev_counts
        integer(int64), dimension(:), target, intent(in) :: host_displs, dev_displs
        type(c_ptr), value, intent(in) :: host_ptr
        type(c_ptr), value, intent(in) :: dev_ptr
        integer, intent(in) :: mpi_type
        integer(int32), intent(in) :: NODECOMM

#ifdef QUOP_GPU_AWARE_MPI
        character(len=64) :: env_val
        logical :: force_staged_htod
        logical :: env_is_valid
        integer(int32) :: node_rank, ierr

        call read_env_flag('QUOP_FORCE_STAGED_HTOD', .false., force_staged_htod, env_is_valid, env_val)
        if (.not. env_is_valid) then
            call MPI_Comm_rank(NODECOMM, node_rank, ierr)
            if (node_rank == 0) then
                write (error_unit, '(A,A,A)') &
                    'WARNING: QUOP_FORCE_STAGED_HTOD has unrecognised value "', &
                    trim(env_val), '". Using 0.'
            end if
        end if

        if (force_staged_htod) then
            ! Diagnostic escape hatch: force the host-staged HtoD path even
            ! when GPU-aware MPI is enabled, so the DtoH path can be tested
            ! independently of device-target MPI receives.
            call gpu_allscatterv_htod_staged(host_counts, dev_counts, &
                                             host_displs, dev_displs, host_ptr, &
                                             dev_ptr, mpi_type, NODECOMM)
        else
            ! GPU-aware MPI path: use MPI directly with device pointers
            call gpu_allscatterv_htod_mpi(host_counts, dev_counts, host_displs, &
                                          dev_displs, host_ptr, dev_ptr, mpi_type, &
                                          NODECOMM)
        end if
#else
        ! Host-staged fallback: MPI redistribute, then copy host->device
        call gpu_allscatterv_htod_staged(host_counts, dev_counts, &
                                         host_displs, dev_displs, host_ptr, &
                                         dev_ptr, mpi_type, NODECOMM)
#endif

    end subroutine gpu_allscatterv_htod

#ifdef QUOP_GPU_AWARE_MPI
    !> GPU-aware MPI implementation for device-to-host gather
    !>
    !> This redistributes data from device ranks to host ranks using MPI_Alltoallv.
    !> Each host rank receives only its local portion (host_counts(rank+1) elements),
    !> gathered from the device ranks that hold that data.
    !>
    !> The MPI library handles the GPU memory access transparently when compiled
    !> with GPU-aware MPI support.
    subroutine gpu_allgatherv_dtoh_mpi(dev_counts, host_counts, dev_displs, &
                                       host_displs, dev_ptr, host_ptr, mpi_type, &
                                       NODECOMM)
        integer(int64), dimension(:), intent(in) :: dev_counts, host_counts
        integer(int64), dimension(:), intent(in) :: dev_displs, host_displs
        type(c_ptr), value, intent(in) :: dev_ptr, host_ptr
        integer, intent(in) :: mpi_type
        integer(int32), intent(in) :: NODECOMM

        integer(c_int) :: ierr, rank, n_ranks
        integer(int64) :: dev_start, dev_end, host_start, host_end
        integer(int64) :: overlap_start, overlap_end
        integer, allocatable :: send_cnts(:), recv_cnts(:), send_displs(:), recv_displs(:)
        integer(c_size_t) :: element_size

        call MPI_Comm_size(NODECOMM, n_ranks, ierr)
        call MPI_Comm_rank(NODECOMM, rank, ierr)
        call get_mpi_element_size(mpi_type, element_size)

        ! Ensure all prior GPU work (kernels, async copies) has completed
        ! before reading device memory via hipMemcpy or GPU-aware MPI.
        ! NOTE: hipDeviceSynchronize does not provide an explicit L2 flush API.
        ! For GPU-aware MPI on coarse-grained memory, this call is the
        ! conservative ordering barrier before RDMA reads device buffers.
        call sync_device_if_nonempty(dev_counts(rank + 1))

        dev_start = dev_displs(rank + 1)
        dev_end = dev_start + dev_counts(rank + 1)

        host_start = host_displs(rank + 1)
        host_end = host_start + host_counts(rank + 1)

        ! ---- Local overlap: device->host via hipMemcpy (no MPI needed) ----
        overlap_start = max(dev_start, host_start)
        overlap_end = min(dev_end, host_end)
        call copy_local_overlap(dev_ptr, dev_start, host_ptr, host_start, overlap_start, &
                                overlap_end, element_size, hipMemcpyDeviceToHost)

        allocate (send_cnts(n_ranks), recv_cnts(n_ranks))
        allocate (send_displs(n_ranks), recv_displs(n_ranks))
        call build_transfer_schedule(dev_counts, host_counts, dev_displs, host_displs, &
                                     rank, NODECOMM, send_cnts, send_displs, recv_cnts, recv_displs)
        call exchange_nonlocal_segments(dev_ptr, host_ptr, send_cnts, send_displs, recv_cnts, &
                                        recv_displs, mpi_type, element_size, rank, NODECOMM)
        deallocate (send_cnts, recv_cnts, send_displs, recv_displs)

    end subroutine gpu_allgatherv_dtoh_mpi

    !> GPU-aware MPI implementation for host-to-device scatter
    !>
    !> This redistributes data from host ranks to device ranks using MPI_Alltoallv.
    !> Each device rank receives only its local portion (dev_counts(rank+1) elements),
    !> gathered from the host ranks that hold that data.
    subroutine gpu_allscatterv_htod_mpi(host_counts, dev_counts, host_displs, &
                                        dev_displs, host_ptr, dev_ptr, mpi_type, &
                                        NODECOMM)
        integer(int64), dimension(:), intent(in) :: host_counts, dev_counts
        integer(int64), dimension(:), intent(in) :: host_displs, dev_displs
        type(c_ptr), value, intent(in) :: host_ptr, dev_ptr
        integer, intent(in) :: mpi_type
        integer(int32), intent(in) :: NODECOMM

        integer(c_int) :: ierr, rank, n_ranks
        integer(int64) :: dev_start, dev_end, host_start, host_end
        integer(int64) :: overlap_start, overlap_end
        integer, allocatable :: send_cnts(:), recv_cnts(:), send_displs(:), recv_displs(:)
        integer(c_size_t) :: element_size

        call MPI_Comm_size(NODECOMM, n_ranks, ierr)
        call MPI_Comm_rank(NODECOMM, rank, ierr)
        call get_mpi_element_size(mpi_type, element_size)

        ! Ensure no prior device work is still writing dev_ptr before MPI starts
        ! receiving directly into it. This is the receive-side counterpart to the
        ! pre-send synchronization used before exposing device buffers to MPI.
        call sync_device_if_nonempty(dev_counts(rank + 1))

        host_start = host_displs(rank + 1)
        host_end = host_start + host_counts(rank + 1)

        dev_start = dev_displs(rank + 1)
        dev_end = dev_start + dev_counts(rank + 1)

        ! ---- Local overlap: host->device via hipMemcpy (no MPI needed) ----
        overlap_start = max(host_start, dev_start)
        overlap_end = min(host_end, dev_end)
        call copy_local_overlap(host_ptr, host_start, dev_ptr, dev_start, overlap_start, &
                                overlap_end, element_size, hipMemcpyHostToDevice)

        allocate (send_cnts(n_ranks), recv_cnts(n_ranks))
        allocate (send_displs(n_ranks), recv_displs(n_ranks))
        call build_transfer_schedule(host_counts, dev_counts, host_displs, dev_displs, &
                                     rank, NODECOMM, send_cnts, send_displs, recv_cnts, recv_displs)
        call exchange_nonlocal_segments(host_ptr, dev_ptr, send_cnts, send_displs, recv_cnts, &
                                        recv_displs, mpi_type, element_size, rank, NODECOMM)

        ! Ensure all MPI RDMA writes are visible to subsequent GPU kernels.
        ! For coarse-grained memory in GPU-aware MPI paths, this is the
        ! conservative synchronization point before device reads.
        call sync_device_if_nonempty(dev_counts(rank + 1))

        deallocate (send_cnts, recv_cnts, send_displs, recv_displs)

    end subroutine gpu_allscatterv_htod_mpi
#endif

    !> Host-staged fallback implementation for device-to-host gather
    !>
    !> When GPU-aware MPI is not available, we need to:
    !> 1. Copy data from device to a host staging buffer
    !> 2. Use MPI_Alltoallv to redistribute data (each rank receives only its portion)
    !> 3. The destination is already on host, so no additional copy needed
    !>
    !> Note: Uses the caller-supplied MPI datatype only to derive element size.
    !> The character staging buffers are exchanged with a byte-derived MPI type.
    subroutine gpu_allgatherv_dtoh_staged(dev_counts, host_counts, dev_displs, &
                                          host_displs, dev_ptr, host_ptr, mpi_type, &
                                          NODECOMM)
        integer(int64), dimension(:), intent(in) :: dev_counts, host_counts
        integer(int64), dimension(:), intent(in) :: dev_displs, host_displs
        type(c_ptr), value, intent(in) :: dev_ptr, host_ptr
        integer, intent(in) :: mpi_type
        integer(int32), intent(in) :: NODECOMM

        integer(c_int) :: ierr, rank, n_ranks
        integer(int64) :: my_dev_count, staging_bytes, host_buffer_bytes
        character(len=1), dimension(:), allocatable, target :: staging_buffer
        character(len=1), dimension(:), pointer :: host_buffer_fptr
        character(len=1) :: dummy_buffer(1) ! separate recv buffer to avoid MPI aliasing
        integer(c_size_t) :: elem_sz
        integer :: elem_type
        integer, allocatable :: scounts(:), sdispls(:), rcounts(:), rdispls(:)

        call MPI_Comm_rank(NODECOMM, rank, ierr)
        call MPI_Comm_size(NODECOMM, n_ranks, ierr)
        call get_mpi_element_size(mpi_type, elem_sz)
        call create_byte_element_type(elem_sz, elem_type)

        ! Determine this rank's device data count
        my_dev_count = dev_counts(rank + 1)

        ! Allocate staging buffer in bytes for this rank's device data
        staging_bytes = my_dev_count * int(elem_sz, int64)

        ! Calculate host buffer size for this rank (for Fortran pointer conversion)
        host_buffer_bytes = host_counts(rank + 1) * int(elem_sz, int64)

        if (staging_bytes > 0) then
            allocate (staging_buffer(staging_bytes))
            ! Ensure all prior GPU work has completed before DMA read.
            ! NOTE: on AMD gfx90a, hipDeviceSynchronize does NOT flush L2.
            ! hipMemcpy (same-device DMA) is expected to handle L2
            ! coherence internally for coarse-grained memory.
            call sync_device_if_nonempty(my_dev_count)
            ! Copy from device to staging buffer
            call copy_local_overlap(dev_ptr, 0_int64, c_loc(staging_buffer), 0_int64, &
                                    0_int64, my_dev_count, elem_sz, hipMemcpyDeviceToHost)
        else
            allocate (staging_buffer(1)) ! Dummy allocation for MPI call
        end if

        ! Allocate send/receive counts and displacements for MPI_Alltoallv (in elements)
        ! All ranks must participate in the collective, even with zero counts
        allocate (scounts(n_ranks), sdispls(n_ranks))
        allocate (rcounts(n_ranks), rdispls(n_ranks))
        call build_transfer_schedule(dev_counts, host_counts, dev_displs, host_displs, &
                                     rank, NODECOMM, scounts, sdispls, rcounts, rdispls)

        ! Convert C pointer to Fortran pointer for MPI receive buffer
        ! This is necessary because passing type(c_ptr) directly to MPI_Alltoallv
        ! as a receive buffer can cause the Fortran MPI bindings to pass the wrong address
        if (host_buffer_bytes > 0) then
            call c_f_pointer(host_ptr, host_buffer_fptr, [host_buffer_bytes])
        else
            ! Null pointer case - allocate dummy for MPI
            nullify (host_buffer_fptr)
        end if

        ! Use a byte-derived MPI datatype for the character staging buffers.
        if (associated(host_buffer_fptr)) then
            call MPI_Alltoallv(staging_buffer, scounts, sdispls, elem_type, &
                               host_buffer_fptr, rcounts, rdispls, elem_type, &
                               NODECOMM, ierr)
        else
            ! All zero counts - use separate dummy buffer to avoid
            ! MPI send/recv buffer aliasing (undefined per MPI standard)
            call MPI_Alltoallv(staging_buffer, scounts, sdispls, elem_type, &
                               dummy_buffer, rcounts, rdispls, elem_type, &
                               NODECOMM, ierr)
        end if

        call MPI_Type_free(elem_type, ierr)
        deallocate (staging_buffer, scounts, sdispls, rcounts, rdispls)

    end subroutine gpu_allgatherv_dtoh_staged

    !> Host-staged fallback implementation for host-to-device scatter
    !>
    !> When GPU-aware MPI is not available, we need to:
    !> 1. Use MPI_Alltoallv to redistribute host data to staging buffer
    !> 2. Copy from host staging buffer to device
    !>
    !> Note: Uses the caller-supplied MPI datatype only to derive element size.
    !> The character staging buffers are exchanged with a byte-derived MPI type.
    subroutine gpu_allscatterv_htod_staged(host_counts, dev_counts, host_displs, &
                                           dev_displs, host_ptr, dev_ptr, mpi_type, &
                                           NODECOMM)
        integer(int64), dimension(:), intent(in) :: host_counts, dev_counts
        integer(int64), dimension(:), intent(in) :: host_displs, dev_displs
        type(c_ptr), value, intent(in) :: host_ptr, dev_ptr
        integer, intent(in) :: mpi_type
        integer(int32), intent(in) :: NODECOMM

        integer(c_int) :: ierr, rank, n_ranks
        integer(int64) :: my_dev_count, staging_bytes, host_buffer_bytes
        character(len=1), dimension(:), allocatable, target :: staging_buffer
        character(len=1), dimension(:), pointer :: host_buffer_fptr
        character(len=1) :: dummy_buffer(1) ! separate send buffer to avoid MPI aliasing
        integer(c_size_t) :: elem_sz
        integer :: elem_type
        integer, allocatable :: scounts(:), sdispls(:), rcounts(:), rdispls(:)

        call MPI_Comm_rank(NODECOMM, rank, ierr)
        call MPI_Comm_size(NODECOMM, n_ranks, ierr)
        call get_mpi_element_size(mpi_type, elem_sz)
        call create_byte_element_type(elem_sz, elem_type)

        ! Determine this rank's device data count
        my_dev_count = dev_counts(rank + 1)

        ! Calculate host buffer size for Fortran pointer conversion
        host_buffer_bytes = host_counts(rank + 1) * int(elem_sz, int64)

        ! Allocate staging buffer in bytes for this rank's device data
        staging_bytes = my_dev_count * int(elem_sz, int64)
        if (staging_bytes > 0) then
            allocate (staging_buffer(staging_bytes))
        else
            allocate (staging_buffer(1)) ! Dummy allocation for MPI call
        end if

        ! Allocate send/receive counts and displacements for MPI_Alltoallv (in elements)
        ! All ranks must participate in the collective, even with zero counts
        allocate (scounts(n_ranks), sdispls(n_ranks))
        allocate (rcounts(n_ranks), rdispls(n_ranks))
        call build_transfer_schedule(host_counts, dev_counts, host_displs, dev_displs, &
                                     rank, NODECOMM, scounts, sdispls, rcounts, rdispls)

        ! Convert C pointer to Fortran pointer for MPI send buffer
        ! This is necessary because passing type(c_ptr) directly to MPI_Alltoallv
        ! can cause the Fortran MPI bindings to pass the wrong address
        if (host_buffer_bytes > 0) then
            call c_f_pointer(host_ptr, host_buffer_fptr, [host_buffer_bytes])
        else
            nullify (host_buffer_fptr)
        end if

        ! Use a byte-derived MPI datatype for the character staging buffers.
        if (associated(host_buffer_fptr)) then
            call MPI_Alltoallv(host_buffer_fptr, scounts, sdispls, elem_type, &
                               staging_buffer, rcounts, rdispls, elem_type, &
                               NODECOMM, ierr)
        else
            ! Zero host count - use separate dummy buffer to avoid
            ! MPI send/recv buffer aliasing (undefined per MPI standard)
            call MPI_Alltoallv(dummy_buffer, scounts, sdispls, elem_type, &
                               staging_buffer, rcounts, rdispls, elem_type, &
                               NODECOMM, ierr)
        end if

        ! Copy from staging buffer to device (skip if nothing to copy)
        if (staging_bytes > 0) then
            call copy_local_overlap(c_loc(staging_buffer), 0_int64, dev_ptr, 0_int64, &
                                    0_int64, my_dev_count, elem_sz, hipMemcpyHostToDevice)
            ! Ensure DMA write completes before subsequent GPU kernels.
            ! NOTE: on AMD gfx90a, hipDeviceSynchronize does NOT
            ! invalidate L2.  hipMemcpy (same-device DMA) is expected to
            ! handle L2 coherence internally for coarse-grained memory.
            call sync_device_if_nonempty(my_dev_count)
        end if

        call MPI_Type_free(elem_type, ierr)
        deallocate (staging_buffer, scounts, sdispls, rcounts, rdispls)

    end subroutine gpu_allscatterv_htod_staged

    !> Safely convert int64 to int32, aborting on overflow.
    !> Prevents silent truncation when int64 element counts or displacements
    !> exceed the 32-bit range required by MPI count/displacement arguments.
    !> In practice counts and displacements here are nonneg, so the lower
    !> bound check is defensive.
    integer(int32) function safe_int32(val, comm)
        integer(int64), intent(in) :: val
        integer(int32), intent(in) :: comm
        integer :: ierr
        integer(int64), parameter :: i32_max = 2147483647_int64 ! huge(1_int32)
        integer(int64), parameter :: i32_min = -2147483648_int64
        if (val > i32_max .or. val < i32_min) then
            write (error_unit, '(A,I0,A)') &
                "FATAL [gpu_transfer]: value ", val, " exceeds 32-bit MPI limit"
            call MPI_Abort(comm, 1, ierr)
        end if
        safe_int32 = int(val, int32)
    end function safe_int32

#ifdef QUOP_GPU_AWARE_MPI
    subroutine read_env_flag(name, default_value, value, env_is_valid, raw_value)
        character(len=*), intent(in) :: name
        logical, intent(in) :: default_value
        logical, intent(out) :: value
        logical, intent(out) :: env_is_valid
        character(len=*), intent(out) :: raw_value

        raw_value = ''
        call get_environment_variable(name, raw_value)
        raw_value = trim(adjustl(raw_value))

        if (len_trim(raw_value) == 0) then
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
#endif

end module gpu_transfer
