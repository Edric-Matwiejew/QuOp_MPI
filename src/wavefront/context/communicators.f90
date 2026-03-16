module communicators
    use, intrinsic :: iso_fortran_env, only: real32, real64, real128, int32, int64
    use mpi
    use hipfort
    use hipfort_check
    use gpu_topology, only: gpu_topology_t, init_gpu_topology
    implicit none

    private

    public :: create_NODECOMM, free_communicators, create_devcomm_with_topology, create_devcomm_with_data

contains

    ! Return an MPI communicator that contains all the processes on the same physical node
    subroutine create_NODECOMM(COMM, node_comm)

        integer(int32), intent(in) :: COMM
        integer(int32), intent(out) :: node_comm
        integer(int32) :: rank, ierr

        call MPI_COMM_rank(COMM, rank, ierr)
        call MPI_Comm_split_type(COMM, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, node_comm, ierr)

    end subroutine create_NODECOMM

    !> Create DEVCOMM and DEVCOMM_NODE using pre-computed GPU topology.
    !!
    !! This is the preferred method for creating device communicators as it:
    !! - Respects external GPU binding (e.g., --gpu-bind=closest)
    !! - Supports multi-rank-per-GPU configurations
    !! - Uses the correct assigned_device_id from topology detection
    !!
    !! @param[in]  COMM         Global communicator (e.g., MPI_COMM_WORLD or SUBCOMM)
    !! @param[in]  NODECOMM     Node-local communicator
    !! @param[in]  topology     Pre-computed GPU topology from init_gpu_topology
    !! @param[out] devcomm      Output: communicator containing all GPU ranks (global)
    !! @param[out] devcomm_node Output: communicator containing GPU ranks on this node
    subroutine create_devcomm_with_topology(COMM, NODECOMM, topology, devcomm, devcomm_node)

        integer(int32), intent(in) :: COMM
        integer(int32), intent(in) :: NODECOMM
        type(gpu_topology_t), intent(in) :: topology
        integer(int32), intent(out) :: devcomm
        integer(int32), intent(out) :: devcomm_node

        integer(int32) :: color, key, ierr, comm_rank

        ! ===================================================================
        ! Create DEVCOMM (global: all GPU ranks across all nodes)
        ! Key = COMM rank so per-node GPU ranks are contiguous in DEVCOMM.
        ! Using node_rank would interleave nodes (all node_rank=0 first, etc.).
        ! ===================================================================
        if (topology%is_gpu_rank) then
            color = 1
        else
            color = MPI_UNDEFINED
        end if
        call MPI_Comm_rank(COMM, comm_rank, ierr)
        key = comm_rank

        call MPI_Comm_split(COMM, color, key, devcomm, ierr)

        ! ===================================================================
        ! Create DEVCOMM_NODE (node-local: GPU ranks on THIS node only)
        ! node_rank is unique within NODECOMM, so ordering is correct here.
        ! ===================================================================
        key = topology%node_rank
        call MPI_Comm_split(NODECOMM, color, key, devcomm_node, ierr)

        ! ===================================================================
        ! Set the GPU device for GPU ranks
        ! ===================================================================
        if (topology%is_gpu_rank) then
            call hipCheck(hipSetDevice(topology%assigned_device_id))
        end if

    end subroutine create_devcomm_with_topology

    !> Create DEVCOMM and DEVCOMM_NODE based on data distribution.
    !!
    !! Unlike create_devcomm_with_topology which uses GPU topology alone,
    !! this routine creates communicators containing only ranks that have
    !! both a GPU AND non-zero local data. This is needed when the data
    !! distribution (e.g., from SHAFFT) assigns zero elements to some GPU ranks.
    !!
    !! @param[in]  COMM           Global communicator (e.g., MPI_COMM_WORLD or SUBCOMM)
    !! @param[in]  NODECOMM       Node-local communicator
    !! @param[in]  topology       Pre-computed GPU topology from init_gpu_topology
    !! @param[in]  has_data       True if this rank has non-zero local data
    !! @param[out] devcomm        Output: communicator containing ranks with GPUs AND data
    !! @param[out] devcomm_node   Output: node-local communicator for ranks with GPUs AND data
    subroutine create_devcomm_with_data(COMM, NODECOMM, topology, has_data, devcomm, devcomm_node)

        integer(int32), intent(in) :: COMM
        integer(int32), intent(in) :: NODECOMM
        type(gpu_topology_t), intent(in) :: topology
        logical, intent(in) :: has_data
        integer(int32), intent(out) :: devcomm
        integer(int32), intent(out) :: devcomm_node

        integer(int32) :: color, key, ierr, comm_rank
        logical :: is_active_gpu_rank

        ! A rank is an active GPU rank if it has a GPU AND has data to process
        is_active_gpu_rank = topology%is_gpu_rank .and. has_data

        ! ===================================================================
        ! Create DEVCOMM (global: ranks with GPUs AND data across all nodes)
        ! Key = COMM rank so per-node GPU ranks are contiguous in DEVCOMM.
        ! Using node_rank would interleave nodes (all node_rank=0 first, etc.).
        ! ===================================================================
        if (is_active_gpu_rank) then
            color = 1
        else
            color = MPI_UNDEFINED
        end if
        call MPI_Comm_rank(COMM, comm_rank, ierr)
        key = comm_rank

        call MPI_Comm_split(COMM, color, key, devcomm, ierr)

        ! ===================================================================
        ! Create DEVCOMM_NODE (node-local: ranks with GPUs AND data on THIS node)
        ! node_rank is unique within NODECOMM, so ordering is correct here.
        ! ===================================================================
        key = topology%node_rank
        call MPI_Comm_split(NODECOMM, color, key, devcomm_node, ierr)

        ! ===================================================================
        ! Set the GPU device for active GPU ranks
        ! ===================================================================
        if (is_active_gpu_rank) then
            call hipCheck(hipSetDevice(topology%assigned_device_id))
        end if

    end subroutine create_devcomm_with_data

    ! Free the communicators if they are not MPI_COMM_NULL
    subroutine free_communicators(DEVCOMM, NODECOMM, DEVCOMM_NODE)

        integer(int32), intent(inout) :: DEVCOMM, NODECOMM, DEVCOMM_NODE
        integer(int32) :: ierr

        if (DEVCOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(DEVCOMM, ierr)
        end if

        if (NODECOMM /= MPI_COMM_NULL) then
            call MPI_Comm_free(NODECOMM, ierr)
        end if

        if (DEVCOMM_NODE /= MPI_COMM_NULL) then
            call MPI_Comm_free(DEVCOMM_NODE, ierr)
        end if

    end subroutine free_communicators

end module communicators
