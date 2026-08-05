! gpu_topology_stub.f90
!
! MPI-backend stub for gpu_topology_t.
!
! Provides the same type name and public interface as the real
! gpu_topology module (native/wavefront/context/gpu_topology.f90)
! but with no HIP/GPU dependencies.  All GPU fields are initialised
! to zero/false defaults.
!
! Stage 1: Used by comm_info_module.f90 so that quop_mpi_layout_t
! can carry a topology member without depending on the wavefront
! backend.  The real gpu_topology module is swapped in at Stage 6
! via conditional compilation.

module gpu_topology

    use, intrinsic :: iso_fortran_env, only: int32
    use mpi, only: MPI_MAX_PROCESSOR_NAME
    implicit none

    private
    public :: visible_gpu_info_t, gpu_topology_t

    type :: visible_gpu_info_t
        integer(int32) :: device_id = -1
        integer(int32) :: physical_gpu_index = -1
        integer(int32) :: numa_node = -1
        character(len=16) :: pci_bus_id = ''
    end type visible_gpu_info_t

    type :: gpu_topology_t
        ! Configuration (read from environment in wavefront; defaults here)
        integer(int32) :: ranks_per_gpu = 0
        character(len=16) :: binding_mode = 'none'
        character(len=16) :: binding_strategy = 'none'

        ! Detected topology (all zero/false for MPI backend)
        integer(int32) :: visible_device_count = 0
        integer(int32) :: n_physical_gpus = 0
        integer(int32) :: my_gpu_index = -1
        integer(int32) :: assigned_device_id = -1
        integer(int32) :: rank_within_gpu = 0
        integer(int32) :: gpu_slot_ordinal = -1
        type(visible_gpu_info_t), allocatable :: visible_gpus(:)
        integer(int32) :: cpu_numa_node = -1
        integer(int32) :: rank_within_cpu_numa = 0
        logical        :: is_gpu_rank = .false.

        ! Node info (for reference; zero until populated)
        integer(int32) :: node_rank = 0
        integer(int32) :: node_size = 0
        integer(int32) :: devcomm_node_size = 0
        character(len=MPI_MAX_PROCESSOR_NAME) :: hostname = ''

        ! Global node topology (populated by discover_topology)
        integer(int32) :: n_nodes = 1 ! total compute nodes
        integer(int32) :: node_id = 0 ! 0-based sequential node index
    end type gpu_topology_t

end module gpu_topology
