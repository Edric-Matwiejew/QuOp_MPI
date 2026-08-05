# native/tests/mpi/CMakeLists.txt
# CTest configuration for MPI backend tests

# Collect wavefront link dependencies for tests that use comm_info_mod
if(WAVEFRONT_BACKEND)
    set(_COMM_INFO_DEPS comm_info_mod ci_gpu_topology ci_communicators ci_partitions
        MPI::MPI_Fortran ${QUOP_HIPFORT_TARGET})
    if(HIP_PLATFORM STREQUAL "nvidia")
        list(APPEND _COMM_INFO_DEPS ${CUDA_PATH}/lib64/libcudart.so)
    elseif(HIP_PLATFORM STREQUAL "amd")
        list(APPEND _COMM_INFO_DEPS ${ROCM_PATH}/lib/libamdhip64.so)
    endif()
else()
    set(_COMM_INFO_DEPS comm_info_mod MPI::MPI_Fortran)
endif()

# ============================================================================
# Basic MPI initialization test
# ============================================================================
add_executable(test_mpi_init
    test_mpi_init.f90
)

target_link_libraries(test_mpi_init
    MPI::MPI_Fortran
)

add_mpi_test(mpi_init_1proc test_mpi_init 1)
add_mpi_test(mpi_init_2proc test_mpi_init 2)
add_mpi_test(mpi_init_4proc test_mpi_init 4)

# ============================================================================
# quop_mpi_layout_t (comm_info) tests
# ============================================================================

# Main test suite (non-abort tests: T1.1, T1.2, T1.4-T1.9)
add_executable(test_comm_info
    test_comm_info.f90
)
target_link_libraries(test_comm_info
    ${_COMM_INFO_DEPS}
)
target_include_directories(test_comm_info
    PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

add_mpi_test(comm_info_1proc test_comm_info 1)
add_mpi_test(comm_info_2proc test_comm_info 2)
add_mpi_test(comm_info_4proc test_comm_info 4)

# T1.3: set_partitioning on locked layout -> nonzero error_code (exits 0 on success)
add_executable(test_comm_info_abort_lock
    test_comm_info_abort_lock.f90
)
target_link_libraries(test_comm_info_abort_lock
    ${_COMM_INFO_DEPS}
)
target_include_directories(test_comm_info_abort_lock
    PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

add_mpi_test(comm_info_abort_lock test_comm_info_abort_lock 1)

# T1.10: validate on bad partition -> nonzero error_code (exits 0 on success)
add_executable(test_comm_info_abort_validate
    test_comm_info_abort_validate.f90
)
target_link_libraries(test_comm_info_abort_validate
    ${_COMM_INFO_DEPS}
)
target_include_directories(test_comm_info_abort_validate
    PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

add_mpi_test(comm_info_abort_validate test_comm_info_abort_validate 2)

# Entry-point pipeline tests (discover, split, negotiate, create_jaccomm)
add_executable(test_comm_info_entry_points
    test_comm_info_entry_points.f90
)
target_link_libraries(test_comm_info_entry_points
    ${_COMM_INFO_DEPS}
)
target_include_directories(test_comm_info_entry_points
    PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

add_mpi_test(comm_info_entry_1proc test_comm_info_entry_points 1)
add_mpi_test(comm_info_entry_2proc test_comm_info_entry_points 2)
add_mpi_test(comm_info_entry_4proc test_comm_info_entry_points 4)

# ============================================================================
# CPU sparse SpMV correctness test (non-symmetric A, non-uniform x)
# ============================================================================
add_executable(test_sparse_cpu_spmv
    test_sparse_cpu_spmv.f90
)
target_link_libraries(test_sparse_cpu_spmv
    sparse
    MPI::MPI_Fortran
)
target_include_directories(test_sparse_cpu_spmv
    PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

add_mpi_test(sparse_cpu_spmv_1proc test_sparse_cpu_spmv 1)
add_mpi_test(sparse_cpu_spmv_2proc test_sparse_cpu_spmv 2)
add_mpi_test(sparse_cpu_spmv_4proc test_sparse_cpu_spmv 4)
add_mpi_test(sparse_cpu_spmv_8proc test_sparse_cpu_spmv 8)
