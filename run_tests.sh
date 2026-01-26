#!/bin/bash
#
# QuOp_MPI Test Runner
#
# Usage:
#   ./run_tests.sh              # Run all tests (2 procs, then 12 procs with oversubscribe)
#   ./run_tests.sh 4            # Run tests with 4 MPI processes
#   ./run_tests.sh 2 mpi        # Run only MPI tests (2 procs)
#   ./run_tests.sh 1 unit       # Run only unit tests (serial)
#   ./run_tests.sh 2 mpi-full   # Run MPI tests with both 2 and 12 process phases
#

set -e

NPROCS=${1:-2}
TEST_TYPE=${2:-all}

# Number of processes for parallel jacobian tests
NPROCS_PARALLEL_JACOBIAN=12

echo "========================================"
echo "QuOp_MPI Test Suite"
echo "========================================"
echo "MPI Processes: $NPROCS"
echo "Test Type: $TEST_TYPE"
echo "========================================"
echo ""

run_mpi_tests() {
    local nprocs=$1
    local extra_args="${2:-}"
    local mpi_extra="${3:-}"
    
    echo "Running MPI tests with $nprocs processes..."
    mpiexec $mpi_extra -n $nprocs python -m pytest tests/mpi/ -v --tb=short --with-mpi $extra_args 2>&1 | awk '!seen[$0]++'
}

run_parallel_jacobian_tests() {
    echo ""
    echo "========================================"
    echo "Running parallel jacobian tests (12 procs with oversubscribe)..."
    echo "========================================"
    mpiexec --oversubscribe -n $NPROCS_PARALLEL_JACOBIAN python -m pytest tests/mpi/ -v --tb=short --with-mpi -m "requires_nprocs" 2>&1 | awk '!seen[$0]++'
}

case $TEST_TYPE in
    unit)
        echo "Running unit tests (serial)..."
        python -m pytest tests/unit/ -v --tb=short
        ;;
    mpi)
        # Run with specified number of processes, skipping tests that need more
        run_mpi_tests $NPROCS
        ;;
    mpi-full)
        # Run tests that work with NPROCS, then run parallel jacobian tests with 12
        run_mpi_tests $NPROCS
        run_parallel_jacobian_tests
        ;;
    all)
        echo "Running unit tests (serial)..."
        python -m pytest tests/unit/ -v --tb=short || true
        echo ""
        # Run all tests with NPROCS (tests requiring more will be skipped)
        run_mpi_tests $NPROCS
        # Then run tests requiring 12 processes with oversubscribe
        run_parallel_jacobian_tests
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Valid options: unit, mpi, mpi-full, all"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Tests completed!"
echo "========================================"
