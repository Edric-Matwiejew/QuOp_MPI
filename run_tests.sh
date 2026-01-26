#!/bin/bash
#
# QuOp_MPI Test Runner
#
# Usage:
#   ./run_tests.sh              # Run all tests with 2 MPI processes
#   ./run_tests.sh 4            # Run all tests with 4 MPI processes  
#   ./run_tests.sh 2 mpi        # Run only MPI tests
#   ./run_tests.sh 1 unit       # Run only unit tests (serial)
#

set -e

NPROCS=${1:-2}
TEST_TYPE=${2:-all}

echo "========================================"
echo "QuOp_MPI Test Suite"
echo "========================================"
echo "MPI Processes: $NPROCS"
echo "Test Type: $TEST_TYPE"
echo "========================================"
echo ""

case $TEST_TYPE in
    unit)
        echo "Running unit tests (serial)..."
        python -m pytest tests/unit/ -v --tb=short
        ;;
    mpi)
        echo "Running MPI integration tests..."
        mpiexec -n $NPROCS python -m pytest tests/mpi/ -v --tb=short --with-mpi 2>&1 | awk '!seen[$0]++'
        ;;
    all)
        echo "Running unit tests (serial)..."
        python -m pytest tests/unit/ -v --tb=short || true
        echo ""
        echo "Running MPI integration tests..."
        mpiexec -n $NPROCS python -m pytest tests/mpi/ -v --tb=short --with-mpi 2>&1 | awk '!seen[$0]++'
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Valid options: unit, mpi, all"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Tests completed!"
echo "========================================"
