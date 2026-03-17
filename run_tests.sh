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
# The backend is auto-detected from the QUOP_BACKEND environment variable.
# On the wavefront backend each MPI test file is run in its own mpiexec
# invocation to avoid cross-file deadlocks caused by cumulative GPU / MPI
# communicator state.
#

set -e

NPROCS=${1:-2}
TEST_TYPE=${2:-all}

# Number of processes for parallel jacobian tests
NPROCS_PARALLEL_JACOBIAN=12

# Detect backend from environment (default: mpi)
BACKEND="${QUOP_BACKEND:-mpi}"

# Detect Cray systems (use srun instead of mpiexec)
if command -v srun &>/dev/null && [[ -d /opt/cray ]] ; then
    IS_CRAY=1
else
    IS_CRAY=0
fi

echo "========================================"
echo "QuOp_MPI Test Suite"
echo "========================================"
echo "MPI Processes: $NPROCS"
echo "Test Type: $TEST_TYPE"
echo "Backend: $BACKEND"
echo "Launcher: $(if [[ $IS_CRAY -eq 1 ]]; then echo srun; else echo mpiexec; fi)"
echo "========================================"
echo ""

# -- helpers ------------------------------------------------------------

_launch() {
    # Build the MPI launcher command prefix.
    # Usage: _launch <nprocs> [extra_launcher_args...]
    local nprocs=$1
    shift
    if [[ $IS_CRAY -eq 1 ]]; then
        srun -N 1 -n "$nprocs" --gpus="$nprocs" "$@"
    else
        mpiexec -n "$nprocs" "$@"
    fi
}

_run_mpi_single() {
    # Run a single pytest invocation covering a directory or file.
    local nprocs=$1
    local target=$2
    shift 2
    local extra_args="$*"

    _launch "$nprocs" python -m pytest "$target" \
        -v --tb=short --with-mpi $extra_args
}

run_mpi_tests() {
    local nprocs=$1
    local extra_args="${2:-}"

    if [[ "$BACKEND" == "wavefront" ]]; then
        echo "Running MPI tests with $nprocs processes (per-file, wavefront backend)..."
        local failed=0
        for test_file in $(ls tests/mpi/test_*.py | sort); do
            echo ""
            echo "---- $(basename "$test_file") ----"
            if ! _run_mpi_single "$nprocs" "$test_file" $extra_args; then
                echo "FAILED: $test_file"
                failed=1
            fi
        done
        if [[ $failed -ne 0 ]]; then
            echo ""
            echo "Some test files failed (see above)."
            return 1
        fi
    else
        echo "Running MPI tests with $nprocs processes..."
        _launch "$nprocs" python -m pytest tests/mpi/ \
            -v --tb=short --with-mpi $extra_args
    fi
}

run_parallel_jacobian_tests() {
    echo ""
    echo "========================================"
    echo "Running parallel jacobian tests ($NPROCS_PARALLEL_JACOBIAN procs)..."
    echo "========================================"
    if [[ "$BACKEND" == "wavefront" ]]; then
        for test_file in $(ls tests/mpi/test_*.py | sort); do
            echo ""
            echo "---- $(basename "$test_file") (parallel jacobian) ----"
            _run_mpi_single "$NPROCS_PARALLEL_JACOBIAN" "$test_file" \
                -m "requires_nprocs" || true
        done
    else
        if [[ $IS_CRAY -eq 1 ]]; then
            _launch "$NPROCS_PARALLEL_JACOBIAN" python -m pytest tests/mpi/ \
                -v --tb=short --with-mpi -m "requires_nprocs"
        else
            mpiexec --oversubscribe -n "$NPROCS_PARALLEL_JACOBIAN" python -m pytest tests/mpi/ \
                -v --tb=short --with-mpi -m "requires_nprocs"
        fi
    fi
}

# -- main ---------------------------------------------------------------

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
        # Run integration tests with 4 MPI processes
        echo ""
        echo "========================================"
        echo "Running integration tests (4 MPI processes)..."
        echo "========================================"
        if [[ $IS_CRAY -eq 1 ]]; then
            OMP_NUM_THREADS=1 python tests/integration/run_integration_tests.py 4 srun
        else
            OMP_NUM_THREADS=1 python tests/integration/run_integration_tests.py 4 mpiexec
        fi
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Valid options: unit, mpi, mpi-full, all"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "All tests completed!"
echo "========================================"
