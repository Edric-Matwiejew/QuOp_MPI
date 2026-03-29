#!/bin/bash
#
# QuOp_MPI Test Runner
#
# Usage:
#   ./run_tests.sh              # Run all tests (2 procs), then examples
#   ./run_tests.sh 4            # Run all tests with 4 MPI processes
#   ./run_tests.sh 2,12 mpi    # MPI tests at 2 procs, then requires_nprocs tests at 12
#   ./run_tests.sh 1 unit       # Run only unit tests (serial)
#   ./run_tests.sh 2 mpi --isolate  # Per-file isolation (debug GPU/communicator issues)
#
# The first process count runs all MPI tests.  Additional counts run only
# tests marked with @requires_nprocs (with --oversubscribe on non-Cray).
# The backend is auto-detected from the QUOP_BACKEND environment variable.
# Pass --isolate to run each MPI test file in its own mpiexec invocation,
# useful for debugging cross-file GPU / MPI communicator state issues.
#

set -e

NPROCS_LIST=${1:-2}
TEST_TYPE=${2:-all}
ISOLATE=0
for arg in "$@"; do
    if [[ "$arg" == "--isolate" ]]; then
        ISOLATE=1
    fi
done

# Split comma-separated process counts into an array.
IFS=',' read -r -a NPROCS_ARRAY <<< "$NPROCS_LIST"

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
echo "MPI Processes: ${NPROCS_ARRAY[*]}"
echo "Test Type: $TEST_TYPE"
echo "Backend: $BACKEND"
echo "Launcher: $(if [[ $IS_CRAY -eq 1 ]]; then echo srun; else echo mpiexec; fi)"
echo "Isolate: $(if [[ $ISOLATE -eq 1 ]]; then echo yes; else echo no; fi)"
echo "========================================"
echo ""

# -- helpers ------------------------------------------------------------

_launch() {
    # Build the MPI launcher command prefix.
    # Usage: _launch <nprocs> [extra_launcher_args...]
    local nprocs=$1
    shift
    if [[ $IS_CRAY -eq 1 ]]; then
        local gpu_args=()
        if [[ "$BACKEND" == "wavefront" ]]; then
            local ngpus=$(( nprocs < 8 ? nprocs : 8 ))
            gpu_args=(--gpus="$ngpus")
        fi
        srun -N 1 -n "$nprocs" "${gpu_args[@]}" "$@"
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
    # Run MPI tests for each process count in NPROCS_ARRAY.
    # The first count runs all tests; subsequent counts run only
    # tests marked with @requires_nprocs.
    local first=1

    for nprocs in "${NPROCS_ARRAY[@]}"; do
        local extra_args=""

        if [[ $first -eq 1 ]]; then
            first=0
            echo "Running MPI tests with $nprocs processes..."
        else
            extra_args="-m requires_nprocs"
            echo ""
            echo "========================================"
            echo "Running requires_nprocs tests ($nprocs procs)..."
            echo "========================================"
        fi

        if [[ $ISOLATE -eq 1 ]]; then
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
            if [[ $IS_CRAY -eq 1 ]]; then
                _launch "$nprocs" python -m pytest tests/mpi/ \
                    -v --tb=short --with-mpi $extra_args
            elif [[ -n "$extra_args" ]]; then
                # Subsequent counts may exceed available slots; oversubscribe.
                mpiexec --oversubscribe -n "$nprocs" python -m pytest tests/mpi/ \
                    -v --tb=short --with-mpi $extra_args
            else
                _launch "$nprocs" python -m pytest tests/mpi/ \
                    -v --tb=short --with-mpi $extra_args
            fi
        fi
    done
}

# -- main ---------------------------------------------------------------

case $TEST_TYPE in
    unit)
        echo "Running unit tests (serial)..."
        python -m pytest tests/unit/ -v --tb=short
        ;;
    mpi)
        run_mpi_tests
        ;;
    all)
        echo "Running unit tests (serial)..."
        python -m pytest tests/unit/ -v --tb=short || true
        echo ""
        run_mpi_tests
        # Run example tests with 4 MPI processes
        echo ""
        echo "========================================"
        echo "Running example tests (4 MPI processes)..."
        echo "========================================"
        if [[ $IS_CRAY -eq 1 ]]; then
            OMP_NUM_THREADS=1 python -m pytest tests/examples/ \
                -v --tb=short --nprocs 4 --launcher srun
        else
            OMP_NUM_THREADS=1 python -m pytest tests/examples/ \
                -v --tb=short --nprocs 4 --launcher mpiexec
        fi
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Valid options: unit, mpi, all"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "All tests completed!"
echo "========================================"
