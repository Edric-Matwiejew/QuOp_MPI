#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# run_topology_tests.sh — Automated topology verification for QuOp_QUISA
#
# Runs diagonal-oracle (Tier 1) and algorithm-class (Tier 2) topology
# tests, verifying dump output against expected values and structural
# invariants.
#
# Usage:
#   bash run_topology_tests.sh [OPTIONS]
#
# Options:
#   --launcher CMD   MPI launcher command (default: "srun" if available,
#                    else "mpirun")
#   --results DIR    Directory for dump files and results (default: ./topology_results)
#   --phase PHASE    Which dump phase to verify: "init" or "locked" (default: locked)
#   --tier N         Run only tier N (1 or 2).  Default: both.
#   --dry-run        Print commands without executing.
#   -h, --help       Show this help.
#
# Prerequisites:
#   - QuOp_QUISA installed (quop_mpi importable)
#   - MPI launcher available (srun or mpirun)
#   - tests/topology/ scripts on PYTHONPATH-relative import path
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────
LAUNCHER=""
RESULTS_DIR="./topology_results"
PHASE="locked"
TIER="both"
DRY_RUN=false

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --launcher)  LAUNCHER="$2";      shift 2 ;;
        --results)   RESULTS_DIR="$2";   shift 2 ;;
        --phase)     PHASE="$2";         shift 2 ;;
        --tier)      TIER="$2";          shift 2 ;;
        --dry-run)   DRY_RUN=true;       shift   ;;
        -h|--help)
            sed -n '2,/^# ─/{ /^# ─/!s/^# //p; }' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Auto-detect launcher
if [[ -z "$LAUNCHER" ]]; then
    if command -v srun &>/dev/null; then
        LAUNCHER="srun"
    elif command -v mpirun &>/dev/null; then
        LAUNCHER="mpirun"
    else
        echo "ERROR: No MPI launcher found. Install mpirun or use --launcher." >&2
        exit 1
    fi
fi

# ── Paths ────────────────────────────────────────────────────────────
GENERATE="$SCRIPT_DIR/generate_expected.py"
VERIFY="$SCRIPT_DIR/verify_topology.py"
DIAG_RUNNER="$SCRIPT_DIR/run_diagonal_topology.py"
ALGO_RUNNER="$SCRIPT_DIR/run_algorithm_topology.py"
EXPECTED_CSV="$RESULTS_DIR/expected.csv"

# ── Colours ──────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'
    BOLD='\033[1m'; RESET='\033[0m'
else
    GREEN=''; RED=''; YELLOW=''; BOLD=''; RESET=''
fi

pass_count=0
fail_count=0
skip_count=0

log_pass() { echo -e "  ${GREEN}PASS${RESET} $1"; ((pass_count++)) || true; }
log_fail() { echo -e "  ${RED}FAIL${RESET} $1"; ((fail_count++)) || true; }
log_skip() { echo -e "  ${YELLOW}SKIP${RESET} $1"; ((skip_count++)) || true; }
log_info() { echo -e "${BOLD}$1${RESET}"; }

# ── Helper: build launcher command ───────────────────────────────────
# Usage: build_launch_cmd <n_nodes> <n_tasks> <tasks_per_node>
build_launch_cmd() {
    local nodes="$1" ntasks="$2" tpn="$3"
    if [[ "$LAUNCHER" == "srun" ]]; then
        echo "srun -N${nodes} -n${ntasks} --ntasks-per-node=${tpn}"
    else
        echo "mpirun -n ${ntasks}"
    fi
}

# ── Helper: run a test and collect dump ──────────────────────────────
# Usage: run_test <test_id> <n_nodes> <rpn> <rpg> <binding> <partition> <cmd_suffix>
run_test() {
    local test_id="$1" nodes="$2" rpn="$3" rpg="$4"
    local binding="$5" partition="$6"
    shift 6
    local cmd_suffix="$*"

    local ntasks=$((nodes * rpn))
    local test_dir="$RESULTS_DIR/$test_id"
    mkdir -p "$test_dir"

    # Build environment
    local env_vars="QUOP_DUMP_COMM_INFO=$test_dir"
    if [[ "$partition" == "gpu" ]]; then
        env_vars="$env_vars QUOP_GPU_BINDING_MODE=$binding QUOP_RANKS_PER_GPU=$rpg"
    fi

    local launch
    launch="$(build_launch_cmd "$nodes" "$ntasks" "$rpn")"
    local full_cmd="$env_vars $launch $cmd_suffix"

    if $DRY_RUN; then
        echo "  [dry-run] $full_cmd"
        log_skip "$test_id (dry-run)"
        return 99
    fi

    # Clean previous dumps
    rm -f "$test_dir"/quop_comm_info_*.txt

    echo "  Running: $full_cmd"
    if env $env_vars $launch $cmd_suffix > "$test_dir/stdout.log" 2>"$test_dir/stderr.log"; then
        return 0
    else
        local rc=$?
        echo "  WARNING: exit code $rc (see $test_dir/stderr.log)" >&2
        return $rc
    fi
}

# ── Helper: find the dump file for a phase ───────────────────────────
find_dump() {
    local test_dir="$1" phase="$2"
    local pattern="$test_dir/quop_comm_info_${phase}_*.txt"
    # shellcheck disable=SC2086
    local files=( $pattern )
    if [[ ${#files[@]} -eq 0 || ! -f "${files[0]}" ]]; then
        echo ""
    else
        # Use the latest if multiple
        ls -t $pattern 2>/dev/null | head -1
    fi
}

# ══════════════════════════════════════════════════════════════════════
#  TIER 1 — Diagonal oracle (exact field comparison)
# ══════════════════════════════════════════════════════════════════════

# Test configs: test_id nodes rpn rpg binding partition system_size
TIER1_TESTS=(
    # Phase 1: single node
    "T01 1  8 1 sequential gpu 128"
    "T02 1  4 1 sequential gpu  64"
    "T03 1 16 1 sequential gpu 256"
    "T04 1 16 2 sequential gpu 256"
    "T07 1  1 1 sequential gpu  16"
    # Phase 2: multi-node
    "T08 2  8 1 sequential gpu 256"
    "T09 2 16 2 sequential gpu 512"
    # Phase 3: CPU-only
    "T13 1  4 1 auto work  64"
)

run_tier1() {
    log_info "═══ Tier 1: Diagonal oracle tests ═══"

    # Generate expected CSV
    log_info "Generating expected topology CSV..."
    python "$GENERATE" "$EXPECTED_CSV"
    echo ""

    for entry in "${TIER1_TESTS[@]}"; do
        # shellcheck disable=SC2086
        set -- $entry
        local test_id="$1" nodes="$2" rpn="$3" rpg="$4"
        local binding="$5" partition="$6" sys_size="$7"

        log_info "[$test_id] nodes=$nodes rpn=$rpn rpg=$rpg partition=$partition sys_size=$sys_size"

        local rc=0
        run_test "$test_id" "$nodes" "$rpn" "$rpg" "$binding" "$partition" \
                python "$DIAG_RUNNER" "$sys_size" || rc=$?
        if [[ $rc -eq 99 ]]; then continue; fi  # dry-run
        if [[ $rc -ne 0 ]]; then
            log_fail "$test_id — execution failed (exit $rc)"
            continue
        fi

        local dump
        dump="$(find_dump "$RESULTS_DIR/$test_id" "$PHASE")"
        if [[ -z "$dump" ]]; then
            log_fail "$test_id — no dump file for phase '$PHASE'"
            continue
        fi

        # Run exact comparison
        local cmp_out
        cmp_out="$(python "$VERIFY" compare "$EXPECTED_CSV" "$test_id" "$dump" 2>&1)" || true
        if echo "$cmp_out" | grep -q "All.*match\|0 differences"; then
            log_pass "$test_id"
        elif echo "$cmp_out" | grep -qi "mismatch\|difference\|expected"; then
            log_fail "$test_id"
            echo "$cmp_out" | head -20 | sed 's/^/    /'
        else
            # No mismatches reported
            log_pass "$test_id"
        fi
    done
    echo ""
}

# ══════════════════════════════════════════════════════════════════════
#  TIER 2 — Algorithm-class integration (consistency checks)
# ══════════════════════════════════════════════════════════════════════

# Test configs: test_id nodes rpn rpg binding partition algo_args...
TIER2_TESTS=(
    "A05 1  8 1 sequential gpu qwoa 128"
    "A06 1 16 1 sequential gpu qwoa 256"
    "A09 1  8 1 sequential gpu qmoa 4 4"
    "A01 1  8 1 sequential gpu qaoa 128"
    "A04 1  4 1 auto       work qaoa 64"
    "A08 1  4 1 auto       work qwoa 64"
    "A12 1  4 1 auto       work qmoa 3 3"
)

run_tier2() {
    log_info "═══ Tier 2: Algorithm-class integration tests ═══"

    for entry in "${TIER2_TESTS[@]}"; do
        # shellcheck disable=SC2086
        set -- $entry
        local test_id="$1" nodes="$2" rpn="$3" rpg="$4"
        local binding="$5" partition="$6"
        shift 6
        local algo_args="$*"

        log_info "[$test_id] nodes=$nodes rpn=$rpn rpg=$rpg partition=$partition algo=$algo_args"

        local rc=0
        run_test "$test_id" "$nodes" "$rpn" "$rpg" "$binding" "$partition" \
                python "$ALGO_RUNNER" $algo_args || rc=$?
        if [[ $rc -eq 99 ]]; then continue; fi  # dry-run
        if [[ $rc -ne 0 ]]; then
            log_fail "$test_id — execution failed (exit $rc)"
            continue
        fi

        local dump
        dump="$(find_dump "$RESULTS_DIR/$test_id" "$PHASE")"
        if [[ -z "$dump" ]]; then
            log_fail "$test_id — no dump file for phase '$PHASE'"
            continue
        fi

        # Run consistency checks
        local chk_out
        chk_out="$(python "$VERIFY" check "$dump" --rpg "$rpg" 2>&1)" || true
        if echo "$chk_out" | grep -qi "issue\|error\|fail"; then
            log_fail "$test_id"
            echo "$chk_out" | head -20 | sed 's/^/    /'
        else
            log_pass "$test_id"
        fi
    done
    echo ""
}

# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

main() {
    log_info "QuOp_QUISA Topology Test Suite"
    log_info "Launcher: $LAUNCHER"
    log_info "Results:  $RESULTS_DIR"
    log_info "Phase:    $PHASE"
    echo ""

    mkdir -p "$RESULTS_DIR"

    if [[ "$TIER" == "both" || "$TIER" == "1" ]]; then
        run_tier1
    fi

    if [[ "$TIER" == "both" || "$TIER" == "2" ]]; then
        run_tier2
    fi

    # Summary
    log_info "═══ Summary ═══"
    echo -e "  ${GREEN}Passed:${RESET}  $pass_count"
    echo -e "  ${RED}Failed:${RESET}  $fail_count"
    echo -e "  ${YELLOW}Skipped:${RESET} $skip_count"

    if [[ $fail_count -gt 0 ]]; then
        echo ""
        echo -e "${RED}Some tests failed. Check $RESULTS_DIR/<test_id>/stderr.log for details.${RESET}"
        exit 1
    fi
}

main
