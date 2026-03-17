#!/bin/bash --login
#
# QuOp_MPI Unified Scaling Benchmark
#
# Requires the profile environment to be activated first:
#
#   source /path/to/activate-<profile>-<backend>.sh
#
# Usage:
#   submit_scaling.sh [--account ACCOUNT] <algorithm> <size_arg> <phase> [--verify]
#   submit_scaling.sh clean
#
# size_arg can be:
#   - A single value:               1048576
#   - Comma-separated values:       65536,131072,262144,524288,1048576
#   - "all" to use profile defaults: all
#   - For qmoa, comma separates dimension configs:
#       "3 3 3 3 3"                  (single)
#       "3 3 3 3 3,4 4 4 4"         (multiple)
#
# When multiple sizes are given, each size runs its full scaling sequence
# before the next size begins.  In SLURM mode the dependency chain spans
# across sizes.
#
# Config structure:
#   [benchmarks]                    shared keys (scheduler, partition, etc.)
#   [benchmarks.<algo>]             per-algorithm overrides + args_intra/args_multi
#
# Per-algorithm keys inherit from [benchmarks] and can be overridden in the
# sub-table.  For example [benchmarks.qaoa] may define its own ranks_per_node.
#
# Examples (SLURM cluster):
#   source /scratch/pawsey0123/quop/activate-pawsey-setonix-wavefront.sh
#   $QUOP_BENCHMARKS_DIR/submit_scaling.sh --account pawsey0123 qwoa 1048576 intra
#   $QUOP_BENCHMARKS_DIR/submit_scaling.sh --account pawsey0123 qaoa all intra
#   $QUOP_BENCHMARKS_DIR/submit_scaling.sh --account pawsey0123 qaoa 65536,131072 intra
#
# Examples (local workstation):
#   source ~/quop-install/activate-ubuntu-24-mpi.sh
#   $QUOP_BENCHMARKS_DIR/submit_scaling.sh qaoa 65536 intra
#   $QUOP_BENCHMARKS_DIR/submit_scaling.sh qaoa all intra
#
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${BENCH_DIR}/results"
SCRIPTS_DIR="${BENCH_DIR}/scripts"

# Source shared helpers
# shellcheck source=lib/bench_common.sh
source "${BENCH_DIR}/lib/bench_common.sh"

# ---------------------------------------------------------------------------
# Verify the profile environment is activated
# ---------------------------------------------------------------------------
if [[ -z "${QUOP_PROFILE:-}" ]]; then
    echo "Error: QUOP_PROFILE is not set." >&2
    echo "       Source the activation script first:" >&2
    echo "       source /path/to/activate-<profile>-<backend>.sh" >&2
    exit 1
fi

if [[ -z "${QUOP_BACKEND:-}" ]]; then
    echo "Error: QUOP_BACKEND is not set." >&2
    exit 1
fi

if [[ -z "${CONFIG_FILE:-}" ]]; then
    echo "Error: CONFIG_FILE is not set." >&2
    exit 1
fi

BACKEND="${QUOP_BACKEND}"
PROFILE="${QUOP_PROFILE}"

# ---------------------------------------------------------------------------
# Load benchmark config from the profile's config.toml
# ---------------------------------------------------------------------------
if [[ -z "${CONFIG_RENDERER:-}" ]]; then
    echo "Error: CONFIG_RENDERER is not set. Is the activation script up to date?" >&2
    exit 1
fi

eval "$("${CONFIG_PYTHON:-python}" "$CONFIG_RENDERER" "$CONFIG_FILE")"

if [[ -z "${CFG_BENCHMARKS_SCHEDULER:-}" ]]; then
    echo "Error: This profile does not define benchmark parameters." >&2
    echo "       Add a [benchmarks] section to config.toml." >&2
    exit 1
fi

SCHEDULER="${CFG_BENCHMARKS_SCHEDULER}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 [--account ACCOUNT] <algorithm> <size_arg> <phase> [--verify]"
    echo "       $0 clean"
    echo ""
    echo "The profile environment must be activated first."
    echo "  algorithm : qaoa | qwoa | qmoa"
    echo "  size_arg  : system size, comma-separated sizes, or 'all' for profile defaults"
    echo "              qaoa/qwoa: integer(s)         e.g. 1048576 or 65536,131072,262144"
    echo "              qmoa:     exponent list(s)    e.g. \"3 3 3 3 3\" or \"3 3 3,4 4 4 4\""
    echo "              'all':    use sizes from [benchmarks.<algo>].args_<phase> in config.toml"
    echo "  phase     : intra | multi"
    echo ""
    echo "Options:"
    echo "  --account ACCOUNT   SLURM account name (required for scheduler=slurm)"
    echo "  --verify            Record state_norm and expectation_value"
    echo "  clean               Remove benchmark artefacts"
}

# Handle clean subcommand
if [[ "${1:-}" == "clean" ]]; then
    clean_results "$BENCH_DIR"
    exit 0
fi

# Extract flags and positional args
VERIFY_FLAG=""
ACCOUNT=""
declare -a POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify)
            VERIFY_FLAG="--verify"
            shift
            ;;
        --account)
            if [[ $# -lt 2 ]]; then
                echo "Error: --account requires a value" >&2
                exit 1
            fi
            ACCOUNT="$2"
            shift 2
            ;;
        --account=*)
            ACCOUNT="${1#--account=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL[@]} -ne 3 ]]; then
    usage >&2
    exit 1
fi

ALGORITHM="${POSITIONAL[0]}"
SIZE_ARG="${POSITIONAL[1]}"
PHASE="${POSITIONAL[2]}"

# Validate algorithm
case "$ALGORITHM" in
    qaoa|qwoa|qmoa) ;;
    *)
        echo "Error: algorithm must be one of qaoa, qwoa, qmoa; got '$ALGORITHM'" >&2
        exit 1
        ;;
esac

# Validate phase
case "$PHASE" in
    intra|multi) ;;
    *)
        echo "Error: phase must be 'intra' or 'multi', got '$PHASE'" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Per-algorithm config resolution helpers
# ---------------------------------------------------------------------------
# Config uses [benchmarks] for shared keys and [benchmarks.<algo>] for
# per-algorithm overrides.  Variables are named:
#   CFG_BENCHMARKS_<KEY>              (shared)
#   CFG_BENCHMARKS_<ALGO>_<KEY>       (per-algorithm override)
#
# resolve_scalar KEY DEFAULT  -> echoes the resolved value
# resolve_array  TARGET KEY   -> populates the TARGET array
# ---------------------------------------------------------------------------
ALGO_UPPER="$(echo "$ALGORITHM" | tr '[:lower:]' '[:upper:]')"

resolve_scalar() {
    local key_upper="$1"
    local default="${2:-}"
    local algo_var="CFG_BENCHMARKS_${ALGO_UPPER}_${key_upper}"
    local shared_var="CFG_BENCHMARKS_${key_upper}"
    if [[ -n "${!algo_var+x}" ]]; then
        echo "${!algo_var}"
    elif [[ -n "${!shared_var+x}" ]]; then
        echo "${!shared_var}"
    else
        echo "$default"
    fi
}

resolve_array() {
    local _target="$1"
    local key_upper="$2"
    local algo_var="CFG_BENCHMARKS_${ALGO_UPPER}_${key_upper}"
    local shared_var="CFG_BENCHMARKS_${key_upper}"
    if declare -p "$algo_var" &>/dev/null; then
        eval "$_target=(\"\${${algo_var}[@]}\")"
    elif declare -p "$shared_var" &>/dev/null; then
        eval "$_target=(\"\${${shared_var}[@]}\")"
    else
        eval "$_target=()"
    fi
}

# ---------------------------------------------------------------------------
# Build the list of sizes to benchmark
# ---------------------------------------------------------------------------
declare -a SIZE_LIST=()

if [[ "$SIZE_ARG" == "all" ]]; then
    PHASE_UPPER="$(echo "$PHASE" | tr '[:lower:]' '[:upper:]')"
    resolve_array SIZE_LIST "ARGS_${PHASE_UPPER}"
    if [[ ${#SIZE_LIST[@]} -eq 0 ]]; then
        echo "Error: [benchmarks.${ALGORITHM}].args_${PHASE} not defined -- cannot use 'all'" >&2
        exit 1
    fi
else
    # Split by comma (supports both "65536,131072" and "3 3 3,4 4 4 4")
    IFS=',' read -r -a SIZE_LIST <<< "$SIZE_ARG"
fi

if [[ ${#SIZE_LIST[@]} -eq 0 ]]; then
    echo "Error: no sizes to benchmark" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}" "${SCRIPTS_DIR}"

# ---------------------------------------------------------------------------
# Build process-count sequences from profile config
# ---------------------------------------------------------------------------
declare -a NPROCS_SEQ=()
declare -a NODES_SEQ=()
declare -a GPUS_PER_NODE_SEQ=()

if [[ "$BACKEND" == "wavefront" ]]; then
    RANKS_PER_GCD="$(resolve_scalar RANKS_PER_GCD 8)"
    GCDS_PER_NODE="$(resolve_scalar GCDS_PER_NODE 8)"

    declare -a _intra_seq=()
    resolve_array _intra_seq INTRA_SEQUENCE
    declare -a _multi_nodes=()
    resolve_array _multi_nodes MULTI_NODES

    if [[ "$PHASE" == "intra" ]]; then
        if [[ ${#_intra_seq[@]} -eq 0 ]]; then
            echo "Error: intra_sequence not defined in [benchmarks]" >&2
            exit 1
        fi
        for g in "${_intra_seq[@]}"; do
            NPROCS_SEQ+=("$((RANKS_PER_GCD * g))")
            NODES_SEQ+=("1")
            GPUS_PER_NODE_SEQ+=("$g")
        done
    else
        if [[ ${#_multi_nodes[@]} -eq 0 ]]; then
            echo "Error: multi_nodes not defined in [benchmarks]" >&2
            exit 1
        fi
        for n in "${_multi_nodes[@]}"; do
            NPROCS_SEQ+=("$((RANKS_PER_GCD * GCDS_PER_NODE * n))")
            NODES_SEQ+=("$n")
            GPUS_PER_NODE_SEQ+=("$GCDS_PER_NODE")
        done
    fi
else
    RANKS_PER_NODE="$(resolve_scalar RANKS_PER_NODE 4)"

    declare -a _intra_seq=()
    resolve_array _intra_seq INTRA_SEQUENCE
    declare -a _multi_nodes=()
    resolve_array _multi_nodes MULTI_NODES

    if [[ "$PHASE" == "intra" ]]; then
        if [[ ${#_intra_seq[@]} -eq 0 ]]; then
            echo "Error: intra_sequence not defined in [benchmarks]" >&2
            exit 1
        fi
        for p in "${_intra_seq[@]}"; do
            NPROCS_SEQ+=("$p")
            NODES_SEQ+=("1")
        done
    else
        if [[ ${#_multi_nodes[@]} -eq 0 ]]; then
            echo "Error: multi_nodes not defined in [benchmarks]" >&2
            exit 1
        fi
        for n in "${_multi_nodes[@]}"; do
            NPROCS_SEQ+=("$((RANKS_PER_NODE * n))")
            NODES_SEQ+=("$n")
        done
    fi
fi

# ---------------------------------------------------------------------------
# SLURM account validation
# ---------------------------------------------------------------------------
if [[ "$SCHEDULER" == "slurm" && -z "$ACCOUNT" ]]; then
    echo "Error: --account is required for SLURM scheduler" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Determine partition and account
# ---------------------------------------------------------------------------
if [[ "$SCHEDULER" == "slurm" ]]; then
    PARTITION="$(resolve_scalar PARTITION work)"
    ACCOUNT_SUFFIX="$(resolve_scalar ACCOUNT_SUFFIX "")"
    SLURM_ACCOUNT="${ACCOUNT}${ACCOUNT_SUFFIX}"
    WALLTIME="$(resolve_scalar WALLTIME "00:20:00")"
fi

# ---------------------------------------------------------------------------
# Resolve exports and srun extra args (per-algorithm with fallback)
# ---------------------------------------------------------------------------
declare -a BENCH_EXPORTS=()
resolve_array BENCH_EXPORTS EXPORTS

SRUN_EXTRA_ARGS="$(resolve_scalar SRUN_EXTRA_ARGS "")"

# ---------------------------------------------------------------------------
# Dispatch: SLURM or local
# ---------------------------------------------------------------------------

# ========================================================================
# SLURM mode: submit a dependency chain of batch jobs
#
# Uses global PREV_JOBID so dependency chains span across sizes.
# ========================================================================
PREV_JOBID=""

submit_slurm_chain() {

    for i in "${!NPROCS_SEQ[@]}"; do
        local np="${NPROCS_SEQ[$i]}"
        local nn="${NODES_SEQ[$i]}"
        local ntasks_per_node=$((np / nn))
        local csv_file="${ALGORITHM}_${BACKEND}_${SIZE_TAG}_${PHASE}_${np}.csv"

        # Skip if results already exist
        if [[ -f "${RESULTS_DIR}/${csv_file}" ]]; then
            echo "SKIP: ${csv_file} already exists"
            continue
        fi

        local prev_np=""
        if [[ $i -gt 0 ]]; then
            prev_np="${NPROCS_SEQ[$((i - 1))]}"
        fi

        local job_name="quop_${ALGORITHM}_${BACKEND}_${SIZE_TAG}_${np}"
        local stop_file="${RESULTS_DIR}/${ALGORITHM}_${BACKEND}_${SIZE_TAG}_${PHASE}.stop"

        # Build SBATCH headers
        local sbatch_headers
        sbatch_headers="#!/bin/bash --login
#SBATCH --job-name=${job_name}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --nodes=${nn}
#SBATCH --time=${WALLTIME}
#SBATCH --output=${RESULTS_DIR}/${job_name}_%j.out
#SBATCH --error=${RESULTS_DIR}/${job_name}_%j.err
#SBATCH --exclusive"

        if [[ "$BACKEND" == "wavefront" ]]; then
            local ng="${GPUS_PER_NODE_SEQ[$i]}"
            sbatch_headers+="
#SBATCH --gres=gpu:${ng}"
        else
            sbatch_headers+="
#SBATCH --ntasks=${np}
#SBATCH --ntasks-per-node=${ntasks_per_node}
#SBATCH --cpus-per-task=1"
        fi

        # Build environment exports block
        local env_exports=""

        if [[ ${#BENCH_EXPORTS[@]} -gt 0 ]]; then
            for exp in "${BENCH_EXPORTS[@]}"; do
                env_exports+="export ${exp}
"
            done
        fi

        # Build srun command
        local srun_cmd="srun --export=ALL -N ${nn} -n ${np} -c 1"
        if [[ "$BACKEND" == "wavefront" ]]; then
            local ng="${GPUS_PER_NODE_SEQ[$i]}"
            srun_cmd+=" --gres=gpu:${ng}"
        fi
        if [[ -n "${SRUN_EXTRA_ARGS}" ]]; then
            srun_cmd+=" ${SRUN_EXTRA_ARGS}"
        fi

        # Build the complete job script
        local job_script
        job_script="${sbatch_headers}

# Early-stop check
if [[ -f \"${stop_file}\" ]]; then
    echo \"Stop file found (${stop_file}), exiting.\"
    exit 0
fi

# Activate the profile environment
source \"${QUOP_INSTALL_ROOT}/activate-${PROFILE}-${BACKEND}.sh\"

export OMP_NUM_THREADS=1
${env_exports}
SIZE_ARG=${SIZE_ARG_LITERAL}

$(declare -f mean_evolve_from_csv)

$(declare -f append_program_wall_to_csv)

$(declare -f check_early_stop)

# Early-stop: check if previous step showed a slowdown
PREV_CSV=\"${RESULTS_DIR}/${ALGORITHM}_${BACKEND}_${SIZE_TAG}_${PHASE}_${prev_np}.csv\"
if [[ -n \"${prev_np}\" && -f \"\${PREV_CSV}\" ]]; then
    PREV_EVOLVE=\$(mean_evolve_from_csv \"\${PREV_CSV}\")
    echo \"Previous step (nprocs=${prev_np}) mean_evolve_s = \${PREV_EVOLVE}\"
fi

# Run benchmark
BENCH_CMD=(python -u \"${BENCH_DIR}/bench.py\" \"${ALGORITHM}\" \"\${SIZE_ARG}\" --phase \"${PHASE}\")
if [[ -n \"${VERIFY_FLAG}\" ]]; then
    BENCH_CMD+=(\"${VERIFY_FLAG}\")
fi
TIME_FILE=\"${RESULTS_DIR}/${csv_file%.csv}.time\"
SRUN_CMD=(${srun_cmd} \"\${BENCH_CMD[@]}\")
TIMEFORMAT='%R'
{ time \"\${SRUN_CMD[@]}\" 2>&3 ; } 3>&2 2>\"\${TIME_FILE}\"
PROGRAM_WALL_S=\$(tr -d '[:space:]' < \"\${TIME_FILE}\")
echo \"Program wall time = \${PROGRAM_WALL_S}s\"
CSV_PATH=\"${RESULTS_DIR}/${csv_file}\"
if [[ -f \"\${CSV_PATH}\" ]]; then
    append_program_wall_to_csv \"\${CSV_PATH}\" \"\${PROGRAM_WALL_S}\"
fi

# Post-run early-stop check
if [[ -n \"${prev_np}\" ]]; then
    check_early_stop \"\${PREV_CSV}\" \"\${CSV_PATH}\" \"${stop_file}\" \"${np}\"
fi"

        # Save job script for debugging
        local script_path="${SCRIPTS_DIR}/${job_name}.sh"
        printf '%s\n' "$job_script" > "$script_path"
        chmod +x "$script_path"

        # Submit with dependency if applicable
        local sbatch_args=""
        if [[ -n "$PREV_JOBID" ]]; then
            sbatch_args="--dependency=afterok:${PREV_JOBID}"
        fi

        local jobid
        jobid=$(sbatch $sbatch_args --parsable "$script_path")
        echo "Submitted ${job_name} -> JobID ${jobid} (nprocs=${np}, nodes=${nn})"
        PREV_JOBID="$jobid"
    done

    echo ""
    echo "All jobs submitted for ${ALGORITHM} / ${BACKEND} / system_size=${SYSTEM_SIZE} / size_spec=${SIZE_SPEC} / phase=${PHASE}"
}

# ========================================================================
# Local mode: run benchmarks sequentially with mpiexec
# ========================================================================
run_local_benchmarks() {
    local mpiexec_cmd
    mpiexec_cmd="$(command -v mpiexec 2>/dev/null || command -v mpirun 2>/dev/null || true)"
    if [[ -z "$mpiexec_cmd" ]]; then
        echo "Error: neither mpiexec nor mpirun found in PATH" >&2
        exit 1
    fi

    echo "Running benchmarks locally with ${mpiexec_cmd}"
    echo "Profile: ${PROFILE}, Backend: ${BACKEND}"
    echo "Algorithm: ${ALGORITHM}, Size: ${CURRENT_SIZE}, Phase: ${PHASE}"
    echo ""

    local stop_file="${RESULTS_DIR}/${ALGORITHM}_${BACKEND}_${SIZE_TAG}_${PHASE}.stop"

    for i in "${!NPROCS_SEQ[@]}"; do
        local np="${NPROCS_SEQ[$i]}"
        local csv_file="${ALGORITHM}_${BACKEND}_${SIZE_TAG}_${PHASE}_${np}.csv"

        # Check for stop file
        if [[ -f "$stop_file" ]]; then
            echo "Stop file found (${stop_file}), skipping remaining runs."
            break
        fi

        # Skip if results already exist
        if [[ -f "${RESULTS_DIR}/${csv_file}" ]]; then
            echo "SKIP: ${csv_file} already exists"
            continue
        fi

        local prev_np=""
        if [[ $i -gt 0 ]]; then
            prev_np="${NPROCS_SEQ[$((i - 1))]}"
        fi

        echo "--- Running with nprocs=${np} ---"

        # Build command
        local -a bench_cmd=(python -u "${BENCH_DIR}/bench.py" "${ALGORITHM}" "${CURRENT_SIZE}" --phase "${PHASE}")
        if [[ -n "${VERIFY_FLAG}" ]]; then
            bench_cmd+=("${VERIFY_FLAG}")
        fi

        export OMP_NUM_THREADS=1

        # Apply benchmark exports
        if [[ ${#BENCH_EXPORTS[@]} -gt 0 ]]; then
            for exp in "${BENCH_EXPORTS[@]}"; do
                export "${exp?}"
            done
        fi

        local time_file="${RESULTS_DIR}/${csv_file%.csv}.time"
        local start_time end_time elapsed

        start_time=$(date +%s.%N 2>/dev/null || date +%s)
        "$mpiexec_cmd" -n "$np" "${bench_cmd[@]}"
        local rc=$?
        end_time=$(date +%s.%N 2>/dev/null || date +%s)

        if [[ $rc -ne 0 ]]; then
            echo "Error: benchmark failed with exit code $rc (nprocs=$np)" >&2
            continue
        fi

        elapsed=$(awk -v e="$end_time" -v s="$start_time" 'BEGIN { printf "%.3f", e - s }')
        echo "$elapsed" > "$time_file"
        echo "Program wall time = ${elapsed}s"

        local csv_path="${RESULTS_DIR}/${csv_file}"
        if [[ -f "$csv_path" ]]; then
            append_program_wall_to_csv "$csv_path" "$elapsed"
        fi

        # Early-stop check
        if [[ -n "$prev_np" ]]; then
            local prev_csv="${RESULTS_DIR}/${ALGORITHM}_${BACKEND}_${SIZE_TAG}_${PHASE}_${prev_np}.csv"
            if check_early_stop "$prev_csv" "$csv_path" "$stop_file" "$np"; then
                echo "Stopping early due to slowdown."
                break
            fi
        fi

        echo ""
    done

    echo "All benchmarks complete for ${ALGORITHM} / ${BACKEND} / system_size=${SYSTEM_SIZE} / size_spec=${SIZE_SPEC} / phase=${PHASE}"
}

# ---------------------------------------------------------------------------
# Run the appropriate mode -- loop over all requested sizes
# ---------------------------------------------------------------------------
for CURRENT_SIZE in "${SIZE_LIST[@]}"; do
    # Trim whitespace from edges (comma-split can leave spaces).
    # NOTE: CURRENT_SIZE is the raw user/config input for this iteration.
    # compute_size_metadata (below) derives SYSTEM_SIZE, SIZE_SPEC,
    # SIZE_TAG, and SIZE_ARG_LITERAL from it.  The SLURM heredoc uses
    # SIZE_ARG_LITERAL (the shell-safe form) rather than CURRENT_SIZE.
    CURRENT_SIZE="$(echo -n "$CURRENT_SIZE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -z "$CURRENT_SIZE" ]] && continue

    compute_size_metadata "$ALGORITHM" "$CURRENT_SIZE"

    echo "=== Size: ${CURRENT_SIZE} (system_size=${SYSTEM_SIZE}, size_spec=${SIZE_SPEC}) ==="

    if [[ "$SCHEDULER" == "slurm" ]]; then
        submit_slurm_chain
    elif [[ "$SCHEDULER" == "local" ]]; then
        run_local_benchmarks
    else
        echo "Error: unknown scheduler '$SCHEDULER' (expected 'slurm' or 'local')" >&2
        exit 1
    fi

    echo ""
done
