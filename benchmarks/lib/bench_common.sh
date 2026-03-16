#!/usr/bin/env bash
# =============================================================================
# Shared helpers for QuOp_MPI benchmark scripts.
#
# Sourced by submit_scaling.sh after the profile activation script, so helpers
# from setup/lib/common.sh (info, step, require_command, etc.) are available.
# =============================================================================

# ---------------------------------------------------------------------------
# compute_size_metadata -- derive SYSTEM_SIZE, SIZE_SPEC, SIZE_TAG, SIZE_ARGS
#
# Input:  $ALGORITHM, $SIZE_ARG
# Output: SYSTEM_SIZE, SIZE_SPEC, SIZE_TAG, SIZE_ARGS (array),
#         SIZE_ARG_LITERAL
# ---------------------------------------------------------------------------
compute_size_metadata() {
    local algorithm="$1"
    local size_arg="$2"

    if [[ "$algorithm" == "qmoa" ]]; then
        read -r -a SIZE_ARGS <<< "$size_arg"
        if [[ ${#SIZE_ARGS[@]} -eq 0 ]]; then
            echo "Error: qmoa expects a quoted space-separated exponent list" >&2
            return 1
        fi
        local sum_ns=0
        for n in "${SIZE_ARGS[@]}"; do
            if ! [[ "$n" =~ ^[0-9]+$ ]]; then
                echo "Error: qmoa size arguments must be non-negative integers, got '$n'" >&2
                return 1
            fi
            sum_ns=$((sum_ns + n))
        done
        SYSTEM_SIZE=$((1 << sum_ns))
        SIZE_SPEC=$(IFS=x; echo "${SIZE_ARGS[*]}")
        SIZE_TAG="${SYSTEM_SIZE}_${SIZE_SPEC}"
    else
        if ! [[ "$size_arg" =~ ^[0-9]+$ ]]; then
            echo "Error: ${algorithm} expects a single integer size_arg, got '$size_arg'" >&2
            return 1
        fi
        SIZE_ARGS=("$size_arg")
        SYSTEM_SIZE="$size_arg"
        SIZE_SPEC="$SYSTEM_SIZE"
        SIZE_TAG="$SYSTEM_SIZE"
    fi

    SIZE_ARG_LITERAL=$(printf '%q' "$size_arg")
}

# ---------------------------------------------------------------------------
# mean_evolve_from_csv -- extract mean_evolve_s value from a result CSV
# ---------------------------------------------------------------------------
mean_evolve_from_csv() {
    awk -F, '
        NR == 1 {
            for (i = 1; i <= NF; ++i) {
                if ($i == "mean_evolve_s") {
                    col = i
                    break
                }
            }
            next
        }
        NR > 1 && col {
            print $col
            exit
        }
    ' "$1"
}

# ---------------------------------------------------------------------------
# append_program_wall_to_csv -- add or update program_wall_s column
# ---------------------------------------------------------------------------
append_program_wall_to_csv() {
    local csv_path="$1"
    local program_wall_s="$2"
    local results_dir
    results_dir="$(dirname "$csv_path")"
    local tmp_path
    tmp_path=$(mktemp "${results_dir}/program_wall.XXXXXX")

    awk -F, -v OFS=, -v wall="${program_wall_s}" '
        NR == 1 {
            wall_col = 0
            for (i = 1; i <= NF; ++i) {
                if ($i == "program_wall_s") {
                    wall_col = i
                    break
                }
            }
            if (wall_col) {
                print
            } else {
                print $0, "program_wall_s"
            }
            next
        }
        NR == 2 {
            if (wall_col) {
                $wall_col = wall
                print
            } else {
                print $0, wall
            }
            next
        }
        { print }
    ' "${csv_path}" > "${tmp_path}"
    mv "${tmp_path}" "${csv_path}"
}

# ---------------------------------------------------------------------------
# check_early_stop -- detect slowdown and write .stop file
#
# Returns 0 (stop) or 1 (continue).
# ---------------------------------------------------------------------------
check_early_stop() {
    local prev_csv="$1"
    local curr_csv="$2"
    local stop_file="$3"
    local nprocs="$4"

    if [[ ! -f "$prev_csv" || ! -f "$curr_csv" ]]; then
        return 1
    fi

    local prev_evolve curr_evolve slowdown
    prev_evolve=$(mean_evolve_from_csv "$prev_csv")
    curr_evolve=$(mean_evolve_from_csv "$curr_csv")

    slowdown=$(awk -v c="$curr_evolve" -v p="$prev_evolve" 'BEGIN { print (c >= p) ? 1 : 0 }')
    if [[ "$slowdown" -eq 1 ]]; then
        echo "Slowdown detected: curr=${curr_evolve} >= prev=${prev_evolve}"
        echo "nprocs=${nprocs}" > "$stop_file"
        echo "Stop file written: ${stop_file}"
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# clean_results -- remove ephemeral benchmark artefacts
# ---------------------------------------------------------------------------
clean_results() {
    local bench_dir="$1"
    local results_dir="${bench_dir}/results"
    local scripts_dir="${bench_dir}/scripts"

    echo "Removing .stop files, per-job CSVs, and saved scripts from ${results_dir}/ and ${scripts_dir}/"
    rm -fv "${results_dir}"/*.stop "${results_dir}"/*.out "${results_dir}"/*.err "${results_dir}"/*.time 2>/dev/null || true
    rm -fv "${scripts_dir}"/*.sh 2>/dev/null || true
    find "${results_dir}" -maxdepth 1 -name '*.csv' ! -name 'summary.csv' -print -delete 2>/dev/null || true
    echo "Clean complete."
}

