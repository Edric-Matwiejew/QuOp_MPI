#!/usr/bin/env bash

COMMON_SH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_RENDERER="${CONFIG_RENDERER:-$COMMON_SH_DIR/render_config.py}"
PATH_HELPER="${PATH_HELPER:-$COMMON_SH_DIR/path_helper.py}"

info() { echo "==> $*"; }
step() { printf '\n===> %s\n\n' "$*"; }

# ---- Shared discovery helpers ------------------------------------------------
# These were extracted from the Ubuntu and Setonix hooks where they were
# duplicated.  Profile hooks can override or extend behaviour by wrapping these.

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command '$cmd' not found in PATH"
        return 1
    fi
}

resolve_command() {
    command -v "$1" 2>/dev/null || true
}

prefix_from_executable() {
    local exe
    exe="$(command -v "$1" 2>/dev/null || true)"
    if [[ -z "$exe" ]]; then
        return 1
    fi
    (cd "$(dirname "$exe")/.." && pwd)
}

prefix_from_pkgconfig() {
    local package
    for package in "$@"; do
        if pkg-config --exists "$package" 2>/dev/null; then
            pkg-config --variable=prefix "$package"
            return 0
        fi
    done
    return 1
}

find_rocm_path() {
    if [[ -n "${ROCM_PATH:-}" ]]; then
        printf '%s\n' "$ROCM_PATH"
        return 0
    fi
    if command -v hipconfig >/dev/null 2>&1; then
        local rocm_path
        rocm_path="$(hipconfig --path 2>/dev/null || true)"
        if [[ -n "$rocm_path" && -d "$rocm_path" ]]; then
            printf '%s\n' "$rocm_path"
            return 0
        fi
    fi
    prefix_from_executable hipcc
}

resolve_shafft_path() {
    if [[ -n "${SHAFFT_PATH:-}" ]]; then
        printf '%s\n' "$SHAFFT_PATH"
        return 0
    fi
    if [[ -n "${CFG_SHAFFT_PATH:-}" ]]; then
        printf '%s\n' "$CFG_SHAFFT_PATH"
        return 0
    fi
    if [[ -n "${FETCHCONTENT_BASE_DIR:-}" ]]; then
        printf '%s/shafft-install\n' "$FETCHCONTENT_BASE_DIR"
        return 0
    fi
    if [[ -n "${QUOP_FETCHCONTENT_BASE_DIR:-}" ]]; then
        printf '%s/shafft-install\n' "$QUOP_FETCHCONTENT_BASE_DIR"
        return 0
    fi
    return 1
}

find_hdf5_dir() {
    if [[ -n "${HDF5_DIR:-}" ]]; then
        printf '%s\n' "$HDF5_DIR"
        return 0
    fi
    prefix_from_executable h5pcc ||
        prefix_from_executable h5cc ||
        prefix_from_pkgconfig hdf5-openmpi hdf5
}

find_fftw_root() {
    if [[ -n "${FFTW_ROOT:-}" ]]; then
        printf '%s\n' "$FFTW_ROOT"
        return 0
    fi
    if [[ -n "${FFTW_DIR:-}" ]]; then
        printf '%s\n' "$FFTW_DIR"
        return 0
    fi
    prefix_from_pkgconfig fftw3 ||
        prefix_from_executable fftw-wisdom
}

join_by_delimiter() {
    local delimiter="$1"
    shift
    local IFS="$delimiter"
    printf '%s\n' "$*"
}

# ---- End shared discovery helpers --------------------------------------------

apply_profile_defaults() {
    PROFILE_DESCRIPTION="${PROFILE_DESCRIPTION:-${CFG_PROFILE_DESCRIPTION:-}}"
    WAVEFRONT_SUPPORTED="${WAVEFRONT_SUPPORTED:-${CFG_WAVEFRONT_SUPPORTED:-false}}"
    PYTHON_VERSION="${PYTHON_VERSION:-${CFG_PYTHON_VERSION:-}}"
}

available_profile_names() {
    local profiles_dir="${PROFILES_DIR:-${SITES_DIR:-}}"
    local profile_dir
    local nullglob_was_set=0
    local -a profile_names=()

    if [[ -z "$profiles_dir" || ! -d "$profiles_dir" ]]; then
        return 0
    fi

    if shopt -q nullglob; then
        nullglob_was_set=1
    fi
    shopt -s nullglob
    for profile_dir in "$profiles_dir"/*; do
        [[ -d "$profile_dir" ]] || continue
        profile_names+=("$(basename "$profile_dir")")
    done
    if ((nullglob_was_set == 0)); then
        shopt -u nullglob
    fi

    if ((${#profile_names[@]} == 0)); then
        return 0
    fi

    printf '%s\n' "${profile_names[@]}" | sort
}

list_available_profiles() {
    local profile_name
    local joined=""

    while IFS= read -r profile_name; do
        [[ -z "$profile_name" ]] && continue
        if [[ -n "$joined" ]]; then
            joined+=", "
        fi
        joined+="$profile_name"
    done < <(available_profile_names)

    printf '%s\n' "$joined"
}

print_available_profiles() {
    local profile_name

    while IFS= read -r profile_name; do
        [[ -z "$profile_name" ]] && continue
        printf '  %s\n' "$profile_name"
    done < <(available_profile_names)
}

# Backward-compatible aliases for the previous site-based naming.
available_site_names() {
    available_profile_names "$@"
}

list_available_sites() {
    list_available_profiles "$@"
}

print_available_sites() {
    print_available_profiles "$@"
}

ensure_module_command() {
    if command -v module >/dev/null 2>&1; then
        return 0
    fi

    if [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then
        # shellcheck disable=SC1090
        source "${MODULESHOME}/init/bash"
    elif [[ -f /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    elif [[ -f /etc/profile.d/lmod.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/lmod.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    fi

    command -v module >/dev/null 2>&1
}

load_module_list() {
    local array_name="$1"
    local -a modules_ref=()

    if ! declare -p "$array_name" >/dev/null 2>&1; then
        return 0
    fi

    eval "modules_ref=(\"\${${array_name}[@]}\")"

    if ((${#modules_ref[@]} == 0)); then
        return 0
    fi

    if ! ensure_module_command; then
        echo "Error: environment modules are not available, but profile requested module loads"
        return 1
    fi

    module load "${modules_ref[@]}"
}

run_profile_environment_hooks() {
    # Load common modules from config by default.  If a profile defines
    # profile_load_modules it takes full control of common module loading.
    if declare -f profile_load_modules >/dev/null 2>&1; then
        profile_load_modules || return 1
    elif declare -p CFG_PROFILE_MODULES_COMMON >/dev/null 2>&1 && \
         (( ${#CFG_PROFILE_MODULES_COMMON[@]} > 0 )); then
        load_module_list CFG_PROFILE_MODULES_COMMON || return 1
    fi

    # Load wavefront modules from config by default.  If a profile defines
    # profile_load_modules_gpu it takes full control of GPU module loading.
    if [[ "${BACKEND:-mpi}" == "wavefront" ]]; then
        if declare -f profile_load_modules_gpu >/dev/null 2>&1; then
            profile_load_modules_gpu || return 1
        elif declare -p CFG_PROFILE_MODULES_WAVEFRONT >/dev/null 2>&1 && \
             (( ${#CFG_PROFILE_MODULES_WAVEFRONT[@]} > 0 )); then
            load_module_list CFG_PROFILE_MODULES_WAVEFRONT || return 1
        fi
    fi

    if declare -f profile_post_modules_env >/dev/null 2>&1; then
        profile_post_modules_env || return 1
    fi
}

prepare_profile_venv_args() {
    VENV_ARGS=()
    if declare -f profile_configure_venv >/dev/null 2>&1; then
        profile_configure_venv || return 1
    fi
}

collect_profile_cmake_args() {
    local -a extra_args=()
    local line

    CMAKE_ARGS_ARRAY=(-DBUILD_TESTING=ON)
    if declare -f profile_cmake_args >/dev/null 2>&1; then
        while IFS= read -r line; do
            [[ "$line" =~ ^[[:space:]]*$ ]] && continue
            extra_args+=("$line")
        done < <(profile_cmake_args)
        CMAKE_ARGS_ARRAY+=("${extra_args[@]}")
    fi
}

validate_profile_cmake_args() {
    local expected_wavefront_flag="-DWAVEFRONT_BACKEND=OFF"

    if [[ "${BACKEND:-mpi}" == "wavefront" ]]; then
        expected_wavefront_flag="-DWAVEFRONT_BACKEND=ON"
    fi

    if [[ ! " ${CMAKE_ARGS_ARRAY[*]} " =~ [[:space:]]${expected_wavefront_flag}[[:space:]] ]]; then
        echo "Error: profile_cmake_args did not provide expected backend flag: ${expected_wavefront_flag}" >&2
        echo "       Backend was: ${BACKEND:-mpi}" >&2
        printf '       CMAKE_ARGS was: %s\n' "${CMAKE_ARGS_ARRAY[*]}" >&2
        return 1
    fi
}

export_profile_cmake_args() {
    collect_profile_cmake_args || return 1
    validate_profile_cmake_args || return 1

    # Intentionally use CMAKE_ARGS (space-separated) for scikit-build-core.
    # SKBUILD_CMAKE_ARGS is semicolon-split and can corrupt values that include
    # CMake list separators (for example: -DCMAKE_PREFIX_PATH=a;b;c).
    CMAKE_ARGS_STRING="${CMAKE_ARGS_ARRAY[*]}"
    export CMAKE_ARGS="$CMAKE_ARGS_STRING"
}

resolve_python_executable() {
    local candidate="$1"

    if [[ -z "$candidate" ]]; then
        return 1
    fi

    if [[ "$candidate" == */* ]]; then
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
        return 1
    fi

    command -v "$candidate" 2>/dev/null
}

python_major_minor() {
    local python_bin="$1"

    "$python_bin" - <<'PY'
import sys

print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
}

require_python_interpreter_version() {
    local python_bin="$1"
    local requested_version="$2"
    local context="${3:-Python interpreter}"
    local resolved_version

    if [[ -z "$requested_version" ]]; then
        return 0
    fi

    if ! resolved_version="$(python_major_minor "$python_bin" 2>/dev/null)"; then
        echo "Error: could not determine the version for $context '$python_bin'"
        return 1
    fi

    if [[ "$resolved_version" != "$requested_version" ]]; then
        echo "Error: $context '$python_bin' is Python $resolved_version, expected $requested_version"
        return 1
    fi
}

resolve_python_interpreter() {
    local requested_version="${1:-}"
    local requested_major="${requested_version%%.*}"
    local override="${CONFIG_PYTHON:-}"
    local -a candidates=()
    local candidate
    local candidate_path
    local seen=""

    if [[ -n "$override" ]]; then
        if ! candidate_path="$(resolve_python_executable "$override")"; then
            echo "Error: CONFIG_PYTHON '$override' is not executable"
            return 1
        fi

        require_python_interpreter_version "$candidate_path" "$requested_version" "CONFIG_PYTHON" || return 1
        printf '%s\n' "$candidate_path"
        return 0
    fi

    if [[ -n "$requested_version" ]]; then
        candidates=("python${requested_version}" "python${requested_major}" python3 python)
    else
        candidates=(python3 python)
    fi

    for candidate in "${candidates[@]}"; do
        if [[ -z "$candidate" || "$seen" == *"|${candidate}|"* ]]; then
            continue
        fi
        seen="${seen}|${candidate}|"

        if ! candidate_path="$(resolve_python_executable "$candidate")"; then
            continue
        fi

        if ! require_python_interpreter_version "$candidate_path" "$requested_version" "Python interpreter" >/dev/null 2>&1; then
            continue
        fi

        printf '%s\n' "$candidate_path"
        return 0
    done

    if [[ -n "$requested_version" ]]; then
        echo "Error: could not find a Python interpreter matching version '$requested_version'"
    else
        echo "Error: could not find a usable Python interpreter"
    fi
    return 1
}

run_path_helper() {
    local python_bin="${CONFIG_PYTHON:-python3}"

    if [[ ! -f "$PATH_HELPER" ]]; then
        echo "Error: path helper '$PATH_HELPER' not found"
        return 1
    fi

    "$python_bin" "$PATH_HELPER" "$@"
}

canonicalize_path() {
    local path_value="$1"

    run_path_helper canonicalize "$path_value"
}

expand_install_path() {
    local path_template="$1"
    local base_dir="$2"

    run_path_helper expand \
        "$path_template" \
        "$base_dir" \
        "$PROJECT_ROOT" \
        "${PROFILE_ID:-}" \
        "$BACKEND" \
        "${INSTALL_ROOT:-}"
}

ensure_path_within_root() {
    local root_dir="$1"
    local candidate_path="$2"
    local label="${3:-path}"

    run_path_helper ensure-within "$root_dir" "$candidate_path" "$label"
}

relative_path_from_root() {
    local root_dir="$1"
    local candidate_path="$2"

    run_path_helper relative "$root_dir" "$candidate_path"
}

load_profile_config() {
    local config_file="$1"
    local renderer_python="${CONFIG_PYTHON:-python3}"
    local rendered_config

    if [[ -z "$config_file" ]]; then
        return 0
    fi
    if [[ ! -f "$config_file" ]]; then
        echo "Error: config file '$config_file' not found"
        return 1
    fi
    if [[ ! -f "$CONFIG_RENDERER" ]]; then
        echo "Error: config renderer '$CONFIG_RENDERER' not found"
        return 1
    fi

    if ! rendered_config="$("$renderer_python" "$CONFIG_RENDERER" "$config_file")"; then
        echo "Error: failed to parse config file '$config_file' with '$renderer_python'"
        return 1
    fi

    eval "$rendered_config"
}
