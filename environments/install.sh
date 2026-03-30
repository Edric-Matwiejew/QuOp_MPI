#!/usr/bin/env bash
# =============================================================================
# QuOp_MPI -- profile-based environment installer
#
# Creates a virtual environment, builds mpi4py / h5py according to the chosen
# profile, and installs the QuOp_MPI Python package with profile-specific
# CMake arguments.
#
# Usage:  ./environments/install.sh -p <profile> [-b mpi|wavefront] [--prefix <dir>] [--clean]
# =============================================================================
set -euo pipefail

quop_shell_source_path() {
    if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
        printf '%s\n' "${BASH_SOURCE[0]}"
        return 0
    fi
    if [[ -n "${ZSH_VERSION:-}" ]]; then
        eval 'printf "%s\n" "${(%):-%x}"'
        return 0
    fi
    printf '%s\n' "$0"
}

# ---- Resolve project root (one level above this script) --------------------
CALLER_CWD="$(pwd)"
PROJECT_ROOT="$(cd -- "$(dirname -- "$(quop_shell_source_path)")/.." && pwd)"
ENVIRONMENTS_DIR="$PROJECT_ROOT/environments"
LIB_DIR="$ENVIRONMENTS_DIR/lib"
PROFILES_DIR="$ENVIRONMENTS_DIR/profiles"
COMMON_LIB="$LIB_DIR/common.sh"
CONFIG_RENDERER="$LIB_DIR/render_config.py"
PATH_HELPER="$LIB_DIR/path_helper.py"
INSTALL_VALIDATOR="$LIB_DIR/validate_install.py"
PYPROJECT_DEPS_HELPER="$LIB_DIR/pyproject_deps.py"
PYPROJECT_TOML="$PROJECT_ROOT/pyproject.toml"

if [[ ! -f "$COMMON_LIB" ]]; then
    echo "Error: shared environments library '$COMMON_LIB' not found"
    exit 1
fi
if [[ ! -f "$INSTALL_VALIDATOR" ]]; then
    echo "Error: install validator '$INSTALL_VALIDATOR' not found"
    exit 1
fi
if [[ ! -f "$PATH_HELPER" ]]; then
    echo "Error: path helper '$PATH_HELPER' not found"
    exit 1
fi
if [[ ! -f "$PYPROJECT_DEPS_HELPER" ]]; then
    echo "Error: pyproject dependency helper '$PYPROJECT_DEPS_HELPER' not found"
    exit 1
fi
if [[ ! -f "$PYPROJECT_TOML" ]]; then
    echo "Error: project metadata '$PYPROJECT_TOML' not found"
    exit 1
fi

# shellcheck disable=SC1091
source "$COMMON_LIB"

ACTIVATION_LIB="$LIB_DIR/activation.sh"
POST_INSTALL_LIB="$LIB_DIR/post_install.sh"
WHEEL_LIB="$LIB_DIR/wheel.sh"

for _lib_file in "$ACTIVATION_LIB" "$POST_INSTALL_LIB" "$WHEEL_LIB"; do
    if [[ ! -f "$_lib_file" ]]; then
        echo "Error: environments library '$_lib_file' not found"
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$_lib_file"
done
unset _lib_file

# ---- Defaults --------------------------------------------------------------
BACKEND="mpi"
PROFILE=""
CONFIG_FILE=""
INSTALL_PREFIX=""
CLEAN_MODE="none"
WITH_DOCS="false"
PACKAGE="false"
QUOP_VERBOSE="false"

# ---- Usage ------------------------------------------------------------------
usage() {
    local profiles
    profiles="$(list_available_profiles)"
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Set up a Python virtual environment for QuOp_MPI with all dependencies
correctly built for the selected profile and backend.

Options:
  -p, --profile PROFILE   Profile to use (required)
                          Available: ${profiles:-<none found>}
  -c, --config FILE       TOML config file for profile overrides
                          Default: environments/profiles/<profile>/<backend>/config.toml when present
  -b, --backend BACKEND   Build backend: mpi (default) or wavefront
  --prefix DIR            Install prefix for the venv, caches, deps, and
                          activation script. Relative paths use the caller's cwd.
  --with-docs             Build HTML documentation into <prefix>/docs/
  --package               Create a redistributable tar.gz archive of the
                          install prefix (implies --with-docs)
  --clean                 Remove <prefix>/.cache before installing
  --veryclean             Remove <prefix>/.cache and <prefix>/.deps before installing
  -v, --verbose           Show full build output (pip, cmake, f2py)
  -h, --help              Show this help message

Examples:
  $(basename "$0") -p generic -b mpi --prefix ./quop-install
  $(basename "$0") -p pawsey-setonix -b wavefront --prefix /scratch/\$USER/quop
  $(basename "$0") -p pawsey-setonix -b wavefront --veryclean
EOF
    exit "${1:-0}"
}

cleanup_install_state() {
    case "$CLEAN_MODE" in
        none)
            return 0
            ;;
        clean)
            step "Cleaning install cache"
            rm -rf "$CACHE_ROOT"
            ;;
        veryclean)
            step "Cleaning install cache and dependency cache"
            rm -rf "$CACHE_ROOT" "$DEPS_ROOT"
            ;;
        *)
            echo "Error: unsupported clean mode '$CLEAN_MODE'"
            return 1
            ;;
    esac
}

ensure_python_build_requirements() {
    step "Ensuring Python build requirements"
    local -a _pip_quiet=()
    if [[ "${QUOP_VERBOSE:-false}" != "true" ]]; then _pip_quiet=(--quiet); fi
    python -m pip install "${_pip_quiet[@]}" --upgrade \
        pip \
        setuptools \
        wheel \
        build \
        "scikit-build-core>=0.10" \
        ninja \
        "cmake>=3.25,<4" \
        "numpy>=2.0"

    if [[ "$(uname -s)" == "Linux" ]]; then
        python -m pip install "${_pip_quiet[@]}" --upgrade auditwheel patchelf
    fi
}

ensure_python_support_packages() {
    info "Ensuring Python support packages"
    local -a _pip_quiet=()
    if [[ "${QUOP_VERBOSE:-false}" != "true" ]]; then _pip_quiet=(--quiet); fi
    python -m pip install "${_pip_quiet[@]}" pytest-mpi scipy pandas networkx
}

validate_installed_package() {
    local validation_dir="$PROFILE_WORK_DIR/validate"
    mkdir -p "$validation_dir"

    (
        cd "$validation_dir"
        PROJECT_ROOT="$PROJECT_ROOT" BACKEND="$BACKEND" python "$INSTALL_VALIDATOR"
    )
}

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--profile) PROFILE="$2"; shift 2 ;;
        -c|--config)  CONFIG_FILE="$2"; shift 2 ;;
        -b|--backend) BACKEND="$2"; shift 2 ;;
        --prefix)     INSTALL_PREFIX="$2"; shift 2 ;;
        --with-docs)  WITH_DOCS="true"; shift ;;
        --package)    PACKAGE="true"; WITH_DOCS="true"; shift ;;
        --clean)      CLEAN_MODE="clean"; shift ;;
        --veryclean)  CLEAN_MODE="veryclean"; shift ;;
        -v|--verbose) QUOP_VERBOSE="true"; shift ;;
        -h|--help)    usage 0               ;;
        *) echo "Error: unknown option '$1'"; usage 1 ;;
    esac
done

if [[ -z "$PROFILE" ]]; then
    echo "Error: --profile is required"
    usage 1
fi

if [[ "$BACKEND" != "mpi" && "$BACKEND" != "wavefront" ]]; then
    echo "Error: --backend must be 'mpi' or 'wavefront', got '$BACKEND'"
    exit 1
fi

# ---- Require Python 3.11+ (needed for tomllib in render_config.py) ----------
_py3_cmd="$(command -v python3 2>/dev/null || true)"
if [[ -z "$_py3_cmd" ]]; then
    echo "Error: python3 not found in PATH"
    exit 1
fi
_py3_version="$("$_py3_cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
_py3_major="${_py3_version%%.*}"
_py3_minor="${_py3_version#*.}"
if [[ "$_py3_major" -lt 3 || ( "$_py3_major" -eq 3 && "$_py3_minor" -lt 11 ) ]]; then
    echo "Error: Python 3.11+ is required (found $_py3_version at $_py3_cmd)"
    exit 1
fi
unset _py3_cmd _py3_version _py3_major _py3_minor

PROFILE_DIR="$PROFILES_DIR/${PROFILE}"
PROFILE_FILE="$PROFILE_DIR/hooks.sh"
if [[ ! -f "$PROFILE_FILE" ]]; then
    echo "Error: profile '$PROFILE' not found at $PROFILE_FILE"
    echo "Available profiles:"
    print_available_profiles
    exit 1
fi

if [[ -z "$CONFIG_FILE" ]]; then
    DEFAULT_CONFIG_FILE="$PROFILE_DIR/$BACKEND/config.toml"
    if [[ -f "$DEFAULT_CONFIG_FILE" ]]; then
        CONFIG_FILE="$DEFAULT_CONFIG_FILE"
    fi
fi

PROFILE_ID="${PROFILE//[^A-Za-z0-9_.-]/_}"

# Clear any CONFIG_PYTHON leaked from a prior activation script -- the venv
# does not exist yet so load_profile_config must fall back to system python3.
unset CONFIG_PYTHON
load_profile_config "$CONFIG_FILE"

# ---- Source the profile -----------------------------------------------------
#
# A profile must define the following functions:
#
#   profile_install_mpi4py   Install / configure mpi4py (called inside venv).
#   profile_install_h5py     Install / configure h5py   (called inside venv).
#   profile_cmake_args       Print CMake flags (one per line) for the backend.
#
# A profile may optionally define:
#
#   profile_load_modules          Load required environment modules.
#   profile_load_modules_gpu      Additional modules for the wavefront backend.
#   profile_post_modules_env      Set environment variables after module loads.
#   profile_configure_venv        Append extra args to VENV_ARGS array.
#
# A profile should set these variables:
#
#   PROFILE_DESCRIPTION     Human-readable profile label.
#   WAVEFRONT_SUPPORTED     "true" if the profile supports the wavefront backend.
#   PYTHON_VERSION          Major.minor string, e.g. "3.11" (auto-detected if omitted).
#
# The variable BACKEND ("mpi" or "wavefront") is available inside every hook.
# The variable PROJECT_ROOT points to the repository root.
#
source "$PROFILE_FILE"

# ---- Validate wavefront support --------------------------------------------
if [[ "$BACKEND" == "wavefront" && "${WAVEFRONT_SUPPORTED:-false}" != "true" ]]; then
    echo "Error: profile '$PROFILE' does not support the wavefront backend"
    exit 1
fi

info "Profile:  ${PROFILE_DESCRIPTION:-$PROFILE}"
info "Backend:  $BACKEND"
info "Project:  $PROJECT_ROOT"
if [[ -n "$CONFIG_FILE" ]]; then
    info "Config:   $CONFIG_FILE"
fi

# ---- Determine paths --------------------------------------------------------
if [[ -n "$INSTALL_PREFIX" ]]; then
    INSTALL_ROOT="$(expand_install_path "$INSTALL_PREFIX" "$CALLER_CWD")"
else
    INSTALL_ROOT="$(expand_install_path "${CFG_INSTALL_ROOT:-.quop-install}" "$PROJECT_ROOT")"
fi
VENV_DIR="$(expand_install_path "${CFG_VENV_DIR_TEMPLATE:-.venv_${PROFILE_ID}_${BACKEND}}" "$INSTALL_ROOT")"
PROFILE_WORK_DIR="$(expand_install_path "${CFG_WORK_DIR_TEMPLATE:-.cache/quop/$PROFILE_ID/$BACKEND}" "$INSTALL_ROOT")"
CACHE_ROOT="$INSTALL_ROOT/.cache"
DEPS_ROOT="$INSTALL_ROOT/.deps"
FETCHCONTENT_BASE_DIR="$INSTALL_ROOT/.deps/$PROFILE_ID/$BACKEND"
SKBUILD_BUILD_DIR="$PROFILE_WORK_DIR/skbuild"
BUILD_DEPS_DIR="$PROFILE_WORK_DIR/deps"
DOCS_VENV_DIR="$PROFILE_WORK_DIR/docs-venv"
ACTIVATION_SCRIPT="$INSTALL_ROOT/activate-${PROFILE_ID}-${BACKEND}.sh"
ACTIVATION_RUNTIME_DIR="$INSTALL_ROOT/.environments-runtime/$PROFILE_ID/$BACKEND"
ACTIVATION_COMMON_LIB="$ACTIVATION_RUNTIME_DIR/common.sh"
ACTIVATION_CONFIG_RENDERER="$ACTIVATION_RUNTIME_DIR/render_config.py"
ACTIVATION_PATH_HELPER="$ACTIVATION_RUNTIME_DIR/path_helper.py"
ACTIVATION_PROFILE_FILE="$ACTIVATION_RUNTIME_DIR/hooks.sh"
ACTIVATION_CONFIG_FILE=""
if [[ -n "$CONFIG_FILE" ]]; then
    ACTIVATION_CONFIG_FILE="$ACTIVATION_RUNTIME_DIR/config.toml"
fi

PROJECT_ROOT_REAL="$(canonicalize_path "$PROJECT_ROOT")"
INSTALL_ROOT_REAL="$(canonicalize_path "$INSTALL_ROOT")"
if [[ "$INSTALL_ROOT_REAL" == "$PROJECT_ROOT_REAL" ]]; then
    echo "Error: install prefix must not be the project root: $PROJECT_ROOT"
    echo "Hint: pass --prefix <dir> or set install.root to a dedicated subdirectory"
    exit 1
fi

ensure_path_within_root "$INSTALL_ROOT" "$VENV_DIR" "virtual environment"
ensure_path_within_root "$INSTALL_ROOT" "$PROFILE_WORK_DIR" "profile work directory"
ensure_path_within_root "$INSTALL_ROOT" "$BUILD_DEPS_DIR" "build dependency directory"
ensure_path_within_root "$INSTALL_ROOT" "$DOCS_VENV_DIR" "documentation virtual environment"
ensure_path_within_root "$INSTALL_ROOT" "$FETCHCONTENT_BASE_DIR" "FetchContent cache"
ensure_path_within_root "$INSTALL_ROOT" "$SKBUILD_BUILD_DIR" "scikit-build directory"
ensure_path_within_root "$INSTALL_ROOT" "$ACTIVATION_SCRIPT" "activation script"
ensure_path_within_root "$INSTALL_ROOT" "$ACTIVATION_RUNTIME_DIR" "activation runtime directory"

export PROFILE_WORK_DIR
export BUILD_DEPS_DIR
export FETCHCONTENT_BASE_DIR
export QUOP_FETCHCONTENT_BASE_DIR="$FETCHCONTENT_BASE_DIR"
export SKBUILD_BUILD_DIR
export QUOP_SKBUILD_BUILD_DIR="$SKBUILD_BUILD_DIR"
export QUOP_SKBUILD_BUILD_BASE="$SKBUILD_BUILD_DIR"
export QUOP_VERBOSE
if [[ "${QUOP_VERBOSE}" == "true" ]]; then export VERBOSE=1; fi

# ---- Load modules -----------------------------------------------------------
step "Loading modules"
run_profile_environment_hooks

CONFIG_PYTHON="$(resolve_python_interpreter "${PYTHON_VERSION:-}")"
info "Python:   $CONFIG_PYTHON"

cleanup_install_state

mkdir -p \
    "$INSTALL_ROOT" \
    "$(dirname "$VENV_DIR")" \
    "$PROFILE_WORK_DIR" \
    "$BUILD_DEPS_DIR" \
    "$FETCHCONTENT_BASE_DIR" \
    "$ACTIVATION_RUNTIME_DIR"

# Copy config into the environment so the install is self-contained.
if [[ -n "$CONFIG_FILE" ]]; then
    cp "$CONFIG_FILE" "$ACTIVATION_CONFIG_FILE"
    CONFIG_FILE="$ACTIVATION_CONFIG_FILE"
fi

info "Root:   $INSTALL_ROOT"
info "Cache:  $CACHE_ROOT"
info "Venv:   $VENV_DIR"
info "Work:   $PROFILE_WORK_DIR"
info "Deps:   $FETCHCONTENT_BASE_DIR"

# ---- Create venv & install Python dependencies -----------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    step "Creating virtual environment"

    prepare_profile_venv_args

    "$CONFIG_PYTHON" -m venv ${VENV_ARGS[@]+"${VENV_ARGS[@]}"} "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    ensure_python_build_requirements
    ensure_python_support_packages

    step "Installing mpi4py"
    profile_install_mpi4py

    step "Installing h5py"
    profile_install_h5py
else
    require_python_interpreter_version "$VENV_DIR/bin/python" "${PYTHON_VERSION:-}" "virtual environment" || {
        echo "Hint: remove '$VENV_DIR' manually or adjust install.venv_dir to avoid reusing an incompatible environment"
        exit 1
    }
    info "Venv already exists -- activating"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    ensure_python_build_requirements
fi

# ---- Build and install QuOp_MPI ---------------------------------------------
step "Building QuOp_MPI wheel"

export_profile_cmake_args
export QUOP_BACKEND="$BACKEND"
info "CMake args: $CMAKE_ARGS"

# Reconfigure from a clean scikit-build-core directory so profile/backend changes do not
# inherit stale CMake cache entries from an earlier install.
rm -rf "$SKBUILD_BUILD_DIR"

if ! WHEEL_PATH="$(build_and_inspect_wheel)"; then
    exit 1
fi
info "Wheel:   $WHEEL_PATH"

step "Installing QuOp_MPI Python package"
python -m pip install --force-reinstall --no-deps "$WHEEL_PATH"

step "Staging activation runtime"
prepare_activation_runtime

step "Writing activation script"
write_activation_script "$ACTIVATION_SCRIPT"

step "Validating installed package"
if ! INSTALLED_PACKAGE_PATH="$(validate_installed_package)"; then
    exit 1
fi
info "Installed package: $INSTALLED_PACKAGE_PATH"

step "Copying examples and benchmarks"
copy_examples_and_benchmarks

if [[ "$WITH_DOCS" == "true" ]]; then
    step "Building documentation"
    build_docs
fi

step "Writing build manifest"
write_manifest

if [[ "$PACKAGE" == "true" ]]; then
    create_package
fi

# ---- Done -------------------------------------------------------------------
step "Environment ready"
info "Activate the environment with:"
info "  source $ACTIVATION_SCRIPT"
info ""
info "Environment variables available after activation:"
info "  QUOP_EXAMPLES_DIR    -- examples directory"
info "  QUOP_BENCHMARKS_DIR  -- benchmarks directory"
info "  QUOP_DOCS_DIR        -- documentation directory"
