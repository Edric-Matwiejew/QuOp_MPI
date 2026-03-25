#!/usr/bin/env bash
# =============================================================================
# Activation runtime staging and activation script generation.
#
# Extracted from install.sh to keep it focused on orchestration.
# Sourced by install.sh; depends on variables set by the main installer.
# =============================================================================

prepare_activation_runtime() {
    mkdir -p "$ACTIVATION_RUNTIME_DIR"

    cp "$COMMON_LIB" "$ACTIVATION_COMMON_LIB"
    cp "$CONFIG_RENDERER" "$ACTIVATION_CONFIG_RENDERER"
    cp "$PATH_HELPER" "$ACTIVATION_PATH_HELPER"
    cp "$PROFILE_FILE" "$ACTIVATION_PROFILE_FILE"

    if [[ -n "$CONFIG_FILE" ]]; then
        cp "$CONFIG_FILE" "$ACTIVATION_CONFIG_FILE"
    fi
}

write_activation_script() {
    local activation_script="$1"
    local activation_config_line='CONFIG_FILE=""'
    local runtime_rel
    local venv_rel
    local profile_work_rel
    local build_deps_rel
    local fetchcontent_rel
    local skbuild_rel

    runtime_rel="$(relative_path_from_root "$INSTALL_ROOT" "$ACTIVATION_RUNTIME_DIR")"
    venv_rel="$(relative_path_from_root "$INSTALL_ROOT" "$VENV_DIR")"
    profile_work_rel="$(relative_path_from_root "$INSTALL_ROOT" "$PROFILE_WORK_DIR")"
    build_deps_rel="$(relative_path_from_root "$INSTALL_ROOT" "$BUILD_DEPS_DIR")"
    fetchcontent_rel="$(relative_path_from_root "$INSTALL_ROOT" "$FETCHCONTENT_BASE_DIR")"
    skbuild_rel="$(relative_path_from_root "$INSTALL_ROOT" "$SKBUILD_BUILD_DIR")"
    if [[ -n "$ACTIVATION_CONFIG_FILE" ]]; then
        activation_config_line='CONFIG_FILE="$ENVIRONMENTS_DIR/config.toml"'
    fi

    cat >"$activation_script" <<EOF
#!/usr/bin/env bash

if [[ "\${BASH_SOURCE[0]}" == "\$0" ]]; then
    echo "Error: source this script instead of executing it:"
    echo "  source $(printf '%q' "$activation_script")"
    exit 1
fi

INSTALL_ROOT="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="\$INSTALL_ROOT"
ENVIRONMENTS_DIR="\$INSTALL_ROOT/$(printf '%q' "$runtime_rel")"
# Legacy aliases are kept so older helper scripts continue to work.
SETUP_DIR="\$ENVIRONMENTS_DIR"
LIB_DIR="\$ENVIRONMENTS_DIR"
PROFILES_DIR="\$ENVIRONMENTS_DIR"
SITES_DIR="\$PROFILES_DIR"
COMMON_LIB="\$ENVIRONMENTS_DIR/common.sh"
CONFIG_RENDERER="\$ENVIRONMENTS_DIR/render_config.py"
PATH_HELPER="\$ENVIRONMENTS_DIR/path_helper.py"
PROFILE_DIR="\$ENVIRONMENTS_DIR"
SITE_DIR="\$PROFILE_DIR"
PROFILE_FILE="\$ENVIRONMENTS_DIR/hooks.sh"
$activation_config_line
PROFILE=$(printf '%q' "$PROFILE")
PROFILE_ID=$(printf '%q' "$PROFILE_ID")
BACKEND=$(printf '%q' "$BACKEND")
VENV_DIR="\$INSTALL_ROOT/$(printf '%q' "$venv_rel")"
CONFIG_PYTHON="\$VENV_DIR/bin/python"
PROFILE_WORK_DIR="\$INSTALL_ROOT/$(printf '%q' "$profile_work_rel")"
SITE_WORK_DIR="\$PROFILE_WORK_DIR"
BUILD_DEPS_DIR="\$INSTALL_ROOT/$(printf '%q' "$build_deps_rel")"
FETCHCONTENT_BASE_DIR="\$INSTALL_ROOT/$(printf '%q' "$fetchcontent_rel")"
SKBUILD_BUILD_DIR="\$INSTALL_ROOT/$(printf '%q' "$skbuild_rel")"
QUOP_SKBUILD_BUILD_DIR="\$SKBUILD_BUILD_DIR"

export PROJECT_ROOT
export ENVIRONMENTS_DIR
export SETUP_DIR
export LIB_DIR
export PROFILES_DIR
export SITES_DIR
export COMMON_LIB
export CONFIG_RENDERER
export PATH_HELPER
export CONFIG_PYTHON
export PROFILE_DIR
export SITE_DIR
export PROFILE_FILE
export CONFIG_FILE
export PROFILE_WORK_DIR
export SITE_WORK_DIR
export BUILD_DEPS_DIR
export FETCHCONTENT_BASE_DIR
export QUOP_FETCHCONTENT_BASE_DIR="\$FETCHCONTENT_BASE_DIR"
export SKBUILD_BUILD_DIR
export QUOP_SKBUILD_BUILD_DIR
export QUOP_SKBUILD_BUILD_BASE="\$QUOP_SKBUILD_BUILD_DIR"
export QUOP_PROFILE="\$PROFILE"
export QUOP_BACKEND="\$BACKEND"
export QUOP_INSTALL_ROOT="\$INSTALL_ROOT"
export QUOP_VENV_DIR="\$VENV_DIR"
export QUOP_EXAMPLES_DIR="\$INSTALL_ROOT/examples"
export QUOP_BENCHMARKS_DIR="\$INSTALL_ROOT/benchmarks"
export QUOP_DOCS_DIR="\$INSTALL_ROOT/docs"

# shellcheck disable=SC1091
source "\$COMMON_LIB" || return 1

load_profile_config "\$CONFIG_FILE" || return 1

# shellcheck disable=SC1091
source "\$PROFILE_FILE" || return 1

run_profile_environment_hooks || return 1
prepare_profile_venv_args || return 1

# shellcheck disable=SC1091
source "\$VENV_DIR/bin/activate" || return 1

export_profile_cmake_args || return 1
EOF

    chmod +x "$activation_script"
}
