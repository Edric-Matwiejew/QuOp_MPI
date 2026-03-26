#!/usr/bin/env bash
# =============================================================================
# Post-install outputs: manifest, examples/benchmarks copy, docs, packaging.
#
# Extracted from install.sh to keep it focused on orchestration.
# Sourced by install.sh; depends on variables set by the main installer.
# =============================================================================

write_manifest() {
    local manifest_file="$INSTALL_ROOT/manifest.json"
    local git_commit="unknown"
    local git_dirty="false"
    local cmake_args_json="[]"
    local python_version
    local timestamp

    if command -v git >/dev/null 2>&1 && git -C "$PROJECT_ROOT" rev-parse HEAD >/dev/null 2>&1; then
        git_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
        if ! git -C "$PROJECT_ROOT" diff --quiet HEAD 2>/dev/null \
           || [[ -n "$(git -C "$PROJECT_ROOT" ls-files --others --exclude-standard 2>/dev/null)" ]]; then
            git_dirty="true"
        fi
    fi

    python_version="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
    timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    if [[ -n "${CMAKE_ARGS:-}" ]]; then
        cmake_args_json="$(python -c "
import json, shlex, os
args = shlex.split(os.environ.get('CMAKE_ARGS', ''))
print(json.dumps(args))
")"
    fi

    # NOTE: The heredoc below injects shell variables directly into Python
    # string literals.  This is safe because PROFILE is sanitised to
    # [A-Za-z0-9_.-], BACKEND is "mpi"/"wavefront", git_commit is a hex hash,
    # and cmake_args_json is produced by Python's json.dumps (double-quoted).
    # If any of these invariants change, switch to environment-variable passing.
    python - "$manifest_file" <<PY
import json, os, sys

manifest = {
    "project": "QuOp_MPI",
    "timestamp": "$timestamp",
    "profile": "$PROFILE",
    "backend": "$BACKEND",
    "git_commit": "$git_commit",
    "git_dirty": $([[ "$git_dirty" == "true" ]] && echo "True" || echo "False"),
    "python_version": "$python_version",
    "hostname": "$(hostname -s 2>/dev/null || echo unknown)",
    "cmake_args": json.loads('''$cmake_args_json'''),
    "with_docs": $([[ "$WITH_DOCS" == "true" ]] && echo "True" || echo "False"),
}

with open(sys.argv[1], "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\\n")
PY

    info "Manifest: $manifest_file"
}

copy_examples_and_benchmarks() {
    local src_examples="$PROJECT_ROOT/examples"
    local src_benchmarks="$PROJECT_ROOT/benchmarks"
    local dst_examples="$INSTALL_ROOT/examples"
    local dst_benchmarks="$INSTALL_ROOT/benchmarks"

    if [[ -d "$src_examples" ]]; then
        rsync -a --delete "$src_examples/" "$dst_examples/"
        info "Examples:    $dst_examples"
    else
        info "Warning: examples directory not found at $src_examples"
    fi

    if [[ -d "$src_benchmarks" ]]; then
        rsync -a --delete "$src_benchmarks/" "$dst_benchmarks/"
        info "Benchmarks:  $dst_benchmarks"
    else
        info "Warning: benchmarks directory not found at $src_benchmarks"
    fi
}

build_docs() {
    local src_docs="$PROJECT_ROOT/docs"
    local dst_docs="$INSTALL_ROOT/docs"
    local docs_python="$DOCS_VENV_DIR/bin/python"
    local -a docs_packages=()
    local package

    if [[ ! -f "$src_docs/source/conf.py" ]]; then
        echo "Error: docs source not found at $src_docs/source/conf.py" >&2
        return 1
    fi

    while IFS= read -r package; do
        [[ -n "$package" ]] && docs_packages+=("$package")
    done < <("$CONFIG_PYTHON" "$PYPROJECT_DEPS_HELPER" docs-build "$PYPROJECT_TOML")

    if [[ "${#docs_packages[@]}" -eq 0 ]]; then
        echo "Error: no documentation dependencies were resolved from $PYPROJECT_TOML" >&2
        return 1
    fi

    info "Creating isolated documentation environment"
    rm -rf "$DOCS_VENV_DIR"
    "$CONFIG_PYTHON" -m venv "$DOCS_VENV_DIR"

    info "Installing documentation dependencies from pyproject.toml"
    "$docs_python" -m pip install "${docs_packages[@]}"

    rm -rf "$dst_docs"
    "$docs_python" -m sphinx -b html "$src_docs/source" "$dst_docs" || return 1
    info "Docs: $dst_docs"
}

create_package() {
    local archive_name="quop-${PROFILE_ID}-${BACKEND}"
    local archive_path
    local install_root_parent
    local install_root_basename

    install_root_parent="$(dirname "$INSTALL_ROOT")"
    install_root_basename="$(basename "$INSTALL_ROOT")"
    archive_path="${install_root_parent}/${archive_name}.tar.gz"

    step "Creating redistributable archive"
    info "Archiving $INSTALL_ROOT (excluding .cache/ and .deps/)"

    tar -czf "$archive_path" \
        -C "$install_root_parent" \
        --exclude="${install_root_basename}/.cache" \
        --exclude="${install_root_basename}/.deps" \
        "$install_root_basename"

    info "Archive: $archive_path"
}
