#!/usr/bin/env bash
# =============================================================================
# Wheel build, inspection, and repair pipeline.
#
# Extracted from install.sh to keep it focused on orchestration.
# Sourced by install.sh; depends on variables set by the main installer.
# =============================================================================

inspect_built_wheel() {
    local wheel_path="$1"

    WHEEL_PATH="$wheel_path" \
    BACKEND="$BACKEND" \
    INSTALL_VALIDATOR="$INSTALL_VALIDATOR" \
    REQUIRED_WHEEL_MEMBER_SUBSTRING="${REQUIRED_WHEEL_MEMBER_SUBSTRING:-}" \
    python - <<'PY'
import importlib.util
from pathlib import PurePosixPath
import os
import zipfile

wheel_path = os.environ["WHEEL_PATH"]
backend = os.environ["BACKEND"]
validator_path = os.environ["INSTALL_VALIDATOR"]
required_member_substring = os.environ.get("REQUIRED_WHEEL_MEMBER_SUBSTRING", "")

spec = importlib.util.spec_from_file_location("validate_install", validator_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"Could not import install validator from {validator_path}")

validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
required_prefixes = validator.required_extension_stems(backend)

with zipfile.ZipFile(wheel_path) as wheel:
    names = wheel.namelist()

bad_paths = [
    name for name in names
    if name.startswith("quop_mpi/_lib/quop_mpi/")
]
if bad_paths:
    raise SystemExit(
        "Built wheel contains nested package install paths, which indicates "
        "the CMake install tree is being staged incorrectly. "
        f"Examples: {', '.join(bad_paths[:3])}"
    )

missing = []
for prefix in required_prefixes:
    if not any(name.startswith(prefix) and name.endswith(".so") for name in names):
        missing.append(prefix)

if missing:
    raise SystemExit(
        "Wheel is missing required extension modules: "
        + ", ".join(missing)
    )

if required_member_substring and not any(
    required_member_substring in PurePosixPath(name).name for name in names
):
    raise SystemExit(
        "Wheel is missing required vendored members matching substring: "
        + required_member_substring
    )

print(PurePosixPath(wheel_path).name)
PY
}

collect_auditwheel_dependency_files() {
    if [[ -d "${SKBUILD_BUILD_DIR:-}" ]]; then
        find "$SKBUILD_BUILD_DIR" -type f -name '*.so'
    fi

    # SHAFFT may be the only GPU dependency in QuOp's own DT_NEEDED entries.
    # Inspect its installed libraries too, including the ExternalProject cache,
    # and hipFFT wrappers whose CUDA dependencies must remain site-provided.
    typeset -a dependency_prefixes=("${SHAFFT_PATH:-}")
    if [[ -n "${FETCHCONTENT_BASE_DIR:-}" ]]; then
        dependency_prefixes+=("${FETCHCONTENT_BASE_DIR}/shafft-install")
    fi
    local prefix
    local libdir
    {
        for prefix in "${dependency_prefixes[@]}"; do
            [[ -n "$prefix" ]] || continue
            printf '%s\n' "${prefix}/lib" "${prefix}/lib64"
        done
        auditwheel_library_path | tr ':' '\n'
    } | sort -u | while IFS= read -r libdir; do
        [[ -d "$libdir" ]] || continue
        find "$libdir" -maxdepth 1 \( -type f -o -type l \) \
            \( -name 'libshafft*.so*' -o -name 'libhipfft*.so*' \)
    done
}

collect_auditwheel_excludes() {
    typeset -a site_lib_prefixes=(
        libmpi
        libmpich
        libamdhip
        libhipfort
        libhipfft
        libhsa
        libroc
        libnuma
        libcudart
        libcufft
        libcuda
        libcublas
        libcusparse
        libcusolver
        libcurand
        libnvrtc
        libnvJitLink
    )
    local so_path
    local needed
    local prefix
    local auditwheel_ld_path

    if ! command -v readelf >/dev/null 2>&1; then
        return 0
    fi
    auditwheel_ld_path="$(auditwheel_library_path)"

    while IFS= read -r so_path; do
        while IFS= read -r needed; do
            for prefix in "${site_lib_prefixes[@]}"; do
                if [[ "$needed" == "${prefix}"* ]]; then
                    printf '%s\n' "$needed"
                    break
                fi
            done
        done < <(
            readelf -d "$so_path" 2>/dev/null | sed -n 's/.*Shared library: \[\(.*\)\]/\1/p'
            # ldd lists the complete resolved dependency closure, retaining
            # SONAME major versions (for example libcufft.so.12 on CUDA 13).
            if command -v ldd >/dev/null 2>&1; then
                { LD_LIBRARY_PATH="$auditwheel_ld_path" ldd "$so_path" 2>/dev/null || true; } \
                    | awk '/=>/ { print $1 }'
            fi
        )
    done < <(collect_auditwheel_dependency_files | sort -u) | awk '!seen[$0]++'
}

auditwheel_library_path() {
    typeset -a extra_paths=()
    local joined=""

    if [[ -n "${SHAFFT_PATH:-}" ]]; then
        [[ -d "${SHAFFT_PATH}/lib" ]] && extra_paths+=("${SHAFFT_PATH}/lib")
        [[ -d "${SHAFFT_PATH}/lib64" ]] && extra_paths+=("${SHAFFT_PATH}/lib64")
    fi
    local prefix
    local libdir
    for prefix in "${ROCM_PATH:-}" "${HIP_PATH:-}" "${CUDA_PATH:-}" \
        "${HIPFORT_PATH:-}" "${PAWSEY_HIPFFT_HOME:-}"; do
        [[ -n "$prefix" ]] || continue
        for libdir in lib lib64 targets/sbsa-linux/lib; do
            [[ -d "${prefix}/${libdir}" ]] && extra_paths+=("${prefix}/${libdir}")
        done
    done

    if [[ "${#extra_paths[@]}" -gt 0 ]]; then
        joined="$(IFS=:; echo "${extra_paths[*]}")"
    fi

    if [[ -n "$joined" && -n "${LD_LIBRARY_PATH:-}" ]]; then
        printf '%s:%s\n' "$joined" "$LD_LIBRARY_PATH"
    elif [[ -n "$joined" ]]; then
        printf '%s\n' "$joined"
    elif [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        printf '%s\n' "$LD_LIBRARY_PATH"
    fi
}

run_auditwheel() {
    local auditwheel_ld_path="$1"
    shift

    if [[ -n "$auditwheel_ld_path" ]]; then
        LD_LIBRARY_PATH="$auditwheel_ld_path" auditwheel "$@"
    else
        auditwheel "$@"
    fi
}

assert_wheel_supported_by_current_python() {
    local wheel_path="$1"

    WHEEL_PATH="$wheel_path" python - <<'PY'
import os

try:
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
except ModuleNotFoundError:
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename

wheel_path = os.environ["WHEEL_PATH"]
wheel_name = os.path.basename(wheel_path)
_, _, _, wheel_tags = parse_wheel_filename(wheel_name)
supported_tags = list(sys_tags())

if wheel_tags.isdisjoint(supported_tags):
    supported_platforms = []
    for tag in supported_tags:
        if tag.platform not in supported_platforms:
            supported_platforms.append(tag.platform)
        if len(supported_platforms) >= 5:
            break
    raise SystemExit(
        "Wheel is not supported by the active Python interpreter: "
        f"{wheel_name}. "
        "Top supported platforms: "
        + ", ".join(supported_platforms)
    )
PY
}

repair_linux_wheel() {
    local wheel_path="$1"
    local repair_dir
    typeset -a wheel_files=()
    local wheel_file
    typeset -a exclude_args=()
    local exclude_lib
    local auditwheel_ld_path

    if [[ "$BACKEND" != "wavefront" || "$(uname -s)" != "Linux" ]]; then
        printf '%s\n' "$wheel_path"
        return 0
    fi

    if ! command -v auditwheel >/dev/null 2>&1; then
        echo "Error: auditwheel is required for Linux wavefront wheel repair" >&2
        return 1
    fi

    auditwheel_ld_path="$(auditwheel_library_path)"

    while IFS= read -r exclude_lib; do
        [[ -n "$exclude_lib" ]] && exclude_args+=(--exclude "$exclude_lib")
    done < <(collect_auditwheel_excludes)

    step "Inspecting raw wheel with auditwheel" >&2
    run_auditwheel "$auditwheel_ld_path" show "$wheel_path" >&2 || return 1

    repair_dir="$PROFILE_WORK_DIR/repaired"
    rm -rf "$repair_dir"
    mkdir -p "$repair_dir"

    step "Repairing Linux wheel with auditwheel" >&2
    run_auditwheel "$auditwheel_ld_path" repair \
        "${exclude_args[@]}" \
        -w "$repair_dir" \
        "$wheel_path" >&2 || return 1

    while IFS= read -r wheel_file; do
        wheel_files+=("$wheel_file")
    done < <(find "$repair_dir" -maxdepth 1 -type f -name '*.whl' | sort)
    if [[ "${#wheel_files[@]}" -ne 1 ]]; then
        echo "Error: expected exactly one repaired wheel in $repair_dir, found ${#wheel_files[@]}" >&2
        printf '  %s\n' "${wheel_files[@]}" >&2
        return 1
    fi

    # Retag to the native Linux platform so pip accepts this site-specific wheel.
    # auditwheel may produce a manylinux tag higher than what pip's sys_tags()
    # reports (e.g. manylinux_2_39 vs pip supporting up to manylinux_2_38 on
    # SUSE/Cray). Preserve the CPU architecture, including aarch64 on Grace.
    local platform_tag
    platform_tag="$(python -c 'import sysconfig; print(sysconfig.get_platform().replace("-", "_").replace(".", "_"))')"
    step "Retagging repaired wheel to ${platform_tag}" >&2
    python -m wheel tags --remove --platform-tag "$platform_tag" "${wheel_files[0]}" >&2

    # Re-find the wheel after retagging (filename changed)
    wheel_files=()
    while IFS= read -r wheel_file; do
        wheel_files+=("$wheel_file")
    done < <(find "$repair_dir" -maxdepth 1 -type f -name '*.whl' | sort)
    if [[ "${#wheel_files[@]}" -ne 1 ]]; then
        echo "Error: expected exactly one retagged wheel in $repair_dir, found ${#wheel_files[@]}" >&2
        printf '  %s\n' "${wheel_files[@]}" >&2
        return 1
    fi

    step "Inspecting repaired wheel with auditwheel" >&2
    run_auditwheel "$auditwheel_ld_path" show "${wheel_files[0]}" >&2 || return 1
    assert_wheel_supported_by_current_python "${wheel_files[0]}" || return 1

    printf '%s\n' "${wheel_files[0]}"
}

build_and_inspect_wheel() {
    local wheel_dir="$PROFILE_WORK_DIR/dist"
    typeset -a wheel_files=()
    local wheel_file
    local final_wheel

    rm -rf "$wheel_dir"
    mkdir -p "$wheel_dir"

    (
        cd "$PROJECT_ROOT"
        local _skbuild_log="WARNING"
        if [[ "${QUOP_VERBOSE:-false}" == "true" ]]; then _skbuild_log="DEBUG"; fi
        SKBUILD_LOGGING_LEVEL="$_skbuild_log" \
            python -m build --wheel --no-isolation --outdir "$wheel_dir" >&2
    )

    while IFS= read -r wheel_file; do
        wheel_files+=("$wheel_file")
    done < <(find "$wheel_dir" -maxdepth 1 -type f -name '*.whl' | sort)
    if [[ "${#wheel_files[@]}" -ne 1 ]]; then
        echo "Error: expected exactly one wheel in $wheel_dir, found ${#wheel_files[@]}" >&2
        printf '  %s\n' "${wheel_files[@]}" >&2
        return 1
    fi

    inspect_built_wheel "${wheel_files[0]}" >/dev/null || return 1
    final_wheel="$(repair_linux_wheel "${wheel_files[0]}")" || return 1

    if [[ "$BACKEND" == "wavefront" && "$(uname -s)" == "Linux" ]]; then
        REQUIRED_WHEEL_MEMBER_SUBSTRING="libshafft" inspect_built_wheel "$final_wheel" >/dev/null || return 1
    else
        inspect_built_wheel "$final_wheel" >/dev/null || return 1
    fi

    printf '%s\n' "$final_wheel"
}
