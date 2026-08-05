import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_SH = PROJECT_ROOT / "environments" / "lib" / "common.sh"
ACTIVATION_LIB = PROJECT_ROOT / "environments" / "lib" / "activation.sh"
POST_INSTALL_LIB = PROJECT_ROOT / "environments" / "lib" / "post_install.sh"
WHEEL_LIB = PROJECT_ROOT / "environments" / "lib" / "wheel.sh"
INSTALL_SH = PROJECT_ROOT / "environments" / "install.sh"
VALIDATE_INSTALL = PROJECT_ROOT / "environments" / "lib" / "validate_install.py"
PYPROJECT_DEPS_HELPER = PROJECT_ROOT / "environments" / "lib" / "pyproject_deps.py"
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
CMAKE_LISTS = PROJECT_ROOT / "CMakeLists.txt"
NATIVE_CMAKE_LISTS = PROJECT_ROOT / "native" / "CMakeLists.txt"
FORTRAN_PREPROCESS_CMAKE = PROJECT_ROOT / "cmake" / "QuOpFortranPreprocess.cmake"
HIPFORT_DEPENDENCY_CMAKE = PROJECT_ROOT / "cmake" / "HipfortDependency.cmake"
SHAFFT_DEPENDENCY_CMAKE = PROJECT_ROOT / "cmake" / "SHAFFTDependency.cmake"
ADD_F2PY_CMAKE = PROJECT_ROOT / "cmake" / "QuOpF2pyLibrary.cmake"
WAVEFRONT_CONTEXT_CMAKE = PROJECT_ROOT / "native" / "wavefront" / "context" / "CMakeLists.txt"
GENERIC_HOOKS = PROJECT_ROOT / "environments" / "profiles" / "generic" / "hooks.sh"
MACOS_HOOKS = PROJECT_ROOT / "environments" / "profiles" / "macos" / "hooks.sh"
UBUNTU_24_HOOKS = PROJECT_ROOT / "environments" / "profiles" / "ubuntu-24" / "hooks.sh"
PAWSEY_HOOKS = PROJECT_ROOT / "environments" / "profiles" / "pawsey-setonix" / "hooks.sh"
PAWSEY_CONFIG = PROJECT_ROOT / "environments" / "profiles" / "pawsey-setonix" / "mpi" / "config.toml"
MACOS_CONFIG = PROJECT_ROOT / "environments" / "profiles" / "macos" / "mpi" / "config.toml"
UBUNTU_24_CONFIG = PROJECT_ROOT / "environments" / "profiles" / "ubuntu-24" / "mpi" / "config.toml"
GENERIC_CONFIG = PROJECT_ROOT / "environments" / "profiles" / "generic" / "mpi" / "config.toml"
ZSH = shutil.which("zsh")


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_shell(
    script: str,
    *,
    env: dict[str, str] | None = None,
    shell: str = "bash",
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    return subprocess.run(
        [shell, "-lc", script],
        cwd=PROJECT_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_expand_install_path_uses_profile_backend_and_root_placeholders():
    script = f"""
source {shlex.quote(str(COMMON_SH))}
PROJECT_ROOT=/tmp/quop-project
PROFILE_ID=pawsey-setonix
BACKEND=wavefront
INSTALL_ROOT=/tmp/quop-install
CONFIG_PYTHON={shlex.quote(sys.executable)}
expand_install_path '.venv_{{profile}}_{{backend}}' "$INSTALL_ROOT"
expand_install_path '{{root}}/work/{{profile}}' "$PROJECT_ROOT"
"""

    result = run_shell(script)

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.splitlines() == [
        "/tmp/quop-install/.venv_pawsey-setonix_wavefront",
        "/tmp/quop-install/work/pawsey-setonix",
    ]


def test_relative_path_from_root_returns_install_relative_path():
    script = f"""
source {shlex.quote(str(COMMON_SH))}
CONFIG_PYTHON={shlex.quote(sys.executable)}
relative_path_from_root /tmp/quop-install /tmp/quop-install/.cache/quop/generic/mpi
"""

    result = run_shell(script)

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == ".cache/quop/generic/mpi"


def test_resolve_python_interpreter_accepts_matching_config_python():
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    script = f"""
source {shlex.quote(str(COMMON_SH))}
CONFIG_PYTHON={shlex.quote(sys.executable)}
resolve_python_interpreter {shlex.quote(version)}
"""

    result = run_shell(script)

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == sys.executable


def test_resolve_python_interpreter_rejects_mismatched_config_python():
    script = f"""
source {shlex.quote(str(COMMON_SH))}
CONFIG_PYTHON={shlex.quote(sys.executable)}
resolve_python_interpreter 9.9
"""

    result = run_shell(script)

    assert result.returncode != 0
    assert "expected 9.9" in (result.stdout + result.stderr)


def test_find_rocm_path_prefers_hipconfig_prefix_over_wrapper_location():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmpbin = tmpdir_path / "bin"
        tmprocm = tmpdir_path / "rocm-prefix"
        tmpbin.mkdir()
        tmprocm.mkdir()

        hipconfig = tmpbin / "hipconfig"
        hipconfig.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' {shlex.quote(str(tmprocm))}\n"
        )
        os.chmod(hipconfig, 0o755)

        hipcc = tmpbin / "hipcc"
        hipcc.write_text("#!/usr/bin/env bash\nexit 0\n")
        os.chmod(hipcc, 0o755)

        script = f"""
unset ROCM_PATH
unset -f hipconfig hipcc 2>/dev/null || true
PATH={shlex.quote(str(tmpbin))}:$PATH
hash -r
source {shlex.quote(str(COMMON_SH))}
find_rocm_path
"""

        result = run_shell(script)

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == str(tmprocm)


def test_profile_listing_helpers_are_portable_and_sorted():
    script = f"""
PROFILES_DIR={shlex.quote(str(PROJECT_ROOT / "environments" / "profiles"))}
source {shlex.quote(str(COMMON_SH))}
list_available_profiles
print_available_profiles
"""

    result = run_shell(script)

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.splitlines() == [
        "generic, macos, pawsey-setonix, ubuntu-24",
        "  generic",
        "  macos",
        "  pawsey-setonix",
        "  ubuntu-24",
    ]


def test_load_profile_config_exposes_macos_homebrew_overrides():
    script = f"""
CONFIG_RENDERER={shlex.quote(str(PROJECT_ROOT / "environments" / "lib" / "render_config.py"))}
source {shlex.quote(str(COMMON_SH))}
load_profile_config {shlex.quote(str(MACOS_CONFIG))}
printf 'PREFIX=%s\\n' "${{CFG_HOMEBREW_PREFIX-unset}}"
printf 'HDF5=%s\\n' "${{CFG_HOMEBREW_HDF5_FORMULA-unset}}"
printf 'FFTW=%s\\n' "${{CFG_HOMEBREW_FFTW_FORMULA-unset}}"
"""

    result = run_shell(script)

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.splitlines() == [
        "PREFIX=unset",
        "HDF5=hdf5-mpi",
        "FFTW=fftw",
    ]


def test_default_profile_configs_use_dedicated_install_subdir():
    for config_path in (GENERIC_CONFIG, MACOS_CONFIG, UBUNTU_24_CONFIG, PAWSEY_CONFIG):
        assert 'root = ".quop-install"' in config_path.read_text()


def test_profile_hooks_use_common_profile_helpers():
    generic_hooks_text = GENERIC_HOOKS.read_text()
    macos_hooks_text = MACOS_HOOKS.read_text()
    ubuntu_24_hooks_text = UBUNTU_24_HOOKS.read_text()
    pawsey_hooks_text = PAWSEY_HOOKS.read_text()

    assert "apply_profile_defaults" in generic_hooks_text
    assert "apply_profile_defaults" in macos_hooks_text
    assert "apply_profile_defaults" in ubuntu_24_hooks_text
    assert "apply_profile_defaults" in pawsey_hooks_text
    assert "CFG_PROFILE_MODULES_COMMON" in pawsey_hooks_text
    assert "CFG_PROFILE_MODULES_WAVEFRONT" in pawsey_hooks_text
    assert "load_module_list CFG_PROFILE_MODULES_MPI_PYTHON" in pawsey_hooks_text


def test_pyproject_cmake_requirement_matches_cmakelists_minimum():
    with PYPROJECT_TOML.open("rb") as handle:
        pyproject = tomllib.load(handle)

    build_system = pyproject["build-system"]
    tool_scikit_build = pyproject.get("tool", {}).get("scikit-build", {})
    cmake_requirement = tool_scikit_build.get("cmake", {}).get("version")

    if cmake_requirement is None:
        build_requires = build_system["requires"]
        cmake_requirement = next(req for req in build_requires if req.startswith("cmake"))

    cmake_lists_text = CMAKE_LISTS.read_text()
    minimum_match = re.search(r"cmake_minimum_required\(VERSION\s+(\d+\.\d+)", cmake_lists_text)
    assert minimum_match is not None

    requirement_match = re.search(r">=([0-9]+(?:\.[0-9]+)*)", cmake_requirement)
    assert requirement_match is not None

    minimum_version = tuple(int(part) for part in minimum_match.group(1).split("."))
    required_version = tuple(int(part) for part in requirement_match.group(1).split("."))

    assert required_version >= minimum_version


def test_pyproject_uses_scikit_build_core_backend():
    with PYPROJECT_TOML.open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert pyproject["build-system"]["build-backend"] == "scikit_build_core.build"
    assert pyproject["tool"]["scikit-build"]["wheel"]["packages"] == ["src/quop_mpi"]


def test_pyproject_docs_extra_includes_sphinx_and_required_extensions():
    with PYPROJECT_TOML.open("rb") as handle:
        pyproject = tomllib.load(handle)

    docs_extra = pyproject["project"]["optional-dependencies"]["docs"]

    assert "sphinx==6.2.1" in docs_extra
    assert "numpydoc==1.5.0" in docs_extra
    assert "sphinxcontrib-bibtex==2.5.0" in docs_extra
    assert "sphinxcontrib-mermaid" in docs_extra
    assert "sphinx-rtd-theme==1.2.0" in docs_extra


def test_pyproject_deps_helper_collects_docs_build_requirements():
    result = subprocess.run(
        [sys.executable, str(PYPROJECT_DEPS_HELPER), "docs-build", str(PYPROJECT_TOML)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    requirements = result.stdout.splitlines()

    assert any(req.startswith("numpy") for req in requirements)
    assert any(req.startswith("scipy") for req in requirements)
    assert any(req.startswith("pandas") for req in requirements)
    assert any(req.startswith("networkx") for req in requirements)
    assert any(req.startswith("sphinx") for req in requirements)
    assert any(req.startswith("numpydoc") for req in requirements)
    assert not any(req.startswith("mpi4py") for req in requirements)
    assert not any(req.startswith("h5py") for req in requirements)


def test_pyproject_version_matches_cmake_project_version():
    with PYPROJECT_TOML.open("rb") as handle:
        pyproject = tomllib.load(handle)

    cmake_lists_text = CMAKE_LISTS.read_text()
    project_version_match = re.search(
        r"project\(\s*QuOp_MPI\s+VERSION\s+([0-9]+(?:\.[0-9]+)*)",
        cmake_lists_text,
        re.MULTILINE,
    )

    assert project_version_match is not None
    assert pyproject["project"]["version"] == project_version_match.group(1)


def test_pyproject_build_targets_are_backend_aware_without_install_sh():
    with PYPROJECT_TOML.open("rb") as handle:
        pyproject = tomllib.load(handle)

    targets = pyproject["tool"]["scikit-build"]["build"]["targets"]

    assert "comm_info_wrapper_f2py" in targets
    assert "mpi_context_wrapper" in targets
    assert "mpi_transverse_field_propagator_f2py" in targets
    assert "wavefront_context_wrapper" in targets
    assert "quop_f2py_targets" not in targets


def test_native_cmakelists_keeps_wavefront_placeholders_for_direct_builds():
    native_cmake_text = NATIVE_CMAKE_LISTS.read_text()

    assert "if(${WAVEFRONT_BACKEND})" in native_cmake_text
    assert "add_custom_target(wavefront_context_wrapper)" in native_cmake_text
    assert "add_custom_target(wavefront_sparse_propagator_f2py)" in native_cmake_text

def test_top_level_cmake_uses_native_subdir_for_compiled_sources():
    cmake_lists_text = CMAKE_LISTS.read_text()

    assert "add_subdirectory(native)" in cmake_lists_text
    assert "add_subdirectory(src)" not in cmake_lists_text


def test_top_level_cmake_requires_mpi_fortran_but_not_mpi_cxx():
    cmake_lists_text = CMAKE_LISTS.read_text()

    assert "find_package(MPI REQUIRED COMPONENTS Fortran)" in cmake_lists_text
    assert "find_package(MPI REQUIRED COMPONENTS CXX)" not in cmake_lists_text

def test_fortran_barrier_helper_is_shared_by_f2py_and_wavefront_context():
    preprocess_text = FORTRAN_PREPROCESS_CMAKE.read_text()
    hipfort_dependency_text = HIPFORT_DEPENDENCY_CMAKE.read_text()
    cmake_lists_text = CMAKE_LISTS.read_text()
    add_f2py_text = ADD_F2PY_CMAKE.read_text()
    wavefront_context_text = WAVEFRONT_CONTEXT_CMAKE.read_text()

    assert "function(quop_add_fortran_source_barrier)" in preprocess_text
    assert re.search(
        r"if\(NOT HIPFORT_COMPILER\)\s+set\(HIPFORT_COMPILER\s+"
        r'"\$\{CMAKE_Fortran_COMPILER\}"\s+CACHE STRING '
        r'"Fortran compiler for hipfort build \(e\.g\., ftn for Cray, mpifort for others\)" FORCE\s+\)\s+endif\(\)',
        cmake_lists_text,
        re.MULTILINE,
    )
    assert "set(HIPFORT_COMPILER ${CMAKE_Fortran_COMPILER})" not in cmake_lists_text
    assert "-DHIPFORT_COMPILER=${HIPFORT_COMPILER}" in hipfort_dependency_text
    assert "-DHIPFORT_COMPILER=${CMAKE_Fortran_COMPILER}" not in hipfort_dependency_text
    assert '${CMAKE_SOURCE_DIR}/native/.f2py_f2cmap' in add_f2py_text
    assert "quop_add_fortran_source_barrier(" in add_f2py_text
    assert "quop_add_fortran_source_barrier(" in wavefront_context_text
    assert "QUOP_F2PY_TARGETS" not in add_f2py_text
    assert "Removed forced rebuild stamp and target" not in add_f2py_text
    assert 'CACHE STRING "Path to the Python 3 executable"' not in add_f2py_text


def test_build_uses_standard_cmake_build_type_everywhere():
    cmake_lists_text = CMAKE_LISTS.read_text()
    generic_hooks_text = GENERIC_HOOKS.read_text()
    macos_hooks_text = MACOS_HOOKS.read_text()
    ubuntu_24_hooks_text = UBUNTU_24_HOOKS.read_text()
    pawsey_hooks_text = PAWSEY_HOOKS.read_text()
    # Per-backend config files: MPI config is the default, wavefront checked separately.
    ubuntu_24_mpi_config_text = UBUNTU_24_CONFIG.read_text()
    pawsey_mpi_config_text = PAWSEY_CONFIG.read_text()
    ubuntu_24_wf_config = PROJECT_ROOT / "environments" / "profiles" / "ubuntu-24" / "wavefront" / "config.toml"
    pawsey_wf_config = PROJECT_ROOT / "environments" / "profiles" / "pawsey-setonix" / "wavefront" / "config.toml"
    ubuntu_24_wf_config_text = ubuntu_24_wf_config.read_text()
    pawsey_wf_config_text = pawsey_wf_config.read_text()

    assert 'set(BUILD "RELEASE"' not in cmake_lists_text
    assert "cmake_print_variables(BUILD)" not in cmake_lists_text
    assert re.search(
        r"if\(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE\)\s+"
        r'set\(CMAKE_BUILD_TYPE\s+Release\s+CACHE STRING "Build type" FORCE\s+\)\s+endif\(\)',
        cmake_lists_text,
        re.MULTILINE,
    )
    assert "if(CMAKE_BUILD_TYPE STREQUAL \"Debug\")" in cmake_lists_text
    assert "elseif(CMAKE_BUILD_TYPE STREQUAL \"Release\")" in cmake_lists_text

    assert "-DBUILD=" not in generic_hooks_text
    assert "-DBUILD=" not in macos_hooks_text
    assert "-DBUILD=" not in ubuntu_24_hooks_text
    assert "-DBUILD=" not in pawsey_hooks_text
    assert "-DCMAKE_BUILD_TYPE=Release" in generic_hooks_text
    assert "-DCMAKE_BUILD_TYPE=Release" in macos_hooks_text
    assert "-DCMAKE_BUILD_TYPE=${MPI_BUILD_TYPE}" in ubuntu_24_hooks_text
    assert "-DCMAKE_BUILD_TYPE=${WAVEFRONT_BUILD_TYPE}" in ubuntu_24_hooks_text
    assert "-DCMAKE_BUILD_TYPE=${MPI_BUILD_TYPE}" in pawsey_hooks_text
    assert "-DCMAKE_BUILD_TYPE=${WAVEFRONT_BUILD_TYPE}" in pawsey_hooks_text
    assert 'mpi_build_type = "Release"' in ubuntu_24_mpi_config_text
    assert 'wavefront_build_type = "Release"' in ubuntu_24_wf_config_text
    assert 'mpi_build_type = "Release"' in pawsey_mpi_config_text
    assert 'wavefront_build_type = "Debug"' in pawsey_wf_config_text


def test_pawsey_profile_does_not_disable_hipfort_mod_compatibility_check():
    pawsey_hooks_text = PAWSEY_HOOKS.read_text()

    assert "-DHIPFORT_VERIFY_MOD_COMPAT=OFF" not in pawsey_hooks_text


def test_cmake_no_longer_defaults_install_prefix_to_source_tree():
    cmake_lists_text = CMAKE_LISTS.read_text()

    assert 'set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}")' not in cmake_lists_text


def test_build_and_inspect_wheel_keeps_stdout_for_wheel_path_only():
    install_text = INSTALL_SH.read_text()
    wheel_text = WHEEL_LIB.read_text()

    assert 'WHEEL_PATH="$(build_and_inspect_wheel)"' in install_text
    assert 'python -m build --wheel --no-isolation --outdir "$wheel_dir" >&2' in wheel_text
    assert 'echo "Error: expected exactly one wheel in $wheel_dir, found ${#wheel_files[@]}" >&2' in wheel_text
    assert 'printf \'  %s\\n\' "${wheel_files[@]}" >&2' in wheel_text
    assert 'printf \'%s\\n\' "$final_wheel"' in wheel_text


def test_install_script_repairs_linux_wavefront_wheels_with_auditwheel():
    install_text = INSTALL_SH.read_text()
    wheel_text = WHEEL_LIB.read_text()

    assert 'python -m pip install --upgrade auditwheel' in install_text
    assert 'collect_auditwheel_excludes()' in wheel_text
    assert 'assert_wheel_supported_by_current_python()' in wheel_text
    assert 'repair_linux_wheel()' in wheel_text
    assert 'run_auditwheel "$auditwheel_ld_path" show "$wheel_path" >&2 || return 1' in wheel_text
    assert 'final_wheel="$(repair_linux_wheel "${wheel_files[0]}")" || return 1' in wheel_text
    assert 'REQUIRED_WHEEL_MEMBER_SUBSTRING="libshafft" inspect_built_wheel "$final_wheel" >/dev/null || return 1' in wheel_text
    assert 'assert_wheel_supported_by_current_python "${wheel_files[0]}" || return 1' in wheel_text


def test_install_script_aborts_on_wheel_or_install_validation_failure():
    install_text = INSTALL_SH.read_text()
    wheel_text = WHEEL_LIB.read_text()

    assert 'inspect_built_wheel "${wheel_files[0]}" >/dev/null || return 1' in wheel_text
    assert 'final_wheel="$(repair_linux_wheel "${wheel_files[0]}")" || return 1' in wheel_text
    assert 'if ! WHEEL_PATH="$(build_and_inspect_wheel)"; then' in install_text
    assert 'if ! INSTALLED_PACKAGE_PATH="$(validate_installed_package)"; then' in install_text


def test_install_script_uses_common_environment_helpers():
    install_text = INSTALL_SH.read_text()

    assert "readlink -f" not in install_text
    assert "run_profile_environment_hooks" in install_text
    assert "prepare_profile_venv_args" in install_text
    assert "export_profile_cmake_args" in install_text
    assert "cleanup_install_state" in install_text
    assert "ensure_python_build_requirements" in install_text
    assert 'ensure_path_within_root "$INSTALL_ROOT" "$VENV_DIR" "virtual environment"' in install_text
    assert 'ensure_path_within_root "$INSTALL_ROOT" "$PROFILE_WORK_DIR" "profile work directory"' in install_text
    assert 'ensure_path_within_root "$INSTALL_ROOT" "$ACTIVATION_SCRIPT" "activation script"' in install_text


def test_install_script_builds_docs_in_separate_venv_from_pyproject_metadata():
    install_text = INSTALL_SH.read_text()
    post_install_text = POST_INSTALL_LIB.read_text()
    install_tail = install_text[install_text.index('step "Copying examples and benchmarks"'):]

    assert 'DOCS_VENV_DIR="$PROFILE_WORK_DIR/docs-venv"' in install_text
    assert 'ensure_path_within_root "$INSTALL_ROOT" "$DOCS_VENV_DIR" "documentation virtual environment"' in install_text
    assert '"$CONFIG_PYTHON" "$PYPROJECT_DEPS_HELPER" docs-build "$PYPROJECT_TOML"' in post_install_text
    assert '"$CONFIG_PYTHON" -m venv "$DOCS_VENV_DIR"' in post_install_text
    assert 'local docs_python="$DOCS_VENV_DIR/bin/python"' in post_install_text
    assert '"$docs_python" -m sphinx -b html "$src_docs/source" "$dst_docs"' in post_install_text
    assert "docs/requirements.txt" not in install_text
    assert "docs/requirements.txt" not in post_install_text
    assert install_tail.index('build_docs') < install_tail.index('write_manifest')


def test_install_script_help_mentions_prefix_option():
    install_text = INSTALL_SH.read_text()

    assert "--prefix DIR" in install_text
    assert "--veryclean" in install_text
    assert "Relative paths use the caller's cwd." in install_text
    assert 'INSTALL_ROOT="$(expand_install_path "$INSTALL_PREFIX" "$CALLER_CWD")"' in install_text


def test_install_script_help_works_from_outside_repo():
    shells = ["bash"]
    if ZSH is not None:
        shells.append(ZSH)

    for shell in shells:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [shell, str(INSTALL_SH), "--help"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
            )

        assert result.returncode == 0, (shell, result.stderr or result.stdout)
        assert "--prefix DIR" in result.stdout
        assert "--veryclean" in result.stdout


def test_install_script_clean_modes_only_remove_cache_and_deps():
    install_text = INSTALL_SH.read_text()

    assert 'rm -rf "$CACHE_ROOT"' in install_text
    assert 'rm -rf "$CACHE_ROOT" "$DEPS_ROOT"' in install_text
    assert 'rm -rf "$VENV_DIR"' not in install_text


def test_install_script_stages_activation_runtime_inside_prefix():
    install_text = INSTALL_SH.read_text()
    activation_text = ACTIVATION_LIB.read_text()

    assert 'prepare_activation_runtime' in install_text
    assert 'cp "$COMMON_LIB" "$ACTIVATION_COMMON_LIB"' in activation_text
    assert 'cp "$CONFIG_RENDERER" "$ACTIVATION_CONFIG_RENDERER"' in activation_text
    assert 'cp "$PATH_HELPER" "$ACTIVATION_PATH_HELPER"' in activation_text
    assert 'quop_shell_source_path() {' in activation_text
    assert 'if ! quop_shell_is_sourced; then' in activation_text
    assert 'INSTALL_ROOT="\\$(cd -- "\\$(dirname -- "\\$(quop_shell_source_path)")" && pwd)"' in activation_text
    assert 'CONFIG_PYTHON="\\$VENV_DIR/bin/python"' in activation_text


def test_shafft_dependency_requires_matching_cache_stamp():
    shafft_text = SHAFFT_DEPENDENCY_CMAKE.read_text()

    assert "quop_shafft_stamp.cmake" in shafft_text
    assert "_shafft_check_cache_stamp" in shafft_text
    assert "_shafft_write_cache_stamp" in shafft_text
    assert "Use a new install prefix or rerun the installer with --veryclean." in shafft_text
    assert "SHAFFT_GIT_TAG" in shafft_text
    assert "OFFLOAD_ARCH" in shafft_text
    assert "ROCM_PATH" in shafft_text


def test_install_script_rejects_project_root_prefix_before_build_steps():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["CONFIG_PYTHON"] = sys.executable
        result = subprocess.run(
            ["bash", str(INSTALL_SH), "-p", "generic", "--prefix", str(PROJECT_ROOT)],
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode != 0
    assert "install prefix must not be the project root" in (result.stdout + result.stderr)


def test_export_profile_cmake_args_exports_space_separated_string():
    script = f"""
source {shlex.quote(str(COMMON_SH))}
BACKEND=wavefront
profile_cmake_args() {{
cat <<'ARGS'
-DWAVEFRONT_BACKEND=ON
-DCMAKE_PREFIX_PATH=/tmp/a;/tmp/b
ARGS
}}
export_profile_cmake_args
printf 'CMAKE_ARGS=%s\\n' "$CMAKE_ARGS"
printf 'CMAKE_ARGS_ARRAY=%s\\n' "${{CMAKE_ARGS_ARRAY[*]}}"
"""

    result = run_shell(script)

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.splitlines() == [
        "CMAKE_ARGS=-DBUILD_TESTING=ON -DWAVEFRONT_BACKEND=ON -DCMAKE_PREFIX_PATH=/tmp/a;/tmp/b",
        "CMAKE_ARGS_ARRAY=-DBUILD_TESTING=ON -DWAVEFRONT_BACKEND=ON -DCMAKE_PREFIX_PATH=/tmp/a;/tmp/b",
    ]


def test_export_profile_cmake_args_rejects_backend_mismatch():
    script = f"""
source {shlex.quote(str(COMMON_SH))}
BACKEND=wavefront
profile_cmake_args() {{
    echo -DWAVEFRONT_BACKEND=OFF
}}
export_profile_cmake_args
"""

    result = run_shell(script)

    assert result.returncode != 0
    assert "expected backend flag: -DWAVEFRONT_BACKEND=ON" in (result.stdout + result.stderr)


def test_prepare_profile_venv_args_resets_then_runs_profile_hook():
    script = f"""
source {shlex.quote(str(COMMON_SH))}
VENV_ARGS=(--stale)
profile_configure_venv() {{
    VENV_ARGS+=(--system-site-packages)
}}
prepare_profile_venv_args
printf '%s\\n' "${{VENV_ARGS[*]}}"
"""

    result = run_shell(script)

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "--system-site-packages"


def test_ubuntu_h5py_uses_pkgconfig_without_inheriting_hdf5_dir():
    script = f"""
tmpbin="$(mktemp -d)"
trap 'rm -rf "$tmpbin"' EXIT
printf '%s\\n' '#!/usr/bin/env bash' \
    'if [[ "$1" == "--exists" && "$2" == "hdf5-openmpi" ]]; then exit 0; fi' \
    'if [[ "$1" == "--exists" && "$2" == "hdf5" ]]; then exit 1; fi' \
    'exit 1' >"$tmpbin/pkg-config"
printf '%s\\n' '#!/usr/bin/env bash' \
    'if [[ "$1" == "--showme:compile" ]]; then printf "%s\\n" "-I/fake/mpi/include"; fi' >"$tmpbin/mpicc"
printf '%s\\n' '#!/usr/bin/env bash' \
    'env | grep "^HDF5" | sort' >"$tmpbin/python"
chmod +x "$tmpbin/pkg-config" "$tmpbin/mpicc" "$tmpbin/python"
PATH="$tmpbin:$PATH"
source {shlex.quote(str(COMMON_SH))}
source {shlex.quote(str(UBUNTU_24_HOOKS))}
info() {{ :; }}
CC=/usr/bin/gcc
export HDF5_DIR=/usr
profile_install_h5py
"""

    result = run_shell(script)

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.splitlines() == [
        "HDF5_MPI=ON",
        "HDF5_PKGCONFIG_NAME=hdf5-openmpi",
    ]


def test_ubuntu_h5py_falls_back_to_hdf5_dir_without_inheriting_pkgconfig_name():
    script = f"""
tmpbin="$(mktemp -d)"
trap 'rm -rf "$tmpbin"' EXIT
printf '%s\\n' '#!/usr/bin/env bash' 'exit 1' >"$tmpbin/pkg-config"
printf '%s\\n' '#!/usr/bin/env bash' \
    'if [[ "$1" == "--showme:compile" ]]; then printf "%s\\n" "-I/fake/mpi/include"; fi' >"$tmpbin/mpicc"
printf '%s\\n' '#!/usr/bin/env bash' \
    'env | grep "^HDF5" | sort' >"$tmpbin/python"
chmod +x "$tmpbin/pkg-config" "$tmpbin/mpicc" "$tmpbin/python"
PATH="$tmpbin:$PATH"
source {shlex.quote(str(COMMON_SH))}
source {shlex.quote(str(UBUNTU_24_HOOKS))}
info() {{ :; }}
CC=/usr/bin/gcc
export HDF5_DIR=/usr
export HDF5_PKGCONFIG_NAME=stale
profile_install_h5py
"""

    result = run_shell(script)

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.splitlines() == [
        "HDF5_DIR=/usr",
        "HDF5_MPI=ON",
    ]


def test_run_profile_environment_hooks_applies_wavefront_sequence():
    script = f"""
source {shlex.quote(str(COMMON_SH))}
BACKEND=wavefront
profile_load_modules() {{ printf 'common\\n'; }}
profile_load_modules_gpu() {{ printf 'gpu\\n'; }}
profile_post_modules_env() {{ printf 'post\\n'; }}
run_profile_environment_hooks
"""

    result = run_shell(script)

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.splitlines() == ["common", "gpu", "post"]


def test_activation_script_rehydrates_profile_cmake_args():
    activation_text = ACTIVATION_LIB.read_text()

    assert 'source "\\$COMMON_LIB" || return 1' in activation_text
    assert 'source "\\$VENV_DIR/bin/activate" || return 1' in activation_text
    assert 'export_profile_cmake_args || return 1' in activation_text


def test_generated_activation_script_works_in_zsh():
    if ZSH is None:
        return

    script = f"""
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
mkdir -p "$tmpdir/.venv_generic_mpi/bin" "$tmpdir/.cache/quop/generic/mpi/deps" "$tmpdir/.cache/quop/generic/mpi/skbuild"
printf '%s\\n' 'export VIRTUAL_ENV="$VENV_DIR"' >"$tmpdir/.venv_generic_mpi/bin/activate"
source {shlex.quote(str(COMMON_SH))}
source {shlex.quote(str(ACTIVATION_LIB))}
INSTALL_ROOT="$tmpdir"
ACTIVATION_RUNTIME_DIR="$tmpdir/.environments-runtime/generic/mpi"
ACTIVATION_COMMON_LIB="$ACTIVATION_RUNTIME_DIR/common.sh"
ACTIVATION_CONFIG_RENDERER="$ACTIVATION_RUNTIME_DIR/render_config.py"
ACTIVATION_PATH_HELPER="$ACTIVATION_RUNTIME_DIR/path_helper.py"
PROFILE_FILE={shlex.quote(str(GENERIC_HOOKS))}
ACTIVATION_PROFILE_FILE="$ACTIVATION_RUNTIME_DIR/hooks.sh"
COMMON_LIB={shlex.quote(str(COMMON_SH))}
CONFIG_RENDERER={shlex.quote(str(PROJECT_ROOT / "environments" / "lib" / "render_config.py"))}
PATH_HELPER={shlex.quote(str(PROJECT_ROOT / "environments" / "lib" / "path_helper.py"))}
PROFILE=generic
PROFILE_ID=generic
BACKEND=mpi
VENV_DIR="$tmpdir/.venv_generic_mpi"
CONFIG_PYTHON={shlex.quote(sys.executable)}
PROFILE_WORK_DIR="$tmpdir/.cache/quop/generic/mpi"
BUILD_DEPS_DIR="$PROFILE_WORK_DIR/deps"
FETCHCONTENT_BASE_DIR="$tmpdir/.deps/generic/mpi"
SKBUILD_BUILD_DIR="$PROFILE_WORK_DIR/skbuild"
ACTIVATION_CONFIG_FILE=""
prepare_activation_runtime
write_activation_script "$tmpdir/activate-generic-mpi.sh"
cd /
source "$tmpdir/activate-generic-mpi.sh"
printf 'install=%s\\n' "$QUOP_INSTALL_ROOT"
printf 'venv=%s\\n' "$QUOP_VENV_DIR"
printf 'backend=%s\\n' "$QUOP_BACKEND"
printf 'cmake=%s\\n' "$CMAKE_ARGS"
"""

    result = run_shell(script, shell=ZSH)

    assert result.returncode == 0, result.stderr or result.stdout
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    install_root = lines[0].split("=", 1)[1]
    venv_dir = lines[1].split("=", 1)[1]
    assert install_root
    assert lines[0] == f"install={install_root}"
    assert lines[1] == f"venv={venv_dir}"
    assert venv_dir == f"{install_root}/.venv_generic_mpi"
    assert lines[2] == "backend=mpi"
    assert lines[3] == (
        "cmake=-DBUILD_TESTING=ON -DWAVEFRONT_BACKEND=OFF "
        "-DWITH_SHAFFT=OFF -DGPU_AWARE_MPI=OFF -DCMAKE_BUILD_TYPE=Release"
    )


def test_readme_docs_build_instructions_use_python_module_invocation():
    readme_text = (PROJECT_ROOT / "README.rst").read_text()

    assert "python -m pip install '.[docs]'" in readme_text
    assert "python -m sphinx -b html docs/source docs/build/html" in readme_text


def test_validate_install_rejects_source_tree_origin():
    validator = load_module(VALIDATE_INSTALL, "validate_install")

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        source_init = project_root / "src" / "quop_mpi" / "__init__.py"
        source_init.parent.mkdir(parents=True)
        source_init.write_text("# source package\n")

        try:
            validator.ensure_not_source_tree(source_init, project_root)
        except validator.ValidationError as exc:
            assert "source tree" in str(exc)
        else:
            raise AssertionError("expected source-tree validation to fail")


def test_docs_conf_uses_src_layout_for_autodoc_imports():
    docs_conf_text = (PROJECT_ROOT / "docs" / "source" / "conf.py").read_text()

    assert 'sys.path.insert(0, os.path.abspath("../../src"))' in docs_conf_text


def test_validate_install_accepts_extension_suffix_matches():
    validator = load_module(VALIDATE_INSTALL, "validate_install")

    with tempfile.TemporaryDirectory() as tmpdir:
        package_root = Path(tmpdir) / "quop_mpi"
        extension_dir = package_root / "_lib" / "wavefront"
        extension_dir.mkdir(parents=True)
        extension_file = extension_dir / f"context_wrapper{validator.EXTENSION_SUFFIXES[0]}"
        extension_file.write_text("")

        assert validator.has_extension_module(package_root, "quop_mpi/_lib/wavefront/context_wrapper")


def test_validate_install_reports_missing_required_extensions():
    validator = load_module(VALIDATE_INSTALL, "validate_install")

    with tempfile.TemporaryDirectory() as tmpdir:
        package_root = Path(tmpdir) / "quop_mpi"
        (package_root / "_lib" / "mpi").mkdir(parents=True)
        (package_root / "_lib" / f"comm_info_wrapper{validator.EXTENSION_SUFFIXES[0]}").write_text("")

        try:
            validator.ensure_required_extensions(package_root, "mpi")
        except validator.ValidationError as exc:
            assert "cartesian" in str(exc)
            assert "csr_generators" in str(exc)
            assert "parallel_io" in str(exc)
            assert "context_wrapper" in str(exc)
            assert "mpi_diagonal_propagator" in str(exc)
            assert "mpi_sparse_propagator" in str(exc)
            assert "mpi_circulant_propagator" in str(exc)
            assert "mpi_composite_propagator" in str(exc)
            assert "mpi_momentum_propagator" in str(exc)
            assert "mpi_transverse_field_propagator" in str(exc)
        else:
            raise AssertionError("expected missing extension validation to fail")


def test_validate_install_requires_full_mpi_backend():
    validator = load_module(VALIDATE_INSTALL, "validate_install")

    required = validator.required_extension_stems("mpi")

    assert required == (
        "quop_mpi/_lib/cartesian",
        "quop_mpi/_lib/comm_info_wrapper",
        "quop_mpi/_lib/csr_generators",
        "quop_mpi/_lib/parallel_io",
        "quop_mpi/_lib/mpi/context_wrapper",
        "quop_mpi/_lib/mpi/mpi_diagonal_propagator",
        "quop_mpi/_lib/mpi/mpi_sparse_propagator",
        "quop_mpi/_lib/mpi/mpi_circulant_propagator",
        "quop_mpi/_lib/mpi/mpi_composite_propagator",
        "quop_mpi/_lib/mpi/mpi_momentum_propagator",
        "quop_mpi/_lib/mpi/mpi_transverse_field_propagator",
    )


def test_validate_install_requires_full_wavefront_backend():
    validator = load_module(VALIDATE_INSTALL, "validate_install")

    required = validator.required_extension_stems("wavefront")

    assert required == (
        "quop_mpi/_lib/cartesian",
        "quop_mpi/_lib/comm_info_wrapper",
        "quop_mpi/_lib/csr_generators",
        "quop_mpi/_lib/parallel_io",
        "quop_mpi/_lib/wavefront/context_wrapper",
        "quop_mpi/_lib/wavefront/wavefront_diagonal_propagator",
        "quop_mpi/_lib/wavefront/wavefront_sparse_propagator",
        "quop_mpi/_lib/wavefront/wavefront_circulant_propagator",
        "quop_mpi/_lib/wavefront/wavefront_composite_propagator",
        "quop_mpi/_lib/wavefront/wavefront_momentum_propagator",
    )


def test_validate_install_no_longer_requires_fixed_vendor_lib_paths():
    validator = load_module(VALIDATE_INSTALL, "validate_install")

    assert not hasattr(validator, "required_vendor_libs")


CONFIG_RENDERER = PROJECT_ROOT / "environments" / "lib" / "render_config.py"


def test_render_config_benchmark_section_round_trips_without_warnings():
    """A config.toml with a valid [benchmarks] section (nested per-algorithm
    sub-tables) produces CFG_BENCHMARKS_* variables and no stderr warnings."""
    config_content = """\
[profile]
description = "test"

[install]
root = ".quop-install"

[benchmarks]
scheduler = "slurm"
partition = "work"
account_suffix = "-gpu"
walltime = "00:20:00"
ranks_per_node = 128
ranks_per_gcd = 8
gcds_per_node = 8
intra_sequence = [1, 2, 4, 8]
multi_nodes = [1, 2, 4]
exports = ["MPICH_OFI_STARTUP_CONNECT=1"]
srun_extra_args = "-m block:block:block"

[benchmarks.qaoa]
args_intra = [65536, 131072, 262144]
args_multi = [65536, 131072, 262144]

[benchmarks.qwoa]
args_intra = [65536, 131072]
args_multi = [65536, 131072]

[benchmarks.qmoa]
args_intra = ["3 3 3 3 3", "4 4 4 4"]
args_multi = ["3 3 3 3 3", "4 4 4 4"]
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, str(CONFIG_RENDERER), config_path],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Warning" not in result.stderr
        assert "CFG_BENCHMARKS_SCHEDULER" in result.stdout
        assert "CFG_BENCHMARKS_PARTITION" in result.stdout
        assert "CFG_BENCHMARKS_RANKS_PER_NODE" in result.stdout
        assert "CFG_BENCHMARKS_INTRA_SEQUENCE" in result.stdout
        assert "CFG_BENCHMARKS_EXPORTS" in result.stdout
        assert "CFG_BENCHMARKS_SRUN_EXTRA_ARGS" in result.stdout
        assert "CFG_BENCHMARKS_QAOA_ARGS_INTRA" in result.stdout
        assert "CFG_BENCHMARKS_QWOA_ARGS_INTRA" in result.stdout
        assert "CFG_BENCHMARKS_QMOA_ARGS_INTRA" in result.stdout
        assert "CFG_BENCHMARKS_QAOA_ARGS_MULTI" in result.stdout
        assert "CFG_BENCHMARKS_QWOA_ARGS_MULTI" in result.stdout
        assert "CFG_BENCHMARKS_QMOA_ARGS_MULTI" in result.stdout
    finally:
        os.unlink(config_path)


def test_render_config_benchmark_typo_produces_warning():
    """A typo in a [benchmarks] key should produce an unrecognised-key warning."""
    config_content = """\
[profile]
description = "test"

[install]
root = ".quop-install"

[benchmarks]
scheduler = "local"
ranks_per_node = 4
intra_sequence = [1, 2, 4]
typo_key = "oops"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, str(CONFIG_RENDERER), config_path],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "unrecognised config key 'typo_key' in [benchmarks]" in result.stderr
    finally:
        os.unlink(config_path)


def test_all_profile_configs_have_benchmark_section():
    """Every per-backend config.toml must include a [benchmarks] section with
    at least a scheduler key and per-algorithm args_intra sub-tables."""
    profiles_dir = PROJECT_ROOT / "environments" / "profiles"
    found_configs = list(profiles_dir.glob("*/*/config.toml"))
    assert len(found_configs) >= 4, (
        f"Expected at least 4 per-backend config files, found {len(found_configs)}"
    )
    for config_path in found_configs:
        data = tomllib.loads(config_path.read_text())
        benchmarks = data.get("benchmarks", {})
        assert "scheduler" in benchmarks, (
            f"Missing scheduler in [benchmarks] in {config_path}"
        )
        for algo in ("qaoa", "qwoa", "qmoa"):
            algo_table = benchmarks.get(algo, {})
            assert "args_intra" in algo_table, (
                f"Missing args_intra in [benchmarks.{algo}] in {config_path}"
            )
