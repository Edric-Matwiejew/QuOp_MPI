#!/usr/bin/env python3
"""Render a TOML config file as shell variable assignments.

Requires Python 3.11+ (uses the built-in ``tomllib`` module).
"""

import shlex
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit(
        f"Error: Python 3.11+ is required for tomllib support "
        f"(running {sys.version_info.major}.{sys.version_info.minor})"
    )

import tomllib


def normalize_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    return value


def shell_quote(value):
    return shlex.quote(str(normalize_value(value)))


def emit_scalar(name, value):
    """Emit a shell variable assignment.  Skips only None (absent keys).

    Empty strings are emitted so that explicit TOML values like
    ``srun_extra_args = ""`` propagate correctly.  Shell consumers
    that need a fallback should use ``${VAR:-default}`` which treats
    both unset *and* empty the same way.
    """
    if value is None:
        return
    print(f"{name}={shell_quote(value)}")


def emit_array(name, values):
    if not values:
        return
    joined = " ".join(shell_quote(value) for value in values)
    print(f"{name}=({joined})")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: render_config.py <config.toml>")

    config_path = Path(sys.argv[1])
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    # Track which (table, key) pairs are consumed by emit_* calls.
    consumed_keys: set[tuple[str, str]] = set()

    def get_scalar(table_name, table, key):
        consumed_keys.add((table_name, key))
        return table.get(key)

    def get_array(table_name, table, key, default=None):
        consumed_keys.add((table_name, key))
        return table.get(key, default if default is not None else [])

    profile = data.get("profile", {})
    install = data.get("install", {})
    paths = data.get("paths", {})
    build = data.get("build", {})
    compilers = data.get("compilers", {})
    homebrew = data.get("homebrew", {})
    modules = data.get("modules", {})

    emit_scalar("CFG_PROFILE_DESCRIPTION", get_scalar("profile", profile, "description"))
    emit_scalar("CFG_WAVEFRONT_SUPPORTED", get_scalar("profile", profile, "wavefront_supported"))
    emit_scalar("CFG_PYTHON_VERSION", get_scalar("profile", profile, "python_version"))

    emit_scalar("CFG_INSTALL_ROOT", get_scalar("install", install, "root"))
    emit_scalar("CFG_VENV_DIR_TEMPLATE", get_scalar("install", install, "venv_dir"))
    work_dir = get_scalar("install", install, "work_dir")
    build_dir = get_scalar("install", install, "build_dir")
    emit_scalar("CFG_WORK_DIR_TEMPLATE", work_dir if work_dir is not None else build_dir)

    emit_scalar("CFG_SHAFFT_PATH", get_scalar("paths", paths, "shafft_path"))

    emit_scalar("CFG_OFFLOAD_ARCH", get_scalar("build", build, "offload_arch"))
    emit_scalar("CFG_MPI_BUILD_TYPE", get_scalar("build", build, "mpi_build_type"))
    emit_scalar("CFG_WAVEFRONT_BUILD_TYPE", get_scalar("build", build, "wavefront_build_type"))

    emit_scalar("CFG_CC", get_scalar("compilers", compilers, "cc"))
    emit_scalar("CFG_CXX", get_scalar("compilers", compilers, "cxx"))
    emit_scalar("CFG_FC", get_scalar("compilers", compilers, "fc"))
    emit_scalar("CFG_MPI_CC_WRAPPER", get_scalar("compilers", compilers, "mpi_cc_wrapper"))
    emit_scalar("CFG_MPI_CXX_WRAPPER", get_scalar("compilers", compilers, "mpi_cxx_wrapper"))
    emit_scalar("CFG_MPI_FC_WRAPPER", get_scalar("compilers", compilers, "mpi_fc_wrapper"))

    emit_scalar("CFG_HOMEBREW_PREFIX", get_scalar("homebrew", homebrew, "prefix"))
    emit_scalar("CFG_HOMEBREW_HDF5_FORMULA", get_scalar("homebrew", homebrew, "hdf5_formula"))
    emit_scalar("CFG_HOMEBREW_FFTW_FORMULA", get_scalar("homebrew", homebrew, "fftw_formula"))

    emit_array("CFG_PROFILE_MODULES_COMMON", get_array("modules", modules, "common"))
    emit_array("CFG_PROFILE_MODULES_WAVEFRONT", get_array("modules", modules, "wavefront"))
    emit_array("CFG_PROFILE_MODULES_MPI_PYTHON", get_array("modules", modules, "mpi_python"))

    # ---- Benchmarks (nested tables) ----------------------------------------
    # [benchmarks]           -> shared keys (scheduler, partition, etc.)
    # [benchmarks.<algo>]    -> per-algorithm overrides + args_intra/args_multi
    benchmarks = data.get("benchmarks", {})
    known_algos = {"qaoa", "qwoa", "qmoa"}
    known_benchmark_keys = {
        "scheduler", "partition", "account_suffix", "walltime",
        "ranks_per_node", "ranks_per_gcd", "gcds_per_node",
        "intra_sequence", "multi_nodes",
        "exports", "srun_extra_args",
    }
    known_algo_keys = known_benchmark_keys | {"args_intra", "args_multi"}

    # Shared benchmark keys (scalars and arrays at the [benchmarks] level).
    for key, value in benchmarks.items():
        if isinstance(value, dict):
            # Sub-table -- handled below.
            continue
        var_name = f"CFG_BENCHMARKS_{key.upper()}"
        if key in known_benchmark_keys:
            consumed_keys.add(("benchmarks", key))
        if isinstance(value, list):
            emit_array(var_name, value)
        else:
            emit_scalar(var_name, value)

    # Per-algorithm sub-tables: [benchmarks.qaoa], [benchmarks.qwoa], etc.
    for algo in known_algos:
        algo_table = benchmarks.get(algo, {})
        consumed_keys.add(("benchmarks", algo))
        for key, value in algo_table.items():
            var_name = f"CFG_BENCHMARKS_{algo.upper()}_{key.upper()}"
            if key in known_algo_keys:
                consumed_keys.add((f"benchmarks.{algo}", key))
            if isinstance(value, list):
                emit_array(var_name, value)
            else:
                emit_scalar(var_name, value)

    # Warn about unrecognised keys in the config file.
    known_tables = {
        "profile", "install", "paths", "build", "compilers", "homebrew", "modules",
        "benchmarks",
    }
    for table_name, table in data.items():
        if table_name not in known_tables:
            print(
                f"Warning: unrecognised config table [{table_name}] in {config_path}",
                file=sys.stderr,
            )
            continue
        if not isinstance(table, dict):
            continue
        if table_name == "benchmarks":
            # Sub-tables are validated by the known_algos set above.
            for key in table:
                if isinstance(table[key], dict) and key not in known_algos:
                    print(
                        f"Warning: unrecognised benchmark algorithm '{key}' "
                        f"in [benchmarks] in {config_path}",
                        file=sys.stderr,
                    )
                elif (table_name, key) not in consumed_keys:
                    print(
                        f"Warning: unrecognised config key '{key}' "
                        f"in [{table_name}] in {config_path}",
                        file=sys.stderr,
                    )
            # Also check keys inside each algo sub-table.
            for algo in known_algos:
                algo_table = table.get(algo, {})
                for key in algo_table:
                    if (f"benchmarks.{algo}", key) not in consumed_keys:
                        print(
                            f"Warning: unrecognised config key '{key}' "
                            f"in [benchmarks.{algo}] in {config_path}",
                            file=sys.stderr,
                        )
        else:
            for key in table:
                if (table_name, key) not in consumed_keys:
                    print(
                        f"Warning: unrecognised config key '{key}' "
                        f"in [{table_name}] in {config_path}",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    main()
