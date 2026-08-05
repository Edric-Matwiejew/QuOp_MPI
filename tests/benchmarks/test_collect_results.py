import importlib.util
from pathlib import Path

import pandas as pd


def load_collect_results_module():
    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "benchmarks" / "collect_results.py"
    spec = importlib.util.spec_from_file_location("benchmark_collect_results", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_speedup_rows_includes_program_wall_time():
    collect_results = load_collect_results_module()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "qaoa_sparse",
                "backend": "mpi",
                "system_size": 1024,
                "phase": "intra",
                "nprocs": 1,
                "prepare_s": 10.0,
                "mean_evolve_s": 8.0,
                "program_wall_s": 40.0,
            },
            {
                "algorithm": "qaoa_sparse",
                "backend": "mpi",
                "system_size": 1024,
                "phase": "intra",
                "nprocs": 2,
                "prepare_s": 5.0,
                "mean_evolve_s": 4.0,
                "program_wall_s": 20.0,
            },
        ]
    )

    summary = collect_results.normalise_metadata(summary)
    result = collect_results.compute_speedup_rows(summary).sort_values("nprocs")

    assert result["program_wall_s"].tolist() == [40.0, 20.0]
    assert result["speedup_program_wall"].tolist() == [1.0, 2.0]


def test_normalise_metadata_adds_program_wall_column_for_legacy_csvs():
    collect_results = load_collect_results_module()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "qaoa_sparse",
                "backend": "mpi",
                "system_size": 1024,
                "phase": "intra",
                "nprocs": 1,
                "prepare_s": 10.0,
                "mean_evolve_s": 8.0,
            }
        ]
    )

    normalised = collect_results.normalise_metadata(summary)

    assert "program_wall_s" in normalised.columns
    assert pd.isna(normalised.loc[0, "program_wall_s"])


def test_normalise_metadata_adds_profile_column_for_legacy_csvs():
    collect_results = load_collect_results_module()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "qaoa_sparse",
                "backend": "mpi",
                "system_size": 1024,
                "phase": "intra",
                "nprocs": 1,
                "prepare_s": 10.0,
                "mean_evolve_s": 8.0,
            }
        ]
    )

    normalised = collect_results.normalise_metadata(summary)

    assert "profile" in normalised.columns
    assert normalised.loc[0, "profile"] == ""


def test_compute_speedup_rows_groups_by_profile_when_present():
    collect_results = load_collect_results_module()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "qaoa_sparse",
                "backend": "mpi",
                "profile": "generic",
                "system_size": 1024,
                "phase": "intra",
                "nprocs": 1,
                "prepare_s": 10.0,
                "mean_evolve_s": 8.0,
                "program_wall_s": 40.0,
            },
            {
                "algorithm": "qaoa_sparse",
                "backend": "mpi",
                "profile": "generic",
                "system_size": 1024,
                "phase": "intra",
                "nprocs": 2,
                "prepare_s": 5.0,
                "mean_evolve_s": 4.0,
                "program_wall_s": 20.0,
            },
            {
                "algorithm": "qaoa_sparse",
                "backend": "mpi",
                "profile": "pawsey-setonix",
                "system_size": 1024,
                "phase": "intra",
                "nprocs": 1,
                "prepare_s": 12.0,
                "mean_evolve_s": 10.0,
                "program_wall_s": 50.0,
            },
            {
                "algorithm": "qaoa_sparse",
                "backend": "mpi",
                "profile": "pawsey-setonix",
                "system_size": 1024,
                "phase": "intra",
                "nprocs": 2,
                "prepare_s": 6.0,
                "mean_evolve_s": 5.0,
                "program_wall_s": 25.0,
            },
        ]
    )

    summary = collect_results.normalise_metadata(summary)
    result = collect_results.compute_speedup_rows(summary).sort_values(["profile", "nprocs"])

    # Each profile should have its own speedup baseline
    generic_rows = result[result["profile"] == "generic"]
    pawsey_rows = result[result["profile"] == "pawsey-setonix"]

    assert generic_rows["speedup_evolve"].tolist() == [1.0, 2.0]
    assert pawsey_rows["speedup_evolve"].tolist() == [1.0, 2.0]
