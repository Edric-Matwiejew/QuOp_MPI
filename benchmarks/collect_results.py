#!/usr/bin/env python
"""
Aggregate QuOp_MPI benchmark results.

Globs all per-job CSV files from benchmarks/results/, concatenates them into
summary.csv, computes speedup within each problem group, and prints a table
to stdout.

Usage:
    python benchmarks/collect_results.py
"""

import glob
import os
import sys

import pandas as pd


def normalise_metadata(df):
    if "profile" not in df.columns:
        df["profile"] = ""
    else:
        df["profile"] = df["profile"].fillna("").astype(str)

    if "size_spec" not in df.columns:
        df["size_spec"] = df["system_size"].astype(str)
    else:
        df["size_spec"] = df["size_spec"].where(df["size_spec"].notna(), df["system_size"])
        df["size_spec"] = df["size_spec"].astype(str)

    if "tensor_dims" not in df.columns:
        df["tensor_dims"] = ""
    else:
        df["tensor_dims"] = df["tensor_dims"].fillna("").astype(str)

    if "program_wall_s" not in df.columns:
        df["program_wall_s"] = pd.NA
    df["program_wall_s"] = pd.to_numeric(df["program_wall_s"], errors="coerce")

    return df


def problem_label(algorithm, system_size, size_spec, tensor_dims):
    if algorithm == "qmoa" or str(size_spec) != str(system_size):
        label = f"system_size={system_size}, size_spec={size_spec}"
        if tensor_dims:
            label += f", tensor_dims={tensor_dims}"
        return label
    return f"N={system_size}"


def compute_speedup_rows(summary):
    group_cols = ["algorithm", "backend", "system_size", "size_spec", "phase"]
    if "profile" in summary.columns and summary["profile"].str.len().gt(0).any():
        group_cols.insert(2, "profile")
    groups = summary.groupby(group_cols, dropna=False)

    speedup_rows = []
    for _, group in groups:
        group = group.sort_values("nprocs")
        ref_evolve = group["mean_evolve_s"].iloc[0]
        ref_prepare = group["prepare_s"].iloc[0]
        ref_program_wall = group["program_wall_s"].iloc[0]

        for _, row in group.iterrows():
            r = row.to_dict()
            r["speedup_evolve"] = (
                ref_evolve / row["mean_evolve_s"]
                if row["mean_evolve_s"] > 0
                else float("inf")
            )
            r["speedup_prepare"] = (
                ref_prepare / row["prepare_s"]
                if row["prepare_s"] > 0
                else float("inf")
            )
            wall_ok = (
                pd.notna(ref_program_wall)
                and ref_program_wall > 0
                and pd.notna(row["program_wall_s"])
                and row["program_wall_s"] > 0
            )
            if wall_ok:
                r["speedup_program_wall"] = (
                    ref_program_wall
                    / row["program_wall_s"]
                )
            else:
                r["speedup_program_wall"] = pd.NA
            speedup_rows.append(r)

    return pd.DataFrame(speedup_rows)


def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    pattern = os.path.join(results_dir, "*.csv")

    csv_files = sorted(glob.glob(pattern))
    # Exclude summary.csv itself
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("summary")]

    if not csv_files:
        print(f"No CSV files found in {results_dir}/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} result files.")

    # Read and concatenate
    frames = []
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
            frames.append(df)
        except Exception as e:
            print(f"WARNING: Could not read {fpath}: {e}", file=sys.stderr)

    if not frames:
        print("No valid CSV data found.", file=sys.stderr)
        sys.exit(1)

    summary = pd.concat(frames, ignore_index=True)
    summary = normalise_metadata(summary)

    # Compute speedup within each group
    # Reference = the run with the fewest nprocs in each group
    result = compute_speedup_rows(summary)

    # Sort for display
    sort_cols = ["algorithm", "backend", "system_size", "size_spec", "phase", "nprocs"]
    has_profile = "profile" in result.columns and result["profile"].str.len().gt(0).any()
    if has_profile:
        sort_cols.insert(2, "profile")
    result = result.sort_values(sort_cols)

    # Write summary CSV
    summary_path = os.path.join(results_dir, "summary.csv")
    result.to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path}\n")

    # Print table
    display_cols = [
        "algorithm",
        "backend",
        "system_size",
        "size_spec",
        "phase",
        "nodes",
        "nprocs",
        "prepare_s",
        "mean_evolve_s",
        "std_evolve_s",
        "program_wall_s",
        "speedup_evolve",
        "speedup_program_wall",
    ]
    if has_profile:
        display_cols.insert(2, "profile")
    if "tensor_dims" in result.columns and result["tensor_dims"].str.len().gt(0).any():
        display_cols.insert(4, "tensor_dims")
    if not result["program_wall_s"].notna().any():
        exclude = {
            "program_wall_s",
            "speedup_program_wall",
        }
        display_cols = [
            col for col in display_cols
            if col not in exclude
        ]
    # Add verification columns if present
    if "state_norm" in result.columns:
        display_cols.append("state_norm")
    if "expectation_value" in result.columns:
        display_cols.append("expectation_value")

    avail_cols = [c for c in display_cols if c in result.columns]
    print(result[avail_cols].to_string(index=False))

    # Identify scaling knee per group
    # Uses parallel efficiency: efficiency = speedup / resource_ratio.
    # The knee is where efficiency drops below 50 %.
    print("\n--- Scaling Knee Detection ---")
    knee_group_cols = ["algorithm", "backend", "system_size", "size_spec", "phase"]
    if has_profile:
        knee_group_cols.insert(2, "profile")
    for _, group in result.groupby(knee_group_cols, dropna=False):
        group = group.sort_values("nprocs")
        alg = group["algorithm"].iloc[0]
        backend = group["backend"].iloc[0]
        ssize = group["system_size"].iloc[0]
        size_spec = group["size_spec"].iloc[0]
        phase = group["phase"].iloc[0]
        ref_nprocs = group["nprocs"].iloc[0]
        knee = None
        for _, row in group.iterrows():
            resource_ratio = row["nprocs"] / ref_nprocs
            efficiency = row["speedup_evolve"] / resource_ratio if resource_ratio > 0 else 1.0
            if resource_ratio > 1.0 and efficiency < 0.5:
                knee = row
                break

        plabel = problem_label(
            alg, ssize, size_spec,
            group["tensor_dims"].iloc[0],
        )
        if knee is not None:
            knee_ratio = knee["nprocs"] / ref_nprocs
            knee_eff = knee["speedup_evolve"] / knee_ratio
            print(
                f"  {alg} / {backend}"
                f" / {plabel} / {phase}: "
                f"knee at nprocs={int(knee['nprocs'])} "
                f"(speedup={knee['speedup_evolve']:.2f},"
                f" efficiency={knee_eff:.1%})"
            )
        else:
            print(
                f"  {alg} / {backend}"
                f" / {plabel} / {phase}: "
                "no knee detected"
                " (still scaling well)"
            )


if __name__ == "__main__":
    main()
