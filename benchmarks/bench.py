#!/usr/bin/env python
"""
QuOp_MPI Scaling Benchmark
===========================

Measures prepare() and evolve_state() wall-clock time for qaoa,
qaoa_transverse_field, qwoa, or qmoa.

Usage:
    python bench.py <algorithm> <size_arg> [--verify]

Examples:
    python bench.py qaoa 1048576
    python bench.py qaoa_transverse_field 1048576
    python bench.py qwoa 1048576 --verify
    python bench.py qmoa "3 3 3 3 3"
"""

import argparse
import os
import sys

import numpy as np
from mpi4py import MPI

# ---------------------------------------------------------------------------
# Quality function -- a trivial "QuOp Function" whose parameters are auto-bound
# by the interface class (local_i, local_i_offset, system_size).
# ---------------------------------------------------------------------------


def qualities(local_i, local_i_offset, system_size):
    """Return sequential integers / N for the local partition."""
    return np.arange(local_i_offset, local_i_offset + local_i, dtype=np.float64) / system_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_EVOLVE = 10  # number of evolve_state calls to average


def parse_args():
    parser = argparse.ArgumentParser(
        description="QuOp_MPI scaling benchmark",
    )
    parser.add_argument(
        "algorithm",
        choices=["qaoa", "qaoa_transverse_field", "qwoa", "qmoa"],
        help="Algorithm to benchmark.",
    )
    parser.add_argument(
        "size_arg",
        help=(
            "For qaoa/qaoa_transverse_field/qwoa: a single system_size."
            " For qmoa: a quoted space-separated"
            " Ns exponent list."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=["intra", "multi"],
        default="intra",
        help="Scaling phase (intra-node or multi-node). Recorded in CSV.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Record state_norm and expectation_value after the last evolve.",
    )
    return parser.parse_args()


def qmoa_tensor_dims(size_args):
    return [2**n for n in size_args]


def parse_size_arg(algorithm, size_arg):
    if algorithm == "qmoa":
        parts = size_arg.split()
        if not parts:
            raise ValueError(
                "qmoa expects a quoted"
                " space-separated exponent list,"
                " got an empty size_arg"
            )
        try:
            values = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"qmoa size_arg must contain only integers, got {size_arg!r}") from exc

        if any(value < 0 for value in values):
            raise ValueError(
                "qmoa size_arg exponents must be non-negative integers, "
                f"got {size_arg!r}"
            )

        return values

    try:
        value = int(size_arg)
    except ValueError as exc:
        msg = (
            f"{algorithm} expects a single integer"
            f" system_size, got {size_arg!r}"
        )
        raise ValueError(msg) from exc

    if value <= 0:
        raise ValueError(
            f"{algorithm} system_size must be a positive integer, got {size_arg!r}"
        )

    return [value]


def size_spec(algorithm, size_args):
    if algorithm == "qmoa":
        return "x".join(str(arg) for arg in size_args)
    return str(size_args[0])


def tensor_dims_spec(algorithm, size_args):
    if algorithm != "qmoa":
        return ""
    return "x".join(str(dim) for dim in qmoa_tensor_dims(size_args))


def problem_tag(algorithm, system_size, size_spec_value):
    if algorithm == "qmoa":
        return f"{system_size}_{size_spec_value}"
    return str(system_size)


def make_algorithm(name, size_args):
    """Instantiate the algorithm and return (alg, system_size)."""
    if name == "qaoa":
        from quop_mpi.algorithm.combinatorial import QAOA

        if len(size_args) != 1:
            msg = (
                f"qaoa expects exactly 1 size_arg"
                f" (system_size), got"
                f" {len(size_args)}: {size_args}"
            )
            raise ValueError(msg)
        system_size = size_args[0]
        alg = QAOA(system_size)
        return alg, system_size

    if name == "qaoa_transverse_field":
        from quop_mpi.algorithm.combinatorial import QAOATransverseField

        if len(size_args) != 1:
            msg = (
                f"qaoa_transverse_field expects exactly 1 size_arg"
                f" (system_size), got"
                f" {len(size_args)}: {size_args}"
            )
            raise ValueError(msg)
        system_size = size_args[0]
        alg = QAOATransverseField(system_size)
        return alg, system_size

    if name == "qwoa":
        from quop_mpi.algorithm.combinatorial import QWOA

        if len(size_args) != 1:
            msg = (
                f"qwoa expects exactly 1 size_arg"
                f" (system_size), got"
                f" {len(size_args)}: {size_args}"
            )
            raise ValueError(msg)
        system_size = size_args[0]
        alg = QWOA(system_size)
        return alg, system_size

    if name == "qmoa":
        from quop_mpi.algorithm.multivariable import QMOA

        ns = list(size_args)
        alg = QMOA(ns)
        system_size = 1
        for dim in qmoa_tensor_dims(ns):
            system_size *= dim
        return alg, system_size

    raise ValueError(f"Unknown algorithm: {name}")


def csv_filename(
    algorithm, backend, system_size,
    size_spec_value, nprocs, phase,
):
    tag = problem_tag(
        algorithm, system_size, size_spec_value,
    )
    return f"{algorithm}_{backend}_{tag}_{phase}_{nprocs}.csv"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    backend = os.environ.get("QUOP_BACKEND", "mpi")
    profile = os.environ.get("QUOP_PROFILE", "")
    nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))

    # Create the results directory (rank 0 only)
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
    comm.Barrier()

    try:
        size_args = parse_size_arg(args.algorithm, args.size_arg)
    except ValueError as exc:
        if rank == 0:
            print(f"ERROR: {exc}", file=sys.stderr)
        comm.Abort(1)

    # Instantiate algorithm
    try:
        alg, system_size = make_algorithm(args.algorithm, size_args)
    except ValueError as exc:
        if rank == 0:
            print(f"ERROR: {exc}", file=sys.stderr)
        comm.Abort(1)

    # Set trivial qualities (auto-bound via the interface class)
    alg.set_qualities(qualities)

    # Depth 1
    alg.set_depth(1)

    # ----- Time prepare() -----
    comm.Barrier()
    t0 = MPI.Wtime()
    alg.prepare()
    comm.Barrier()
    prepare_s = MPI.Wtime() - t0

    # Build variational parameters (all ones)
    params = np.ones(alg.total_params * alg.ansatz_depth)

    # ----- Time evolve_state() -----
    evolve_times = []
    for _ in range(N_EVOLVE):
        comm.Barrier()
        t0 = MPI.Wtime()
        alg.evolve_state(params)
        comm.Barrier()
        evolve_times.append(MPI.Wtime() - t0)

    mean_evolve_s = np.mean(evolve_times)
    std_evolve_s = np.std(evolve_times)
    size_spec_value = size_spec(args.algorithm, size_args)
    tensor_dims_value = tensor_dims_spec(args.algorithm, size_args)

    # ----- Verification (optional) -----
    state_norm = None
    expectation_value = None
    if args.verify:
        state_norm = alg.get_state_norm()
        expectation_value = alg.get_expectation_value()

    # ----- Write results (rank 0 only) -----
    if rank == 0:
        fname = csv_filename(
            args.algorithm, backend, system_size,
            size_spec_value, nprocs, args.phase,
        )
        fpath = os.path.join(results_dir, fname)

        header_cols = (
            "algorithm,backend,profile,"
            "system_size,size_spec,tensor_dims,"
            "phase,nprocs,nodes,"
            "prepare_s,mean_evolve_s,std_evolve_s"
        )
        data_cols = (
            f"{args.algorithm},{backend},{profile},{system_size},{size_spec_value},{tensor_dims_value},"
            f"{args.phase},{nprocs},{nodes},{prepare_s:.6f},{mean_evolve_s:.6f},{std_evolve_s:.6f}"
        )

        if args.verify:
            header_cols += ",state_norm,expectation_value"
            data_cols += f",{state_norm:.16e},{expectation_value:.16e}"

        with open(fpath, "w") as f:
            f.write(header_cols + "\n")
            f.write(data_cols + "\n")

        print(f"Results written to {fpath}")
        print(f"  prepare_s      = {prepare_s:.6f}")
        print(f"  mean_evolve_s  = {mean_evolve_s:.6f}")
        print(f"  std_evolve_s   = {std_evolve_s:.6f}")
        if args.verify:
            print(f"  state_norm     = {state_norm:.16e}")
            print(f"  expectation    = {expectation_value:.16e}")

    # ----- Clean up -----
    alg.destroy()


if __name__ == "__main__":
    main()
