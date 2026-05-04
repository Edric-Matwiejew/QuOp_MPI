#!/usr/bin/env python3
"""Measure QAOASparse host memory growth across lifecycle phases.

Launch under MPI after selecting the backend, for example:

  QUOP_BACKEND=mpi mpiexec -n 2 python tools/measure_qaoa_sparse_memory.py
  QUOP_BACKEND=wavefront mpiexec -n 2 python tools/measure_qaoa_sparse_memory.py

The sparse hypercube mixer is unit-valued, so compare results against
``estimate_memory.py --unit-valued``.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from quop_mpi import config
from quop_mpi.algorithm.combinatorial import QAOASparse


@dataclass
class MemorySample:
    label: str
    rss_kib: int
    hwm_kib: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system-size",
        type=int,
        default=256,
        help="QAOASparse system size (must be a power of two)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="QAOA depth to prepare and evolve",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.23,
        help="phase-separation parameter value used in evolve_state",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.41,
        help="mixer parameter value used in evolve_state",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="number of repeated evolve_state calls to sample after prepare",
    )
    return parser.parse_args()


def read_proc_status_memory_kib() -> tuple[int, int]:
    rss_kib = 0
    hwm_kib = 0

    with open("/proc/self/status", encoding="ascii") as status_file:
        for line in status_file:
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kib = int(line.split()[1])

    return rss_kib, hwm_kib


def take_sample(label: str, comm: MPI.Intracomm) -> MemorySample:
    comm.Barrier()
    rss_kib, hwm_kib = read_proc_status_memory_kib()
    comm.Barrier()
    return MemorySample(label=label, rss_kib=rss_kib, hwm_kib=hwm_kib)


def print_samples(samples: list[MemorySample], comm: MPI.Intracomm) -> None:
    gathered = comm.gather(samples, root=0)

    if comm.Get_rank() != 0:
        return

    print(f"backend={config.backend} ranks={comm.Get_size()}")
    print(
        f"QAOASparse measurement: system_size={ARGS.system_size} depth={ARGS.depth} "
        f"gamma={ARGS.gamma} beta={ARGS.beta} repeats={ARGS.repeats}"
    )
    print()

    labels = [sample.label for sample in gathered[0]]
    print(
        f"{'phase':<20} {'max_rss_mib':>12} {'sum_rss_mib':>12} "
        f"{'max_hwm_mib':>12} {'delta_max_rss_mib':>18}"
    )
    print("-" * 78)

    baseline_max_rss = max(rank_samples[0].rss_kib for rank_samples in gathered)

    for index, label in enumerate(labels):
        rss_values = [rank_samples[index].rss_kib for rank_samples in gathered]
        hwm_values = [rank_samples[index].hwm_kib for rank_samples in gathered]
        max_rss_mib = max(rss_values) / 1024.0
        sum_rss_mib = sum(rss_values) / 1024.0
        max_hwm_mib = max(hwm_values) / 1024.0
        delta_max_rss_mib = (max(rss_values) - baseline_max_rss) / 1024.0
        print(
            f"{label:<20} {max_rss_mib:12.1f} {sum_rss_mib:12.1f} "
            f"{max_hwm_mib:12.1f} {delta_max_rss_mib:18.1f}"
        )

    print()
    print("Per-rank final RSS/HWM (MiB):")
    for rank, rank_samples in enumerate(gathered):
        last = rank_samples[-1]
        print(
            f"rank {rank:>3}: rss={last.rss_kib / 1024.0:8.1f} "
            f"hwm={last.hwm_kib / 1024.0:8.1f}"
        )

    evolve_indices = [i for i, label in enumerate(labels) if label.startswith("after_evolve_")]
    if len(evolve_indices) > 1:
        print()
        print("Max RSS drift across repeated evolves (MiB):")
        first_max_rss = max(rank_samples[evolve_indices[0]].rss_kib for rank_samples in gathered)
        for evolve_index in evolve_indices:
            label = labels[evolve_index]
            current_max_rss = max(rank_samples[evolve_index].rss_kib for rank_samples in gathered)
            drift_mib = (current_max_rss - first_max_rss) / 1024.0
            print(f"{label:<20} {drift_mib:8.1f}")


def local_qualities(local_i: int, local_i_offset: int) -> np.ndarray:
    local_indices = np.arange(local_i, dtype=np.float64) + float(local_i_offset)
    return np.cos(local_indices)


def build_params(depth: int, gamma: float, beta: float) -> np.ndarray:
    params = np.empty(2 * depth, dtype=np.float64)
    params[0::2] = gamma
    params[1::2] = beta
    return params


ARGS = parse_args()


def main() -> None:
    comm = MPI.COMM_WORLD
    samples: list[MemorySample] = []

    if ARGS.system_size < 1 or (ARGS.system_size & (ARGS.system_size - 1)) != 0:
        raise ValueError("--system-size must be a positive power of two for QAOASparse")

    params = build_params(ARGS.depth, ARGS.gamma, ARGS.beta)

    samples.append(take_sample("baseline", comm))

    alg = QAOASparse(ARGS.system_size, comm)
    alg.set_qualities(local_qualities)
    alg.set_depth(ARGS.depth)
    samples.append(take_sample("configured", comm))

    alg.setup()
    samples.append(take_sample("after_setup", comm))

    alg.prepare()
    samples.append(take_sample("after_prepare", comm))

    for repeat_index in range(ARGS.repeats):
        alg.evolve_state(params)
        samples.append(take_sample(f"after_evolve_{repeat_index + 1:02d}", comm))

    alg.destroy()
    gc.collect()
    samples.append(take_sample("after_destroy", comm))

    print_samples(samples, comm)


if __name__ == "__main__":
    main()