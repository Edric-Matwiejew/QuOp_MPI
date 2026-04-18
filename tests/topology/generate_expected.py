#!/usr/bin/env python3
"""Generate expected comm_info topology CSV for Pawsey Setonix verification.

Replicates the GPU assignment logic in gpu_topology.f90
(``apply_sequential_assignment``) and the communicator creation in
communicators.f90 (``create_devcomm_with_topology``) to produce exact
per-rank predictions for every field in the QUOP_DUMP_COMM_INFO output.

Usage
-----
  python generate_expected.py expected_topology.csv   # write CSV
  python generate_expected.py --list                   # show config matrix
  python generate_expected.py --jobs                   # show srun parameters
"""

import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ─── Setonix hardware constants ─────────────────────────────────────────────

SETONIX_GPU = {
    "n_physical_gpus": 8,           # 4 × MI250X = 8 GCDs
    "visible_device_count": 8,      # all GCDs visible (no --gpu-bind)
    "backend_flag": 1,              # wavefront backend
}

SETONIX_WORK = {
    "n_physical_gpus": 0,
    "visible_device_count": 0,
    "backend_flag": 0,              # MPI backend
}

# ─── Test configuration ─────────────────────────────────────────────────────


@dataclass
class TestConfig:
    test_id: str
    partition: str          # "gpu" or "work"
    n_nodes: int
    ranks_per_node: int
    binding_mode: str       # "sequential", "auto", "numa"
    ranks_per_gpu: int
    system_size: int        # state-vector dimension (2^n_qubits)
    description: str
    n_workers: int = 1      # parallel jacobian workers


#    ID   Part  Nodes  RPN  Mode          RPG  SysSize  Description
#
# system_size is chosen as total_ranks * 16 so that it divides evenly by
# both sc_s (SUBCOMM size) and dc_s (DEVCOMM size).  This makes the Layer 2
# predictions (local_i, device_local_i, ...) exact and NUMA-independent.
CONFIGS: List[TestConfig] = [
    # ── Sequential binding on GPU partition ──────────────────────────────
    TestConfig("T01", "gpu", 1,  8,  "sequential", 1,  128,
               "saturated: 8 ranks = 8 GPUs × rpg=1"),
    TestConfig("T02", "gpu", 1,  4,  "sequential", 1,   64,
               "under-sat: 4 ranks on 8 GPUs"),
    TestConfig("T03", "gpu", 1, 16,  "sequential", 1,  256,
               "over-sat: 8 GPU + 8 non-GPU"),
    TestConfig("T04", "gpu", 1, 16,  "sequential", 2,  256,
               "saturated: 16 ranks = 8 GPUs × rpg=2"),
    TestConfig("T05", "gpu", 1, 64,  "sequential", 8, 1024,
               "full: 64 ranks = 8 GPUs × rpg=8"),
    TestConfig("T06", "gpu", 1, 64,  "sequential", 1, 1024,
               "extreme over: 8 GPU + 56 non-GPU"),
    TestConfig("T07", "gpu", 1,  1,  "sequential", 1,   16,
               "minimal: 1 rank on GPU 0"),
    TestConfig("T08", "gpu", 2,  8,  "sequential", 1,  256,
               "2-node saturated"),
    TestConfig("T09", "gpu", 2, 16,  "sequential", 2,  512,
               "2-node × rpg=2 saturated"),
    TestConfig("T10", "gpu", 2,  4,  "sequential", 1,  128,
               "2-node under-sat"),
    TestConfig("T11", "gpu", 1, 32,  "sequential", 4,  512,
               "32 ranks = 8 GPUs × rpg=4"),
    TestConfig("T12", "gpu", 1,  2,  "sequential", 1,   32,
               "2 ranks, GPUs 0-1 only"),

    # ── Work partition (MPI backend, no GPUs) ────────────────────────────
    TestConfig("T13", "work", 1,   4, "auto", 1,   64, "basic CPU-only"),
    TestConfig("T14", "work", 2,   4, "auto", 1,  128, "2-node CPU-only"),
    TestConfig("T15", "work", 1, 128, "auto", 1, 2048, "full CPU node"),
]

# ─── CSV field order ────────────────────────────────────────────────────────

FIELD_NAMES = [
    "test_id", "partition", "n_nodes", "ranks_per_node", "binding_mode",
    "ranks_per_gpu", "system_size", "mpi_rank", "node_id",
    "sc_r", "sc_s", "nc_r", "nc_s",
    "dc_r", "dc_s", "dn_r", "dn_s", "is_gpu_rank", "my_gpu_index",
    "gpu_devcomm_flag", "rank_within_gpu", "gpu_slot_ordinal",
    "n_physical_gpus", "visible_device_count", "backend_flag",
    "worker_id", "n_workers",
    "local_i", "local_i_offset", "alloc_local",
    "device_local_i", "device_local_i_offset", "device_alloc_local",
    "n_processes",
]

# ─── Oracle: worker split assignment ────────────────────────────────────────


def gpu_aware_worker_split(
    n_workers: int,
    n_nodes: int,
    ranks_per_node: int,
    is_gpu_ranks: List[bool],
    gpu_slot_ordinals: List[int],
    node_ids: List[int],
) -> List[int]:
    """Replicate the GPU-aware intra-node split from split_workers.

    Parameters
    ----------
    n_workers : int
        Number of worker subcommunicators.
    n_nodes : int
        Number of compute nodes.
    ranks_per_node : int
        Ranks per node.
    is_gpu_ranks : list[bool]
        Per-MPI-rank GPU flag.
    gpu_slot_ordinals : list[int]
        Per-MPI-rank gpu_slot_ordinal.
    node_ids : list[int]
        Per-MPI-rank node id.

    Returns
    -------
    list[int]
        worker_id for each MPI rank.
    """
    total_ranks = len(is_gpu_ranks)

    if n_workers <= n_nodes:
        # Node-aligned split
        nodes_per_worker = n_nodes // n_workers
        node_remainder = n_nodes % n_workers
        worker_ids = []
        for mpi_rank in range(total_ranks):
            nid = node_ids[mpi_rank]
            if nid < node_remainder * (nodes_per_worker + 1):
                color = nid // (nodes_per_worker + 1)
            else:
                color = node_remainder + (
                    nid - node_remainder * (nodes_per_worker + 1)
                ) // nodes_per_worker
            worker_ids.append(color)
        return worker_ids

    # GPU-aware intra-node split (n_workers > n_nodes)
    # Count active GPU ranks per node (devcomm_node_size)
    node_active_gpu = [0] * n_nodes
    for mpi_rank in range(total_ranks):
        if is_gpu_ranks[mpi_rank]:
            node_active_gpu[node_ids[mpi_rank]] += 1

    # Distribute workers across nodes
    workers_per_node = [0] * n_nodes
    remaining = n_workers
    for nid in range(n_nodes):
        if node_active_gpu[nid] > 0 and remaining > 0:
            workers_per_node[nid] = 1
            remaining -= 1

    while remaining > 0:
        progress = False
        for nid in range(n_nodes):
            if workers_per_node[nid] < node_active_gpu[nid]:
                workers_per_node[nid] += 1
                remaining -= 1
                progress = True
                if remaining == 0:
                    break
        if not progress:
            break

    # Assign each rank to a worker
    worker_ids = []
    # Track node_rank for non-GPU round-robin
    node_rank_counters = [0] * n_nodes
    for mpi_rank in range(total_ranks):
        nid = node_ids[mpi_rank]
        nr = node_rank_counters[nid]
        node_rank_counters[nid] += 1

        w0 = sum(workers_per_node[:nid])
        nw = workers_per_node[nid]
        active = node_active_gpu[nid]

        if is_gpu_ranks[mpi_rank]:
            ordinal = gpu_slot_ordinals[mpi_rank]
            color = w0 + ordinal * nw // active
        else:
            color = w0 + (nr % nw)
        worker_ids.append(color)

    return worker_ids


def mpi_backend_worker_split(
    n_workers: int, total_ranks: int
) -> List[int]:
    """Replicate rank-based round-robin split for MPI backend."""
    if n_workers <= 1:
        return [0] * total_ranks

    rpw = total_ranks // n_workers
    remainder = total_ranks % n_workers
    worker_ids = []
    for rank in range(total_ranks):
        if rank < remainder * (rpw + 1):
            color = rank // (rpw + 1)
        else:
            color = remainder + (rank - remainder * (rpw + 1)) // rpw
        worker_ids.append(color)
    return worker_ids


# ─── Oracle: sequential GPU assignment ──────────────────────────────────────


def sequential_assignment(
    n_ranks: int, n_gpus: int, rpg: int
) -> List[Tuple[int, bool]]:
    """Replicate ``apply_sequential_assignment`` from gpu_topology.f90.

    Assumes all GPUs visible to every rank (Setonix without --gpu-bind).

    Returns
    -------
    list of (my_gpu_index, is_gpu_rank) per node_rank (0-based)
    """
    gpu_loads = [0] * n_gpus
    result: List[Tuple[int, bool]] = []

    for r in range(n_ranks):
        target_gpu = (r // rpg) % n_gpus

        # find_visible_gpu_with_capacity: circular scan from target
        assigned = -1
        for offset in range(n_gpus):
            g = (target_gpu + offset) % n_gpus
            if gpu_loads[g] < rpg:
                assigned = g
                gpu_loads[g] += 1
                result.append((g, True))
                break

        if assigned < 0:
            # Capacity exhausted — find_visible_gpu_from_target returns
            # first visible from target (all visible on Setonix → target).
            result.append((target_gpu, False))

    return result


# ─── Derived topology fields ────────────────────────────────────────────────


def compute_rank_within_gpu(assigned_indices: List[int], node_rank: int) -> int:
    """Count earlier node-ranks assigned to the same physical GPU."""
    my_gpu = assigned_indices[node_rank]
    if my_gpu < 0:
        return 0
    return sum(1 for i in range(node_rank) if assigned_indices[i] == my_gpu)


def compute_gpu_slot_ordinal(
    assigned_indices: List[int],
    is_gpu_ranks: List[bool],
    node_rank: int,
    n_gpus: int,
) -> int:
    """Dense ordinal among active GPU ranks on this node (-1 if non-GPU)."""
    if not is_gpu_ranks[node_rank]:
        return -1

    my_gpu = assigned_indices[node_rank]

    # Count active GPU ranks per physical GPU.
    active = [0] * n_gpus
    for g, is_gpu in zip(assigned_indices, is_gpu_ranks):
        if is_gpu and g >= 0:
            active[g] += 1

    slot = compute_rank_within_gpu(assigned_indices, node_rank)
    for g in range(my_gpu):
        slot += active[g]
    return slot


# ─── Expected per-rank rows ─────────────────────────────────────────────────


def compute_expected(cfg: TestConfig) -> List[Dict]:
    """Return expected per-rank topology rows for *cfg*."""
    hw = SETONIX_GPU if cfg.partition == "gpu" else SETONIX_WORK
    n_gpus = hw["n_physical_gpus"]
    total_ranks = cfg.n_nodes * cfg.ranks_per_node

    # Per-node GPU assignment (same for every node on Setonix).
    if cfg.partition == "gpu" and n_gpus > 0:
        node_assignment = sequential_assignment(
            cfg.ranks_per_node, n_gpus, cfg.ranks_per_gpu
        )
    else:
        node_assignment = [(-1, False)] * cfg.ranks_per_node

    assigned_indices = [a[0] for a in node_assignment]
    is_gpu_ranks = [a[1] for a in node_assignment]
    gpu_ranks_per_node = sum(is_gpu_ranks)
    total_gpu_ranks = gpu_ranks_per_node * cfg.n_nodes

    # Global GPU-rank list ordered by SUBCOMM (= MPI) rank.
    global_gpu_mpi_ranks: List[int] = []
    for node in range(cfg.n_nodes):
        for nr in range(cfg.ranks_per_node):
            if is_gpu_ranks[nr]:
                global_gpu_mpi_ranks.append(node * cfg.ranks_per_node + nr)

    # Per-node GPU node_ranks (for DEVCOMM_NODE ordering).
    node_gpu_nrs = [nr for nr in range(cfg.ranks_per_node) if is_gpu_ranks[nr]]

    # ── Layer 2: data partitioning for diagonal-only (no FFT constraints) ──
    # system_size is chosen as a multiple of total_ranks (and hence dc_s),
    # so block_distribute has zero remainder and all predictions are exact
    # regardless of NUMA-biased host assignment ordering.

    rows: List[Dict] = []
    for node in range(cfg.n_nodes):
        for nr in range(cfg.ranks_per_node):
            mpi_rank = node * cfg.ranks_per_node + nr
            is_gpu = is_gpu_ranks[nr]
            my_gpu = assigned_indices[nr]
            rwg = compute_rank_within_gpu(assigned_indices, nr)
            slot = compute_gpu_slot_ordinal(
                assigned_indices, is_gpu_ranks, nr, max(n_gpus, 1)
            )

            if is_gpu:
                dc_r = global_gpu_mpi_ranks.index(mpi_rank)
                dc_s = total_gpu_ranks
                dn_r = node_gpu_nrs.index(nr)
                dn_s = gpu_ranks_per_node
            else:
                dc_r, dc_s = -1, 0
                dn_r, dn_s = -1, 0

            # Layer 2 fields — diagonal propagator, no constraints.
            if cfg.partition == "gpu" and total_gpu_ranks > 0:
                # Wavefront backend: device block_distribute over DEVCOMM,
                # then host derived via NODECOMM (even split, no NUMA effect).
                if is_gpu:
                    dev_li = cfg.system_size // total_gpu_ranks
                    dev_li_off = dc_r * dev_li
                else:
                    dev_li = 0
                    dev_li_off = 0
                # Host: system_size / total_ranks per rank (even split).
                li = cfg.system_size // total_ranks
                li_off = mpi_rank * li
            else:
                # MPI backend: block_distribute over SUBCOMM.
                li = cfg.system_size // total_ranks
                li_off = mpi_rank * li
                dev_li = 0
                dev_li_off = 0

            rows.append(
                {
                    "test_id": cfg.test_id,
                    "partition": cfg.partition,
                    "n_nodes": cfg.n_nodes,
                    "ranks_per_node": cfg.ranks_per_node,
                    "binding_mode": cfg.binding_mode,
                    "ranks_per_gpu": cfg.ranks_per_gpu,
                    "system_size": cfg.system_size,
                    "mpi_rank": mpi_rank,
                    "node_id": node,
                    "sc_r": mpi_rank,
                    "sc_s": total_ranks,
                    "nc_r": nr,
                    "nc_s": cfg.ranks_per_node,
                    "dc_r": dc_r,
                    "dc_s": dc_s,
                    "dn_r": dn_r,
                    "dn_s": dn_s,
                    "is_gpu_rank": 1 if is_gpu else 0,
                    "my_gpu_index": my_gpu,
                    "gpu_devcomm_flag": 1 if is_gpu else 0,
                    "rank_within_gpu": rwg,
                    "gpu_slot_ordinal": slot,
                    "n_physical_gpus": n_gpus,
                    "visible_device_count": hw["visible_device_count"],
                    "backend_flag": hw["backend_flag"],
                    "worker_id": 0,
                    "n_workers": 1,
                    "local_i": li,
                    "local_i_offset": li_off,
                    "alloc_local": li,
                    "device_local_i": dev_li,
                    "device_local_i_offset": dev_li_off,
                    "device_alloc_local": dev_li,
                    "n_processes": total_ranks,
                }
            )

    return rows


# ─── CSV output ─────────────────────────────────────────────────────────────


def write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        writer.writeheader()
        writer.writerows(rows)


# ─── CLI modes ──────────────────────────────────────────────────────────────


def print_list() -> None:
    hdr = (f"{'ID':<5} {'Part':<5} {'Nodes':>5} {'RPN':>5} {'Mode':<12} "
           f"{'RPG':>4} {'SysSize':>7}  Description")
    print(hdr)
    print("-" * len(hdr))
    for c in CONFIGS:
        print(
            f"{c.test_id:<5} {c.partition:<5} {c.n_nodes:>5} "
            f"{c.ranks_per_node:>5} {c.binding_mode:<12} "
            f"{c.ranks_per_gpu:>4} {c.system_size:>7}  {c.description}"
        )


def print_jobs() -> None:
    """Print srun parameters and env vars for each test configuration."""
    print("# Setonix topology test job parameters")
    print("# Set QUOP_DUMP_COMM_INFO=1 for all jobs.")
    print("# Run any QuOp program that triggers ansatz.execute().\n")
    for c in CONFIGS:
        total = c.n_nodes * c.ranks_per_node
        part = "gpu" if c.partition == "gpu" else "work"
        print(f"# --- {c.test_id}: {c.description} ---")
        print(f"#SBATCH --partition={part}")
        print(f"#SBATCH --nodes={c.n_nodes}")
        print(f"#SBATCH --ntasks={total}")
        print(f"#SBATCH --ntasks-per-node={c.ranks_per_node}")
        if c.partition == "gpu":
            n_gpus = min(c.ranks_per_node, 8)
            print(f"#SBATCH --gres=gpu:{n_gpus}")
        print(f"export QUOP_DUMP_COMM_INFO=1")
        if c.partition == "gpu":
            print(f"export QUOP_GPU_BINDING_MODE={c.binding_mode}")
            print(f"export QUOP_RANKS_PER_GPU={c.ranks_per_gpu}")
            print(f"export MPICH_GPU_SUPPORT_ENABLED=1")
        print(f"srun -N{c.n_nodes} -n{total} python run_diagonal_topology.py {c.system_size}")
        print()


# ─── Algorithm-class integration tests (consistency checks only) ────────────

@dataclass
class AlgorithmTestConfig:
    test_id: str
    algorithm: str          # "qaoa", "qwoa", "qmoa"
    partition: str
    n_nodes: int
    ranks_per_node: int
    binding_mode: str
    ranks_per_gpu: int
    algo_args: str          # CLI args for run_algorithm_topology.py
    description: str


# These configs exercise full negotiate paths with FFT-dependent propagators.
# Verification uses consistency checks only (no exact predictions).
#
# QAOA: transverse_field mixer requires power-of-2 system_size.
# QWOA: circulant mixer uses FFTW (MPI) or SHAFFT (wavefront).
# QMOA: composite mixer uses FFTW (MPI) or SHAFFT (wavefront).
ALGO_CONFIGS: List[AlgorithmTestConfig] = [
    # ── QAOA on GPU partition ────────────────────────────────────────────
    AlgorithmTestConfig("A01", "qaoa", "gpu", 1,  8, "sequential", 1,
                        "qaoa 128", "QAOA saturated 8 GPUs"),
    AlgorithmTestConfig("A02", "qaoa", "gpu", 1, 16, "sequential", 1,
                        "qaoa 256", "QAOA over-sat 8 GPU + 8 non-GPU"),
    AlgorithmTestConfig("A03", "qaoa", "gpu", 2,  8, "sequential", 1,
                        "qaoa 256", "QAOA 2-node saturated"),
    AlgorithmTestConfig("A04", "qaoa", "work", 1,  4, "auto", 1,
                        "qaoa 64", "QAOA CPU-only"),
    # ── QWOA on GPU partition ────────────────────────────────────────────
    AlgorithmTestConfig("A05", "qwoa", "gpu", 1,  8, "sequential", 1,
                        "qwoa 128", "QWOA saturated 8 GPUs"),
    AlgorithmTestConfig("A06", "qwoa", "gpu", 1, 16, "sequential", 1,
                        "qwoa 256", "QWOA over-sat 8 GPU + 8 non-GPU"),
    AlgorithmTestConfig("A07", "qwoa", "gpu", 2,  8, "sequential", 1,
                        "qwoa 256", "QWOA 2-node saturated"),
    AlgorithmTestConfig("A08", "qwoa", "work", 1,  4, "auto", 1,
                        "qwoa 64", "QWOA CPU-only"),
    # ── QMOA on GPU partition ────────────────────────────────────────────
    AlgorithmTestConfig("A09", "qmoa", "gpu", 1,  8, "sequential", 1,
                        "qmoa 4 4", "QMOA 2D (2^4×2^4=256) saturated"),
    AlgorithmTestConfig("A10", "qmoa", "gpu", 1, 16, "sequential", 1,
                        "qmoa 4 4", "QMOA 2D over-sat"),
    AlgorithmTestConfig("A11", "qmoa", "gpu", 2,  8, "sequential", 1,
                        "qmoa 4 4", "QMOA 2D 2-node saturated"),
    AlgorithmTestConfig("A12", "qmoa", "work", 1,  4, "auto", 1,
                        "qmoa 3 3", "QMOA 2D (2^3×2^3=64) CPU-only"),
]


def print_algo_list() -> None:
    hdr = (f"{'ID':<5} {'Algo':<5} {'Part':<5} {'Nodes':>5} {'RPN':>5} "
           f"{'Mode':<12} {'RPG':>4}  Description")
    print(hdr)
    print("-" * len(hdr))
    for c in ALGO_CONFIGS:
        print(
            f"{c.test_id:<5} {c.algorithm:<5} {c.partition:<5} {c.n_nodes:>5} "
            f"{c.ranks_per_node:>5} {c.binding_mode:<12} "
            f"{c.ranks_per_gpu:>4}  {c.description}"
        )


def print_algo_jobs() -> None:
    """Print srun parameters for algorithm integration tests."""
    print("# Algorithm-class topology integration tests")
    print("# Set QUOP_DUMP_COMM_INFO=1 for all jobs.")
    print("# Verification: verify_topology.py check <dump_file> --rpg N\n")
    for c in ALGO_CONFIGS:
        total = c.n_nodes * c.ranks_per_node
        part = "gpu" if c.partition == "gpu" else "work"
        print(f"# --- {c.test_id}: {c.description} ---")
        print(f"#SBATCH --partition={part}")
        print(f"#SBATCH --nodes={c.n_nodes}")
        print(f"#SBATCH --ntasks={total}")
        print(f"#SBATCH --ntasks-per-node={c.ranks_per_node}")
        if c.partition == "gpu":
            n_gpus = min(c.ranks_per_node, 8)
            print(f"#SBATCH --gres=gpu:{n_gpus}")
        print(f"export QUOP_DUMP_COMM_INFO=1")
        if c.partition == "gpu":
            print(f"export QUOP_GPU_BINDING_MODE={c.binding_mode}")
            print(f"export QUOP_RANKS_PER_GPU={c.ranks_per_gpu}")
            print(f"export MPICH_GPU_SUPPORT_ENABLED=1")
        print(f"srun -N{c.n_nodes} -n{total} python run_algorithm_topology.py {c.algo_args}")
        print()


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output.csv>")
        print(f"       {sys.argv[0]} --list")
        print(f"       {sys.argv[0]} --jobs")
        print(f"       {sys.argv[0]} --algo-list")
        print(f"       {sys.argv[0]} --algo-jobs")
        sys.exit(1)

    if sys.argv[1] == "--list":
        print_list()
        return

    if sys.argv[1] == "--jobs":
        print_jobs()
        return

    if sys.argv[1] == "--algo-list":
        print_algo_list()
        return

    if sys.argv[1] == "--algo-jobs":
        print_algo_jobs()
        return

    output_path = sys.argv[1]
    all_rows: List[Dict] = []
    for cfg in CONFIGS:
        rows = compute_expected(cfg)
        all_rows.extend(rows)
        print(f"  {cfg.test_id}: {len(rows):>3} ranks – {cfg.description}")

    write_csv(all_rows, output_path)
    print(f"\nWrote {len(all_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
