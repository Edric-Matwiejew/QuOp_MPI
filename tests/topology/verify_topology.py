#!/usr/bin/env python3
"""Parse QUOP_DUMP_COMM_INFO files and compare against expected topology CSV.

Usage
-----
  # Compare a dump file against the expected CSV for a specific test:
  python verify_topology.py compare expected.csv T01 quop_comm_info_init_*.txt

  # Run consistency checks on any dump (no expected CSV needed):
  python verify_topology.py check quop_comm_info_init_*.txt [--rpg 1]

  # Parse and display a dump file as CSV:
  python verify_topology.py parse quop_comm_info_init_*.txt
"""

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ─── Dump file parser ───────────────────────────────────────────────────────

# Per-rank table columns in dump output order (after whitespace split).
# Indices 0-24 are numeric, 25 is binding mode string, 26+ is hostname.
DUMP_COL_MAP = {
    0: "mpi_rank",
    1: "li",
    2: "li_off",
    3: "alloc",
    4: "d_li",
    5: "d_alc",
    6: "d_off",
    7: "sc_r",
    8: "sc_s",
    9: "nc_r",
    10: "nc_s",
    11: "dc_r",
    12: "dc_s",
    13: "dn_r",
    14: "dn_s",
    15: "assigned_device_id",   # GPU column
    16: "my_gpu_index",         # phys column
    17: "gpu_devcomm_flag",     # gpu? column
    18: "cpu_numa_node",        # cpuNm column
    19: "rank_within_cpu_numa", # rCpuN column
    20: "rank_within_gpu",      # rGpu column
    21: "gpu_slot_ordinal",     # slot column
    22: "is_gpu_rank",          # igpu column
    23: "ranks_per_gpu",        # rpg column
    24: "node_id",              # node column
    25: "worker_id",            # wid column
    26: "n_workers",            # nwk column
}


def parse_dump_file(filepath: str) -> Dict:
    """Parse a comm_info dump file.

    Returns dict with keys ``phase``, ``header``, ``ranks``.
    """
    with open(filepath) as f:
        lines = f.readlines()

    result: Dict = {"phase": "", "header": {}, "ranks": []}

    # Extract phase from title line.
    for line in lines:
        m = re.search(r"dump\s*\((\w+)\)", line)
        if m:
            result["phase"] = m.group(1)
            break

    # Parse header key=value pairs (between first === and first blank line).
    past_title = False
    in_header = False
    for line in lines:
        if "=" * 20 in line:
            if not past_title:
                past_title = True
                in_header = True
                continue
            else:
                break
        if in_header:
            stripped = line.strip()
            if stripped == "":
                in_header = False
                continue
            if "=" in stripped:
                key, _, value = stripped.partition("=")
                result["header"][key.strip()] = value.strip()

    # Parse per-rank data table.
    in_table = False
    skip = 0
    for line in lines:
        if "Per-rank data" in line:
            in_table = True
            skip = 2  # skip header row + dashes
            continue
        if in_table:
            if skip > 0:
                skip -= 1
                continue
            stripped = line.strip()
            if stripped == "" or stripped.startswith("Partition") or stripped.startswith("Per-"):
                break
            parts = stripped.split()
            if len(parts) < 28:
                continue
            try:
                row: Dict = {}
                for idx, name in DUMP_COL_MAP.items():
                    row[name] = int(parts[idx])
                row["binding_mode"] = parts[27].strip()
                row["hostname"] = " ".join(parts[28:]) if len(parts) > 28 else ""
                result["ranks"].append(row)
            except (ValueError, IndexError):
                continue

    return result


# ─── Expected CSV reader ────────────────────────────────────────────────────


def read_expected_csv(path: str) -> List[Dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        return [
            {k: int(v) if v.lstrip("-").isdigit() else v for k, v in row.items()}
            for row in reader
        ]


# ─── Field-by-field comparison ──────────────────────────────────────────────

# Fields to compare exactly between expected CSV and dump.
COMPARE_FIELDS = [
    "node_id", "sc_r", "sc_s", "nc_r", "nc_s",
    "dc_r", "dc_s", "dn_r", "dn_s",
    "is_gpu_rank", "my_gpu_index", "gpu_devcomm_flag",
    "rank_within_gpu", "gpu_slot_ordinal",
    "worker_id", "n_workers",
]

# Layer 2 (data partitioning) fields — compared when the expected CSV
# contains them.  These are exact for diagonal-only ansatz with a
# system_size that divides evenly by both SUBCOMM and DEVCOMM sizes.
LAYER2_FIELDS = [
    "li", "li_off", "alloc",
    "d_li", "d_off", "d_alc",
]

# Map expected-CSV column names to dump column names.
_EXPECTED_TO_DUMP = {
    "local_i": "li",
    "local_i_offset": "li_off",
    "alloc_local": "alloc",
    "device_local_i": "d_li",
    "device_local_i_offset": "d_off",
    "device_alloc_local": "d_alc",
}


def compare_expected_vs_actual(
    expected_rows: List[Dict],
    actual_ranks: List[Dict],
    test_id: str,
) -> List[str]:
    """Compare expected CSV rows against parsed dump data.

    Returns list of human-readable difference strings.
    """
    expected = {
        int(r["mpi_rank"]): r
        for r in expected_rows
        if r["test_id"] == test_id
    }
    actual = {r["mpi_rank"]: r for r in actual_ranks}

    diffs: List[str] = []

    if len(expected) != len(actual):
        diffs.append(
            f"Rank count mismatch: expected {len(expected)}, got {len(actual)}"
        )

    for mpi_rank in sorted(set(expected) & set(actual)):
        exp = expected[mpi_rank]
        act = actual[mpi_rank]
        for field in COMPARE_FIELDS:
            exp_val = int(exp[field])
            act_val = act[field]
            if exp_val != act_val:
                diffs.append(
                    f"Rank {mpi_rank:>3}, {field:<20s}: "
                    f"expected {exp_val:>4}, got {act_val:>4}"
                )

        # Layer 2 fields (present when the expected CSV includes them).
        for csv_name, dump_name in _EXPECTED_TO_DUMP.items():
            if csv_name not in exp:
                continue
            exp_val = int(exp[csv_name])
            act_val = act[dump_name]
            if exp_val != act_val:
                diffs.append(
                    f"Rank {mpi_rank:>3}, {dump_name:<20s}: "
                    f"expected {exp_val:>4}, got {act_val:>4}"
                )

    return diffs


# ─── Consistency checks (binding-mode agnostic) ────────────────────────────


def check_topology_consistency(
    actual_ranks: List[Dict],
    rpg: int = 1,
) -> List[str]:
    """Verify topology invariants on any dump, regardless of binding mode.

    Parameters
    ----------
    actual_ranks : parsed per-rank rows from a dump file
    rpg : expected QUOP_RANKS_PER_GPU (default 1)

    Returns
    -------
    list of issue strings (empty means all checks passed)
    """
    issues: List[str] = []
    if not actual_ranks:
        issues.append("No rank data found in dump")
        return issues

    # Detect worker count and group ranks by worker_id.
    n_workers = actual_ranks[0].get("n_workers", 1)
    worker_groups: Dict[int, List[Dict]] = {}
    for r in actual_ranks:
        wid = r.get("worker_id", 0)
        worker_groups.setdefault(wid, []).append(r)

    gpu_ranks = [r for r in actual_ranks if r["is_gpu_rank"] == 1]
    non_gpu_ranks = [r for r in actual_ranks if r["is_gpu_rank"] == 0]
    nodes = sorted(set(r["node_id"] for r in actual_ranks))

    # ── Global checks (GPU topology assigned before worker split) ────

    # 1. GPU / DEVCOMM membership consistency.
    for r in gpu_ranks:
        if r["gpu_devcomm_flag"] != 1:
            issues.append(
                f"Rank {r['mpi_rank']}: is_gpu_rank=1 but gpu_devcomm_flag=0"
            )
    for r in non_gpu_ranks:
        if r["gpu_devcomm_flag"] != 0:
            issues.append(
                f"Rank {r['mpi_rank']}: is_gpu_rank=0 but gpu_devcomm_flag=1"
            )

    # 5. Non-GPU ranks excluded from device communicators.
    for r in non_gpu_ranks:
        if r["dc_r"] != -1 or r["dc_s"] != 0:
            issues.append(
                f"Rank {r['mpi_rank']}: non-GPU but dc_r={r['dc_r']}, dc_s={r['dc_s']}"
            )
        if r["dn_r"] != -1 or r["dn_s"] != 0:
            issues.append(
                f"Rank {r['mpi_rank']}: non-GPU but dn_r={r['dn_r']}, dn_s={r['dn_s']}"
            )

    # 6. rank_within_gpu in [0, rpg-1] for GPU ranks.
    for r in gpu_ranks:
        if not (0 <= r["rank_within_gpu"] < rpg):
            issues.append(
                f"Rank {r['mpi_rank']}: rank_within_gpu={r['rank_within_gpu']} "
                f"outside [0, {rpg-1}]"
            )

    # 7. gpu_slot_ordinal dense [0, n) per node.
    for nid in nodes:
        node_gpu = [r for r in gpu_ranks if r["node_id"] == nid]
        if not node_gpu:
            continue
        expected_slots = set(range(len(node_gpu)))
        actual_slots = set(r["gpu_slot_ordinal"] for r in node_gpu)
        if actual_slots != expected_slots:
            issues.append(
                f"Node {nid}: gpu_slot_ordinals {sorted(actual_slots)} "
                f"!= expected {sorted(expected_slots)}"
            )

    # 8. Each physical GPU has at most rpg GPU ranks per node.
    for nid in nodes:
        node_gpu = [r for r in gpu_ranks if r["node_id"] == nid]
        gpu_counts: Dict[int, int] = {}
        for r in node_gpu:
            g = r["my_gpu_index"]
            gpu_counts[g] = gpu_counts.get(g, 0) + 1
        for g, count in sorted(gpu_counts.items()):
            if count > rpg:
                issues.append(
                    f"Node {nid}, GPU {g}: {count} ranks > rpg={rpg}"
                )

    # 12. Non-GPU gpu_slot_ordinal should be -1.
    for r in non_gpu_ranks:
        if r["gpu_slot_ordinal"] != -1:
            issues.append(
                f"Rank {r['mpi_rank']}: non-GPU but gpu_slot_ordinal="
                f"{r['gpu_slot_ordinal']}"
            )

    # 13. rank_within_cpu_numa >= 0 when cpu_numa_node >= 0.
    for r in actual_ranks:
        if r["cpu_numa_node"] >= 0 and r["rank_within_cpu_numa"] < 0:
            issues.append(
                f"Rank {r['mpi_rank']}: cpu_numa_node={r['cpu_numa_node']} "
                f"but rank_within_cpu_numa={r['rank_within_cpu_numa']}"
            )

    # 15. n_workers consistent across all ranks.
    nwk_values = set(r["n_workers"] for r in actual_ranks)
    if len(nwk_values) > 1:
        issues.append(f"Inconsistent n_workers across ranks: {nwk_values}")

    # 16. worker_id in [0, n_workers-1].
    if nwk_values:
        nwk = list(nwk_values)[0]
        wids = set(r["worker_id"] for r in actual_ranks)
        for wid in wids:
            if not (0 <= wid < nwk):
                issues.append(
                    f"worker_id={wid} outside [0, {nwk - 1}]"
                )

    # ── Per-worker checks (SUBCOMM/NODECOMM/DEVCOMM are per-worker) ──

    for wid, group in sorted(worker_groups.items()):
        pfx = f"Worker {wid}: " if n_workers > 1 else ""
        w_gpu = [r for r in group if r["is_gpu_rank"] == 1]
        w_nodes = sorted(set(r["node_id"] for r in group))

        # 2. DEVCOMM size consistent across GPU ranks in this worker.
        if w_gpu:
            dc_sizes = set(r["dc_s"] for r in w_gpu)
            if len(dc_sizes) > 1:
                issues.append(f"{pfx}Inconsistent DEVCOMM sizes: {dc_sizes}")
            elif list(dc_sizes)[0] != len(w_gpu):
                issues.append(
                    f"{pfx}DEVCOMM size {list(dc_sizes)[0]} "
                    f"!= GPU rank count {len(w_gpu)}"
                )

        # 3. DEVCOMM ranks dense [0, n_gpu-1].
        if w_gpu:
            dc_ranks = sorted(r["dc_r"] for r in w_gpu)
            if dc_ranks != list(range(len(w_gpu))):
                issues.append(
                    f"{pfx}DEVCOMM ranks not contiguous 0..{len(w_gpu)-1}"
                )

        # 4. Per-node DEVCOMM_NODE consistency.
        for nid in w_nodes:
            node_gpu = [r for r in w_gpu if r["node_id"] == nid]
            if not node_gpu:
                continue
            dn_sizes = set(r["dn_s"] for r in node_gpu)
            if len(dn_sizes) > 1:
                issues.append(
                    f"{pfx}Node {nid}: inconsistent DEVCOMM_NODE sizes: {dn_sizes}"
                )
            elif list(dn_sizes)[0] != len(node_gpu):
                issues.append(
                    f"{pfx}Node {nid}: DEVCOMM_NODE size {list(dn_sizes)[0]} "
                    f"!= GPU rank count {len(node_gpu)}"
                )
            dn_ranks = sorted(r["dn_r"] for r in node_gpu)
            if dn_ranks != list(range(len(node_gpu))):
                issues.append(f"{pfx}Node {nid}: DEVCOMM_NODE ranks not contiguous")

        # 9. NODECOMM sizes consistent per node within this worker.
        for nid in w_nodes:
            node_ranks = [r for r in group if r["node_id"] == nid]
            nc_sizes = set(r["nc_s"] for r in node_ranks)
            if len(nc_sizes) > 1:
                issues.append(
                    f"{pfx}Node {nid}: inconsistent NODECOMM sizes: {nc_sizes}"
                )
            elif list(nc_sizes)[0] != len(node_ranks):
                issues.append(
                    f"{pfx}Node {nid}: NODECOMM size {list(nc_sizes)[0]} "
                    f"!= rank count {len(node_ranks)}"
                )

        # 10. SUBCOMM size = number of ranks in this worker.
        sc_sizes = set(r["sc_s"] for r in group)
        if len(sc_sizes) > 1:
            issues.append(f"{pfx}Inconsistent SUBCOMM sizes: {sc_sizes}")
        elif list(sc_sizes)[0] != len(group):
            issues.append(
                f"{pfx}SUBCOMM size {list(sc_sizes)[0]} "
                f"!= worker rank count {len(group)}"
            )

        # 11. SUBCOMM ranks dense [0, n-1].
        sc_ranks = sorted(r["sc_r"] for r in group)
        if sc_ranks != list(range(len(group))):
            issues.append(f"{pfx}SUBCOMM ranks not contiguous")

        # 14. DEVCOMM ranks ordered by SUBCOMM rank.
        if w_gpu:
            gpu_by_dc = sorted(w_gpu, key=lambda r: r["dc_r"])
            sc_order = [r["sc_r"] for r in gpu_by_dc]
            if sc_order != sorted(sc_order):
                issues.append(
                    f"{pfx}DEVCOMM rank order does not match SUBCOMM rank order"
                )

        # 17. SUBCOMM sizes consistent within worker (already implied by 10,
        #     but kept for explicit per-worker validation).
        sc_sizes_w = set(r["sc_s"] for r in group)
        if len(sc_sizes_w) > 1:
            issues.append(
                f"{pfx}inconsistent SUBCOMM sizes: {sc_sizes_w}"
            )

    return issues


# ─── Pretty-print dump as CSV ───────────────────────────────────────────────


def dump_to_csv(ranks: List[Dict], out=sys.stdout) -> None:
    if not ranks:
        return
    fields = list(DUMP_COL_MAP.values()) + ["binding_mode", "hostname"]
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for r in ranks:
        writer.writerow({f: r.get(f, "") for f in fields})


# ─── CLI ─────────────────────────────────────────────────────────────────────


USAGE = """\
Usage:
  {prog} compare <expected.csv> <test_id> <dump_file>
      Compare a dump against expected values for the given test ID.

  {prog} check <dump_file> [--rpg N]
      Run consistency checks on any dump (no expected CSV needed).

  {prog} parse <dump_file>
      Print parsed per-rank data as CSV to stdout.
"""


def cmd_compare(args: List[str]) -> int:
    if len(args) < 3:
        print("Usage: compare <expected.csv> <test_id> <dump_file>", file=sys.stderr)
        return 1
    expected_path, test_id, dump_path = args[0], args[1], args[2]

    expected_rows = read_expected_csv(expected_path)
    dump = parse_dump_file(dump_path)

    if not dump["ranks"]:
        print(f"ERROR: no per-rank data found in {dump_path}", file=sys.stderr)
        return 1

    phase = dump["phase"]
    print(f"Dump phase: {phase}")
    print(f"Dump ranks: {len(dump['ranks'])}")
    print(f"Test ID:    {test_id}\n")

    # Field-by-field comparison.
    diffs = compare_expected_vs_actual(expected_rows, dump["ranks"], test_id)

    # Consistency checks.
    test_rows = [r for r in expected_rows if r["test_id"] == test_id]
    rpg = int(test_rows[0]["ranks_per_gpu"]) if test_rows else 1
    issues = check_topology_consistency(dump["ranks"], rpg=rpg)

    ok = True
    if diffs:
        ok = False
        print("=== Field mismatches ===")
        for d in diffs:
            print(f"  FAIL  {d}")
    else:
        print("=== All predicted fields match ===")

    if issues:
        ok = False
        print("\n=== Consistency issues ===")
        for iss in issues:
            print(f"  WARN  {iss}")
    else:
        print("=== All consistency checks passed ===")

    return 0 if ok else 1


def cmd_check(args: List[str]) -> int:
    if not args:
        print("Usage: check <dump_file> [--rpg N]", file=sys.stderr)
        return 1
    dump_path = args[0]
    rpg = 1
    if "--rpg" in args:
        idx = args.index("--rpg")
        if idx + 1 < len(args):
            rpg = int(args[idx + 1])

    dump = parse_dump_file(dump_path)
    if not dump["ranks"]:
        print(f"ERROR: no per-rank data found in {dump_path}", file=sys.stderr)
        return 1

    print(f"Dump phase: {dump['phase']}")
    print(f"Dump ranks: {len(dump['ranks'])}")
    print(f"RPG:        {rpg}\n")

    issues = check_topology_consistency(dump["ranks"], rpg=rpg)
    if issues:
        print("=== Consistency issues ===")
        for iss in issues:
            print(f"  WARN  {iss}")
        return 1
    else:
        print("=== All consistency checks passed ===")
        return 0


def cmd_parse(args: List[str]) -> int:
    if not args:
        print("Usage: parse <dump_file>", file=sys.stderr)
        return 1
    dump = parse_dump_file(args[0])
    if not dump["ranks"]:
        print(f"ERROR: no per-rank data found in {args[0]}", file=sys.stderr)
        return 1
    dump_to_csv(dump["ranks"])
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(USAGE.format(prog=sys.argv[0]))
        return 1

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    dispatch = {
        "compare": cmd_compare,
        "check": cmd_check,
        "parse": cmd_parse,
    }
    if cmd not in dispatch:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print(USAGE.format(prog=sys.argv[0]))
        return 1

    return dispatch[cmd](rest)


if __name__ == "__main__":
    sys.exit(main())
