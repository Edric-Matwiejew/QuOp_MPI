#!/usr/bin/env python3
"""Summarize QuOp MPI line-trace heartbeat files by source line."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RankStatus:
    rank: int
    timestamp: str | None
    reason: str | None
    run_id: str | None
    focus: str
    stack: list[str]
    other_threads: list[str]
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group QuOp line-trace rank status files by their current source line."
    )
    parser.add_argument(
        "trace_dir",
        nargs="?",
        help=(
            "Trace directory produced by QUOP_LINE_TRACE. Defaults to QUOP_TRACE_DIR"
            " or the newest local trace directory."
        ),
    )
    parser.add_argument(
        "--show-stack",
        action="store_true",
        help="Show a representative Python stack for each grouped line.",
    )
    parser.add_argument(
        "--history-lines",
        type=int,
        default=0,
        help=(
            "Show the last N history lines for each grouped line"
            " using the first rank in that group."
        ),
    )
    parser.add_argument(
        "--stale-window-seconds",
        type=float,
        default=30.0,
        help=(
            "Ignore rank status files older than this many seconds"
            " relative to the newest status file."
        ),
    )
    return parser.parse_args()


def resolve_trace_dir(raw_path: str | None) -> Path:
    if raw_path:
        trace_dir = Path(raw_path).expanduser()
    elif os.getenv("QUOP_TRACE_DIR"):
        trace_dir = Path(os.environ["QUOP_TRACE_DIR"]).expanduser()
    else:
        candidates = []
        for base in (Path.cwd(), Path("/tmp")):
            candidates.extend(base.glob("quop_line_trace_*"))

        candidates = [path for path in candidates if path.is_dir()]
        if not candidates:
            raise FileNotFoundError(
                "No trace directory supplied and no quop_line_trace_* directory found"
                " in the current directory or /tmp."
            )

        trace_dir = max(candidates, key=lambda path: path.stat().st_mtime)

    trace_dir = trace_dir.resolve()
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")
    return trace_dir


def parse_rank_from_name(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse rank from file name: {path.name}") from exc


def parse_status(path: Path) -> RankStatus:
    timestamp = None
    reason = None
    run_id = None
    focus = None
    stack = []
    other_threads = []
    in_stack = False
    in_other_threads = False

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if in_other_threads:
                if line:
                    other_threads.append(line)
                continue

            if in_stack:
                if line == "other_threads:":
                    in_other_threads = True
                    continue
                if line:
                    stack.append(line)
                continue

            if line == "stack:":
                in_stack = True
                continue

            if line.startswith("timestamp: "):
                timestamp = line.split(": ", 1)[1]
            elif line.startswith("reason: "):
                reason = line.split(": ", 1)[1]
            elif line.startswith("run_id: "):
                run_id = line.split(": ", 1)[1]
            elif line.startswith("focus: "):
                focus = line.split(": ", 1)[1]

    if focus is None:
        raise ValueError(f"Missing focus line in {path}")

    return RankStatus(
        rank=parse_rank_from_name(path),
        timestamp=timestamp,
        reason=reason,
        run_id=run_id,
        focus=focus,
        stack=stack,
        other_threads=other_threads,
        path=path,
    )


def format_rank_ranges(ranks: list[int]) -> str:
    if not ranks:
        return "<none>"

    ranges = []
    start = ranks[0]
    end = ranks[0]

    for rank in ranks[1:]:
        if rank == end + 1:
            end = rank
            continue

        ranges.append((start, end))
        start = end = rank

    ranges.append((start, end))

    formatted = []
    for start, end in ranges:
        if start == end:
            formatted.append(str(start))
        else:
            formatted.append(f"{start}-{end}")
    return ",".join(formatted)


def history_path_for(status: RankStatus) -> Path:
    return status.path.with_suffix(".history")


def tail_lines(path: Path, n_lines: int) -> list[str]:
    if n_lines <= 0:
        return []

    with path.open("r", encoding="utf-8") as handle:
        return list(deque((line.rstrip("\n") for line in handle), maxlen=n_lines))


def stack_variants(statuses: list[RankStatus]) -> list[tuple[tuple[str, ...], list[int]]]:
    variants = defaultdict(list)
    for status in statuses:
        variants[tuple(status.stack)].append(status.rank)
    return sorted(variants.items(), key=lambda item: (-len(item[1]), item[1][0]))


def print_group(
    index: int, statuses: list[RankStatus], show_stack: bool, history_lines: int
) -> None:
    ranks = sorted(status.rank for status in statuses)
    reasons = sorted({status.reason for status in statuses if status.reason})
    timestamps = sorted({status.timestamp for status in statuses if status.timestamp})
    run_ids = sorted({status.run_id for status in statuses if status.run_id})
    representative = min(statuses, key=lambda status: status.rank)
    variants = stack_variants(statuses)

    print(f"[{index}] {representative.focus}")
    print(f"  ranks: {format_rank_ranges(ranks)} ({len(ranks)} total)")
    if reasons:
        print(f"  reasons: {', '.join(reasons)}")
    if run_ids:
        print(f"  run_ids: {', '.join(run_ids)}")
    if timestamps:
        print(f"  timestamps: {timestamps[0]} .. {timestamps[-1]}")
    print(f"  stack variants: {len(variants)}")
    print(f"  sample: {representative.path}")

    if show_stack and representative.stack:
        if len(variants) == 1:
            print("  stack:")
            for line in representative.stack:
                print(f"    {line}")
        else:
            print("  stack variants:")
            for variant_index, (variant_stack, variant_ranks) in enumerate(variants, start=1):
                rank_str = format_rank_ranges(sorted(variant_ranks))
                print(
                    f"    variant {variant_index}: ranks {rank_str}"
                    f" ({len(variant_ranks)} total)"
                )
                for line in variant_stack:
                    print(f"      {line}")

    if show_stack and representative.other_threads:
        print("  other_threads:")
        for line in representative.other_threads:
            print(f"    {line}")

    if history_lines > 0:
        history_path = history_path_for(representative)
        if history_path.exists():
            lines = tail_lines(history_path, history_lines)
            print(
                f"  history tail ({min(history_lines, len(lines))} lines"
                f" from rank {representative.rank}):"
            )
            for line in lines:
                print(f"    {line}")


def main() -> int:
    args = parse_args()

    try:
        trace_dir = resolve_trace_dir(args.trace_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    status_files = sorted(trace_dir.glob("rank_*.status"))
    if not status_files:
        print(f"No rank_*.status files found in {trace_dir}", file=sys.stderr)
        return 1

    newest_mtime = max(path.stat().st_mtime for path in status_files)
    active_status_files = [
        path
        for path in status_files
        if newest_mtime - path.stat().st_mtime <= max(args.stale_window_seconds, 0.0)
    ]
    stale_status_files = [path for path in status_files if path not in active_status_files]

    if not active_status_files:
        print(
            f"All rank_*.status files in {trace_dir} are older than the stale"
            f" window of {args.stale_window_seconds} seconds.",
            file=sys.stderr,
        )
        return 1

    groups = defaultdict(list)
    parse_errors = []

    for path in active_status_files:
        try:
            status = parse_status(path)
        except ValueError as exc:
            parse_errors.append(str(exc))
            continue
        groups[status.focus].append(status)

    ordered_groups = sorted(
        groups.values(),
        key=lambda statuses: (
            -len(statuses),
            min(status.rank for status in statuses),
            statuses[0].focus,
        ),
    )

    print(f"Trace directory: {trace_dir}")
    print(f"Status files: {len(active_status_files)}")
    if stale_status_files:
        print(
            f"Ignored stale status files: {len(stale_status_files)}"
            f" (older than {args.stale_window_seconds:.1f}s from newest file)"
        )
    print(f"Unique focus lines: {len(ordered_groups)}")

    if parse_errors:
        print(f"Parse errors: {len(parse_errors)}")
        for error in parse_errors:
            print(f"  {error}")

    for index, statuses in enumerate(ordered_groups, start=1):
        print()
        print_group(index, statuses, args.show_stack, args.history_lines)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
