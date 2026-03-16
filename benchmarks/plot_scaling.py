#!/usr/bin/env python3
"""Plot strong- and weak-scaling metrics derived from mean_evolve."""

import argparse
import csv
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required to generate plots. Install the project plotting dependencies first."
    ) from exc

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9.5,
    }
)


REQUIRED_COLUMNS = {
    "algorithm",
    "backend",
    "system_size",
    "phase",
    "nprocs",
    "mean_evolve_s",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]
LINESTYLES = ["--", "-", "-.", ":"]
SERIES_COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#1f78b4",
]
FIGURE_SIZE = (10.2, 4.2)
LINE_WIDTH = 2.0
MARKER_SIZE = 6.2
MARKER_EDGE_WIDTH = 1.2
ANNOTATION_FONT_SIZE = 12
LEGEND_TITLE_FONT_SIZE = 9.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot strong-scaling speedup and weak-scaling efficiency for each algorithm.",
    )
    parser.add_argument("csv_path", type=Path, help="Path to benchmarks/results/summary.csv.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory for output figures. Defaults to <csv-dir>/plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure resolution for saved PNGs.",
    )
    parser.add_argument(
        "--ranks-per-gpu",
        type=int,
        default=8,
        help="Number of MPI ranks that correspond to one GPU for x-axis labels.",
    )
    return parser.parse_args()


def read_rows(csv_path):
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise SystemExit(f"Missing required columns in {csv_path}: {missing_str}")

        rows = []
        for raw in reader:
            if not raw.get("algorithm"):
                continue
            rows.append(
                {
                    "algorithm": raw["algorithm"],
                    "backend": raw["backend"],
                    "phase": raw["phase"],
                    "system_size": int(raw["system_size"]),
                    "size_spec": raw.get("size_spec") or raw["system_size"],
                    "tensor_dims": raw.get("tensor_dims") or "",
                    "nprocs": int(raw["nprocs"]),
                    "mean_evolve_s": float(raw["mean_evolve_s"]),
                }
            )

    if not rows:
        raise SystemExit(f"No benchmark rows found in {csv_path}")

    return rows


def build_series(rows, mode, algorithm):
    groups = defaultdict(list)
    min_nprocs_by_backend_phase = {}

    if mode == "weak":
        for row in rows:
            backend_phase = (row["backend"], row["phase"])
            current = min_nprocs_by_backend_phase.get(backend_phase)
            if current is None or row["nprocs"] < current:
                min_nprocs_by_backend_phase[backend_phase] = row["nprocs"]

    for row in rows:
        if mode == "strong":
            key = (
                row["system_size"],
                row["size_spec"],
                row["tensor_dims"],
                row["backend"],
                row["phase"],
            )
        else:
            work_per_rank = Fraction(row["system_size"], row["nprocs"])
            if algorithm == "qmoa":
                key = (
                    work_per_rank,
                    row["size_spec"],
                    row["tensor_dims"],
                    row["backend"],
                    row["phase"],
                )
            else:
                key = (work_per_rank, "", "", row["backend"], row["phase"])
        groups[key].append(row)

    series = {}
    for key, points in groups.items():
        ordered = sorted(points, key=lambda point: point["nprocs"])
        unique_nprocs = {point["nprocs"] for point in ordered}
        min_points = 3 if mode == "weak" else 2
        if len(unique_nprocs) < min_points:
            continue
        if mode == "strong" and not has_speedup_above_one(ordered):
            continue
        if mode == "weak":
            _, _, _, backend, phase = key
            baseline_nprocs = min_nprocs_by_backend_phase[(backend, phase)]
            if ordered[0]["nprocs"] != baseline_nprocs:
                continue
        series[key] = ordered

    return dict(sorted(series.items(), key=series_sort_key))


def series_sort_key(item):
    value, size_spec, _, backend, phase = item[0]
    numeric_value = value if isinstance(value, Fraction) else Fraction(value, 1)
    return (numeric_value, size_spec, backend, phase)


def is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def format_qubit_count(value):
    if isinstance(value, Fraction):
        numerator = value.numerator
        denominator = value.denominator
        if is_power_of_two(numerator) and is_power_of_two(denominator):
            qubits = numerator.bit_length() - denominator.bit_length()
            return str(qubits)
        return f"{math.log2(float(value)):.2f}"

    if is_power_of_two(value):
        return str(value.bit_length() - 1)
    return f"{math.log2(value):.2f}"


def make_label(
    algorithm, value, size_spec, tensor_dims,
    backend, phase, include_backend, include_phase,
):
    if algorithm == "qmoa":
        parts = [tensor_dims or f"Ns={size_spec}"]
    else:
        parts = [format_qubit_count(value)]
    if include_backend:
        parts.append(backend)
    if include_phase:
        parts.append(phase)
    return " | ".join(parts)


def format_gpu_count(nprocs, ranks_per_gpu):
    gpus = Fraction(nprocs, ranks_per_gpu)
    if gpus.denominator == 1:
        return str(gpus.numerator)
    return f"{float(gpus):g}"


def has_speedup_above_one(points, tolerance=1e-9):
    ref_time = points[0]["mean_evolve_s"]
    for point in points[1:]:
        if point["mean_evolve_s"] <= 0:
            continue
        if (ref_time / point["mean_evolve_s"]) > (1.0 + tolerance):
            return True
    return False


def compute_metric(points, mode):
    ref_time = points[0]["mean_evolve_s"]
    values = []
    for point in points:
        if point["mean_evolve_s"] <= 0:
            values.append(float("nan"))
            continue
        ratio = ref_time / point["mean_evolve_s"]
        if mode == "strong":
            values.append(ratio)
        else:
            values.append(100.0 * ratio)
    return values


def get_series_style(index):
    marker = MARKERS[index % len(MARKERS)]
    linestyle = LINESTYLES[(index // len(MARKERS)) % len(LINESTYLES)]
    color = SERIES_COLORS[index % len(SERIES_COLORS)]
    return {
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
        "linewidth": LINE_WIDTH,
        "markersize": MARKER_SIZE,
        "markerfacecolor": "white",
        "markeredgecolor": color,
        "markeredgewidth": MARKER_EDGE_WIDTH,
    }


def add_side_legend_title(fig, legend, title):
    if legend is None:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    fig.text(
        bbox.x0 - 0.01,
        0.5 * (bbox.y0 + bbox.y1),
        title,
        ha="right",
        va="center",
        fontsize=LEGEND_TITLE_FONT_SIZE,
    )


def plot_panel(
    ax,
    algorithm,
    series,
    include_backend,
    include_phase,
    ranks_per_gpu,
    metric_mode,
    ylabel,
    annotation_text,
    annotation_loc,
):
    ax.set_xlabel("MPI ranks\nGPUs")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_ylabel(ylabel)
    if annotation_loc == "left":
        xy = (0.03, 0.93)
        ha = "left"
    else:
        xy = (0.97, 0.93)
        ha = "right"
    ax.text(
        xy[0],
        xy[1],
        annotation_text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        fontweight="bold",
    )

    if not series:
        ax.text(
            0.5,
            0.5,
            "No series passed the filters",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return None

    x_ticks = sorted({point["nprocs"] for points in series.values() for point in points})
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick}\n{format_gpu_count(tick, ranks_per_gpu)}" for tick in x_ticks])

    for index, (
        (value, size_spec, tensor_dims, backend, phase),
        points,
    ) in enumerate(series.items()):
        xs = [point["nprocs"] for point in points]
        ys = compute_metric(points, metric_mode)
        label = make_label(
            algorithm, value, size_spec, tensor_dims,
            backend, phase,
            include_backend, include_phase,
        )
        style = get_series_style(index)
        ax.plot(xs, ys, label=label, **style)

    if metric_mode == "weak":
        ax.axhline(100.0, color="0.45", linestyle="--", linewidth=1.0)
    ax.set_ylim(bottom=0)
    return ax.legend(
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=5,
    )


def plot_algorithm(algorithm, rows, outdir, dpi, ranks_per_gpu):
    strong = build_series(rows, mode="strong", algorithm=algorithm)
    weak = build_series(rows, mode="weak", algorithm=algorithm)

    backends = {row["backend"] for row in rows}
    phases = {row["phase"] for row in rows}
    include_backend = len(backends) > 1
    include_phase = len(phases) > 1
    algorithm_label = algorithm.upper()

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    fig.subplots_adjust(bottom=0.33, top=0.95, wspace=0.24)

    strong_legend = plot_panel(
        axes[0],
        algorithm,
        strong,
        include_backend=include_backend,
        include_phase=include_phase,
        ranks_per_gpu=ranks_per_gpu,
        metric_mode="strong",
        ylabel="Speedup",
        annotation_text=algorithm_label,
        annotation_loc="left",
    )
    weak_legend = plot_panel(
        axes[1],
        algorithm,
        weak,
        include_backend=include_backend,
        include_phase=include_phase,
        ranks_per_gpu=ranks_per_gpu,
        metric_mode="weak",
        ylabel="Efficiency [%]",
        annotation_text=algorithm_label,
        annotation_loc="right",
    )

    if algorithm == "qmoa":
        add_side_legend_title(fig, strong_legend, "Tensor dims :")
        add_side_legend_title(fig, weak_legend, "Tensor dims :")
    else:
        add_side_legend_title(fig, strong_legend, "Qubits :")
        add_side_legend_title(fig, weak_legend, "Qubits / rank :")

    png_path = outdir / f"{algorithm}_scaling.png"
    pdf_path = outdir / f"{algorithm}_scaling.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def main():
    args = parse_args()
    if args.ranks_per_gpu <= 0:
        raise SystemExit("--ranks-per-gpu must be a positive integer")
    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    outdir = (args.outdir or (csv_path.parent / "plots")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)

    rows_by_algorithm = defaultdict(list)
    for row in rows:
        rows_by_algorithm[row["algorithm"]].append(row)

    written = []
    for algorithm in sorted(rows_by_algorithm):
        written.extend(
            plot_algorithm(
                algorithm,
                rows_by_algorithm[algorithm],
                outdir,
                args.dpi,
                args.ranks_per_gpu,
            )
        )

    for outpath in written:
        print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
