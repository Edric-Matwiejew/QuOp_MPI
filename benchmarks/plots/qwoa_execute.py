from os import listdir
import itertools
from glob import glob
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import pandas as pd
from adjustText import adjust_text

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 16
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm} \usepackage{amsmath}"
plt.rcParams["mathtext.fontset"] = "cm"


def save_legend(name, col):
    ax = plt.gca()
    fig_leg = plt.figure()
    ax_leg = fig_leg.add_subplot(111)
    leg = ax_leg.legend(
        *ax.get_legend_handles_labels(), loc="center", markerscale=1, ncol=col
    )
    ax_leg.axis("off")
    plt.tight_layout()
    leg.get_frame().set_linewidth(0.0)
    bbox = fig_leg.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    fig_leg.savefig(name, bbox_inches=bbox)


files = [
    "../results/cluster/qwoa_execute/csv/1_qwoa_execute_quop_log.csv",
    "../results/cluster/qwoa_execute/csv/2_qwoa_execute_quop_log.csv",
    "../results/cluster/qwoa_execute/csv/4_qwoa_execute_quop_log.csv",
    "../results/cluster/qwoa_execute/csv/8_qwoa_execute_quop_log.csv",
    "../results/cluster/qwoa_execute/csv/16_qwoa_execute_quop_log.csv",
]
qwoa_execute_quop_dfs = []
for file in files:
    qwoa_execute_quop_dfs.append(pd.read_csv(file))

print("QWOA maximum variational parameters (1 hour wall-time):")
for i, data_df in enumerate(qwoa_execute_quop_dfs):

    plt.plot(
        data_df["ansatz_depth"] * 2, data_df["fun"], ".", color="blue", markersize=8
    )
    plt.vlines(
        np.max(data_df["ansatz_depth"]) * 2,
        np.min(data_df["fun"]) + 0.01,
        0.185 - i * 0.02,
        color="black",
    )
    plt.annotate(
        str(int(data_df["MPI_nodes"][0] / 24)) + " nodes",
        xy=(np.max(data_df["ansatz_depth"] * 2 - 6 - i * 0.5), 0.19 - i * 0.02),
    )

    print(
        "\t {} nodes: {} parameters".format(
            int(data_df["MPI_nodes"][0] / 24), 2 * int(np.max(data_df["ansatz_depth"]))
        )
    )

plt.xlabel(r"Variational Parameters $|\boldsymbol{\theta}|$")
plt.ylabel(r"$\langle \boldsymbol{\theta}_f | \hat{Q} | \boldsymbol{\theta}_f \rangle_\text{QWOA}$")
plt.grid(which="major", linestyle="--")
plt.xticks([i for i in range(0, 91, 10)])
plt.xlim([0, 98])
plt.yticks([i * 0.05 for i in range(0, 6)])
plt.tight_layout()
plt.savefig("output/qwoa_execute_n_params", dpi=200)

files = [
    "../results/cluster/qwoa_execute/csv/1_qwoa_execute_bench_log.csv",
    "../results/cluster/qwoa_execute/csv/2_qwoa_execute_bench_log.csv",
    "../results/cluster/qwoa_execute/csv/4_qwoa_execute_bench_log.csv",
    "../results/cluster/qwoa_execute/csv/8_qwoa_execute_bench_log.csv",
    "../results/cluster/qwoa_execute/csv/16_qwoa_execute_bench_log.csv",
]
qwoa_execute_dfs = []
for file in files:
    qwoa_execute_dfs.append(pd.read_csv(file))

marker = itertools.cycle(("D", "v", "^", "<", ">", "8", "s", "p", "P", "*"))

plt.clf()

plt.figure(figsize=(5, 4))

for i, data_df in enumerate(qwoa_execute_dfs):
    times = np.append(
        data_df["time"][0:1].values,
        data_df["time"][1:].values - data_df["time"][:-1].values,
    )
    plt.plot(
        data_df["depth"] * 2,
        times / 60,
        "*",
        marker=next(marker),
        markersize=8,
        label="{} nodes ({} cores)".format(
            np.int(data_df["comm_size"][0] / 24), int(data_df["comm_size"][0])
        ),
    )

plt.xlabel(r"Variational Parameters  $|\boldsymbol{\theta}|$")
plt.ylabel("Time (min)")
plt.grid(which="major", linestyle="--")
plt.xticks([i for i in range(0, 81, 10)])
plt.xlim([0, 90])
plt.tight_layout()
plt.savefig("output/qwoa_execute_time", dpi=200)
save_legend("output/qwoa_execute_time_l", 2)
