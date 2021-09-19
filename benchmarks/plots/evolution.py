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
    leg.get_frame().set_linewidth(0.0)
    fig_leg.savefig(name, dpi=200)


print("Maximum deviation from norm during:")


def evolution_multi(
    test_type,
    base,
    files,
    alg_name,
    qubit_min,
    xlim,
    ylim,
    output_name,
    output_legend_name,
):

    data_df = []
    for path in files:
        data_df.append(pd.read_csv("{}/{}".format(base, path)))

    marker = itertools.cycle(("D", "v", "^", "<", ">", "8", "s", "p", "P", "*"))

    plt.figure(figsize=(5, 4))

    for i, data in enumerate(data_df):
        if i != 0:
            plt.plot(
                data["qubits"][qubit_min:],
                data_df[0]["time"][qubit_min:] / data["time"][qubit_min:],
                "*",
                label="{} cores ({} nodes)".format(
                    data["comm_size"][0], np.int(data["comm_size"][0] / 24)
                ),
                marker=next(marker),
                markersize=8,
            )
    plt.xlabel(r"Qubits ($n$)")
    plt.ylabel(alg_name + r" speedup $(\times)$")
    plt.grid(which="major", linestyle="--")
    plt.xticks(data_df[0]["qubits"][qubit_min:].values)
    plt.yticks([i for i in range(1, int(ylim[1]), 2)])
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    save_legend(output_legend_name, 2)
    plt.clf()

    max_difference = []
    for series in data_df:
        max_difference.append(1 - np.max(series["norm"]))
    print("\t{} evolution: {}".format(test_type, np.max(max_difference)))


def evolution_single(
    test_type,
    base,
    files,
    alg_name,
    qubit_min,
    xlim,
    ylim,
    output_name,
    output_legend_name,
):

    data_df = []
    for path in files:
        data_df.append(pd.read_csv("{}/{}".format(base, path)))

    marker = itertools.cycle(("D", "v", "^", "<", ">", "8", "s", "p", "P", "*"))

    plt.figure(figsize=(5, 4))

    for i, data in enumerate(data_df):
        if i != 0:
            plt.plot(
                data["qubits"][: int(qubit_min)],
                data_df[0]["time"] / data["time"][: int(qubit_min)],
                "*",
                marker=next(marker),
                markersize=8,
                label="{} cores ({} node)".format(
                    data["comm_size"][0], np.int(data["comm_size"][0] / 24 + 1)
                ),
            )

    plt.xlabel(r"Qubits ($n$)")
    plt.ylabel(alg_name + r" speedup $(\times)$")
    plt.grid(which="major", linestyle="--")
    plt.xticks(data_df[0]["qubits"].values[::2])
    plt.yticks([i for i in range(1, int(ylim[1]), 2)])
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    save_legend(output_legend_name, 2)
    plt.clf()

    max_difference = []
    for series in data_df:
        max_difference.append(1 - np.max(series["norm"]))
    print("\t{} evolution: {}".format(test_type, np.max(max_difference)))


qwoa_test = "QWOA multi-node"

qwoa_base = "../results/cluster/qwoa_evolution/csv"

qwoa_files = [
    "1_qwoa_evolution.csv",
    "2_qwoa_evolution.csv",
    "4_qwoa_evolution.csv",
    "6_qwoa_evolution.csv",
    "8_qwoa_evolution.csv",
    "10_qwoa_evolution.csv",
    "12_qwoa_evolution.csv",
    "14_qwoa_evolution.csv",
]

evolution_multi(
    qwoa_test,
    qwoa_base,
    qwoa_files,
    "QWOA",
    12,
    [13.5, 19.5],
    [0, 10],
    "output/qwoa_state_evolution_multi",
    "output/qwoa_state_evolution_multi_legend",
)

qaoa_test = "QAOA multi-node"

qaoa_base = "../results/cluster/qaoa_evolution/csv"

qaoa_files = [
    "1_qaoa_evolution.csv",
    "2_qaoa_evolution.csv",
    "4_qaoa_evolution.csv",
    "6_qaoa_evolution.csv",
    "8_qaoa_evolution.csv",
    "10_qaoa_evolution.csv",
    "12_qaoa_evolution.csv",
    "14_qaoa_evolution.csv",
]

evolution_multi(
    qaoa_test,
    qaoa_base,
    qaoa_files,
    "QAOA",
    16,
    [17.5, 24.5],
    [0, 6.5],
    "output/qaoa_state_evolution_multi",
    "output/qaoa_state_evolution_multi_legend",
)

qwoa_test = "QWOA single-node"

qwoa_base = "../results/workstation/qwoa_evolution/csv/"

qwoa_files = [
    "1_qwoa_evolution.csv",
    "2_qwoa_evolution.csv",
    "4_qwoa_evolution.csv",
    "8_qwoa_evolution.csv",
    "16_qwoa_evolution.csv",
    "24_qwoa_evolution.csv",
]

evolution_single(
    qwoa_test,
    qwoa_base,
    qwoa_files,
    "QWOA",
    18,
    [1, 18],
    [-1, 21],
    "output/qwoa_state_evolution_single_node",
    "output/qwoa_state_evolution_single_node_legend",
)

qaoa_test = "QAOA single-node"

qaoa_base = "../results/workstation/qaoa_evolution/csv/"

qaoa_files = [
    "1_qaoa_evolution.csv",
    "2_qaoa_evolution.csv",
    "4_qaoa_evolution.csv",
    "8_qaoa_evolution.csv",
    "16_qaoa_evolution.csv",
    "24_qaoa_evolution.csv",
]

evolution_single(
    qaoa_test,
    qaoa_base,
    qaoa_files,
    "QAOA",
    22,
    [1, 22],
    [0, 9],
    "output/qaoa_state_evolution_single_node",
    "output/qaoa_state_evolution_single_node_legend",
)
