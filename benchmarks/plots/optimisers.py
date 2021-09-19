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
    plt.tight_layout()
    bbox = fig_leg.get_window_extent().transformed(fig_leg.dpi_scale_trans.inverted())
    fig_leg.savefig(name, bbox_inches=bbox)


def optimisers(
    test_value,
    csv_filename,
    npy_glob,
    names_folder,
    label_dict,
    alg_name,
    plot_name,
    legend_name,
):

    optimiser_df = pd.read_csv(csv_filename)

    print("Target minima for {} optimisation tests:".format(alg_name))
    for i in range(5):
        print(
            "\t - theta set {}, fmin = {} achieved by {}".format(
                i,
                np.round(optimiser_df[i * 13 : (i + 1) * 13].fun.min(), 3),
                optimiser_df[i * 13 : (i + 1) * 13][
                    optimiser_df[i * 13 : (i + 1) * 13].fun
                    == optimiser_df[i * 13 : (i + 1) * 13].fun.min()
                ]["label"].values,
            )
        )

    files = glob(npy_glob)

    open_file = open("{}/scipy_names".format(names_folder), "rb")
    scipy_names = pickle.load(open_file)
    open_file.close()

    open_file = open("{}/nlopt_names".format(names_folder), "rb")
    nlopt_names = pickle.load(open_file)
    open_file.close()

    mins = []
    scipy_series = []
    nlopt_series = []
    nlopt_labels = []
    scipy_labels = []

    for file in files:

        if "LD" in file or "LN" in file:

            for name in nlopt_names:
                if name in file:
                    nlopt_series.append(np.load(file))
                    nlopt_labels.append(name)

        for name in scipy_names:

            if name in file and (not (("LD" in file) or ("LN" in file))):
                scipy_labels.append(name)
                scipy_series.append(np.load(file))

    scipy_dict = {name: [] for name in scipy_names}
    for name, series in zip(scipy_labels, scipy_series):
        scipy_dict[name].append(series)

    nlopt_dict = {name: [] for name in nlopt_names}
    for name, series in zip(nlopt_labels, nlopt_series):
        nlopt_dict[name].append(series)

    plt.figure(figsize=(5, 4))
    plt.grid(which="major", linestyle="--")

    test_pass = []
    labels = []
    success = []
    for key in scipy_dict.keys():
        test_pass_repeats = []
        for series in scipy_dict[key]:
            pass_points = np.where(
                np.logical_and(series[1] < test_value, series[0] <= 5000)
            )[0]

            if len(pass_points) > 0:
                test_pass_repeats.append(series[0][pass_points[0]])

        if len(test_pass_repeats) >= 1:

            test_pass.append(np.mean(test_pass_repeats))
            labels.append(str(key))
            success.append(len(test_pass_repeats) / 5)

    points_scipy = plt.plot(success, test_pass, "o", label="SciPy", markersize=6)

    texts = []
    for i, label in enumerate(labels):
        texts.append(plt.text(success[i], test_pass[i], label_dict[label], size=12))

    test_pass = []
    labels = []
    success = []
    for key in nlopt_dict.keys():
        test_pass_repeats = []

        for series in nlopt_dict[key]:
            pass_points = np.where(
                np.logical_and(series[1] < test_value, series[0] <= 5000)
            )[0]
            if len(pass_points) > 0:
                test_pass_repeats.append(series[0][pass_points[0]])

        if len(test_pass_repeats) >= 1:
            test_pass.append(np.mean(test_pass_repeats))
            labels.append(str(key))
            success.append(len(test_pass_repeats) / 5)

    points_nlopt = plt.plot(success, test_pass, "s", label="NLOpt", markersize=6)

    for i, label in enumerate(labels):
        texts.append(plt.text(success[i], test_pass[i], label_dict[label], size=12))

    adjust_text(
        texts,
        move_only={"text": "xy"},
        add_points=points_scipy + points_nlopt,
        force_points=(0.001, 0.1),
        expand_points=(1.2, 1.2),
        limit=500,
    )

    plt.xlim(0.05, 1.1)
    plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlabel("Success Rate")
    ylabel = (
        r"Number of $| \boldsymbol{\theta} \rangle_\text{" + alg_name + "}$ Evaluations"
    )
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(plot_name, dpi=200)
    save_legend(legend_name, 2)
    plt.clf()


label_dict = {
    "BFGS": "BFGS",
    "Powell": "Powell",
    "CG": "CG",
    "trust-constr": "trust-constr",
    "TNC": "TNC",
    "Nelder-Mead": "Nelder-Mead",
    "LN_NELDERMEAD": "Nelder-Mead",
    "LN_PRAXIS": "Praxis",
    "LN_SBPLX": "SBPLX",
    "LD_LBFGS": "BFGS",
    "LN_BOBYQA": "BOBYQA",
    "LD_MMA": "MMA",
    "LD_CCSAQ": "CCSAQ",
}

optimisers(
    0.08,
    "../results/cluster/optimisers/csv/240_qwoa.csv",
    "../results/cluster/optimisers/npy/240_qwoa*16_5.npy",
    "../results/cluster/optimisers/other",
    label_dict,
    "QWOA",
    "output/qwoa_optimiser_comparison",
    "output/qwoa_optimiser_comparison_legend",
)


optimisers(
    0.08,
    "../results/cluster/optimisers/csv/264_qaoa.csv",
    "../results/cluster/optimisers/npy/264_qaoa*16_5.npy",
    "../results/cluster/optimisers/other",
    label_dict,
    "QAOA",
    "output/qaoa_optimiser_comparison",
    "output/qaoa_optimiser_comparison_legend",
)
