"""Script to plot the accuracy as a function of a parameter
(typically separability or unlabeled proportion)
for a given number of features
(fixed in the case of non-synthetic data)"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["axes.titlesize"] = 15  # titre du plot
mpl.rcParams["axes.labelsize"] = 13  # titres des axes
mpl.rcParams["legend.fontsize"] = 11  # legende des courbes

results_dir = "results_semi/"


df = pd.read_csv(results_dir + f"Acc_Sep_Synth.csv", index_col="Param", sep=";")


colors = ["green", "blue", "orange", "red", "purple"]  # 1 color per algo

# for synthetic data, put here a list of all tested dimensions
# Note: to fill a csv with accuracies obtained with a changing number of features,
# use the parameter resume=True in synthetic_semisup_tests.py
n_features = [1000, 10000]  # synthetic data


plt.figure(figsize=(8, 4))  # good size for 2 values of d
for i, nf in enumerate(n_features):
    plt.subplot(1, len(n_features), i + 1)
    plt.plot(
        df.index,
        df[f"Mean_{nf}f_LabProp"],
        label=f"LabProp",
        c=colors[0],
        marker="x",
        alpha=0.9,
        linestyle="dashed",
    )
    plt.plot(
        df.index,
        df[f"Mean_{nf}f_LabSpread"],
        label=f"LabSpread",
        c=colors[1],
        # alpha=0.5,
        marker="1",
        linestyle="dotted",
    )
    plt.plot(
        df.index,
        df[f"Mean_{nf}f_NN_SWA"],
        label=f"NN",
        alpha=0.7,
        c=colors[3],
        marker="o",
    )
    plt.plot(
        df.index, df[f"Mean_{nf}f_SAE_SWA_2nd"], label=f"SAE", c=colors[2], marker="o",
    )

    plt.legend()

    plt.xlabel(f"Data separability, d={nf}")

    if i == 0:
        # label the left-most y axis
        plt.ylabel("Accuracy")

    plt.ylim(bottom=0.48, top=1.01)
    plt.grid(axis="y")

plt.savefig(
    "plots/synth_acc_sep.png", dpi=400, facecolor="white", bbox_inches="tight",
)
plt.show()
