# plot score for multiple run
import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from collections import defaultdict
from pprint import pprint


def load_scores(cache=True, score_file_name="scores.json"):
    if cache and os.path.exists(score_file_name):
        with open(score_file_name, "r") as in_file:
            scores = json.load(in_file)
        return scores

    # if file not exist or not use cache, load all score files
    scores = defaultdict(dict)

    for dataset_name in dataset_names:
        scores[dataset_name] = defaultdict(dict)

        for method_name in method_names:
            scores[dataset_name][method_name] = defaultdict(list)

            if method_name in ["tsne", "hc-tsne"]:
                s_name = hctsne_pname
            else:
                s_name = method_name

            for seed in range(start_seed, end_seed + 1):
                score_path = f"{score_dir}/{dataset_name}/score-{s_name}-{seed}.json"
                print(score_path)
                with open(score_path, "r") as in_file:
                    data = json.load(in_file)

                key = f"{method_name}_train"
                if key in data:
                    for score_name in score_names:
                        s = data[key][score_name]
                        scores[dataset_name][method_name][score_name].append(s)

            for score_name in score_names:
                ss = scores[dataset_name][method_name][score_name]
                mean, std = np.mean(ss), np.std(ss)
                scores[dataset_name][method_name][score_name] = [mean, std]

    with open(score_file_name, "w") as out_file:
        json.dump(scores, out_file, indent=2)

    return scores


def plot_scores(scores, out_name="scores.png"):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(7.5, 2.2))
    ax0 = axes[0]

    y_pos = np.arange(len(method_names))
    ax0.set_yticks(y_pos)
    ax0.yaxis.set_tick_params(length=0)
    ax0.set_yticklabels(list(method_names.values()), fontsize=16)
    ax0.invert_yaxis()  # labels read top-to-bottom

    for ax, (score_name, score_info) in zip(axes.ravel(), score_names.items()):
        score_name_display, min_val, max_val, color = score_info
        ax.set_title(score_name_display, color=color, fontsize=18)
        score_values, error = zip(
            *[scores[method_name][score_name] for method_name in method_names]
        )
        ax.barh(y_pos, score_values, xerr=error, color=color, align="center")
        ax.set_xlim(left=min_val, right=max_val)

    fig.savefig(out_name, bbox_inches="tight")


if __name__ == "__main__":
    dataset_names = {"mnist": "MNIST", "fmnist": "Fashion-MNIST", "cifar10": "CIFAR10"}
    score_names = {
        "auc_rnx": ("$AUC[R_{NX}]$", 0.0, 0.5, "tab:blue"),
        "auc_knn": ("$AUC[G_{NN}]$", -0.1, 0.6, "tab:orange"),
        "knn10": ("KNN(10)", 0.3, 1.0, "tab:green"),
    }
    method_names = {
        "tsne": "$t$-SNE",
        "catsne": "$cat$-SNE",
        "umap": "UMAP",
        "hc-tsne": "HC$t$-SNE",
    }

    hctsne_pname = "d2-m0.5"  # param name for hctsne

    score_dir = "./scores_10runs"
    out_name = "./plots/scores_10run.png"
    start_seed, end_seed = 2020, 2029

    scores = load_scores(cache=True, score_file_name=f"{score_dir}/scores_avg.json")
    for dataset_name in dataset_names:
        plot_scores(scores[dataset_name], out_name=f"./plots/scores-{dataset_name}.png")
