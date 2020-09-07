# plot score for multiple run
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_scores():
    ...


def plot_scores(scores, out_name="scores.png"):
    ...


if __name__ == "__main__":
    dataset_names = {"mnist": "MNIST", "fmnist": "Fashion-MNIST", "cifar10": "CIFAR10"}

    method_names = ["tsne", "hc-tsne", "catsne", "umap"]

    plot_dir = "./scores_10runs"
    hctsne_pname = "d2-m0.5"  # param name for hctsne
    start_seed, end_seed = 2020, 2029

    out_name = "./plots/scores_10run.png"
    scores = load_scores()
    plot_scores(scores, out_name)
