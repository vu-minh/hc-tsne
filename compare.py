# Run Neighborhood Components Analysis

import os
import joblib
import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from umap import UMAP

from datasets import load_dataset
from hierarchical_constraint import generate_constraints_flat
from plot import scatter
from logger import ScoreLogger
from score import evaluate_scores
from catsne import catsne


def run(args, flat_tree):
    name_suffix = f"{args.method}-{args.seed}"
    run_func = {"nca": run_nca, "umap": run_umap, "catsne": run_catsne}[args.method]
    Z, Z_test = run_func(args)
    joblib.dump(Z, f"{Z_dir}/Z-{args.method}.z")
    joblib.dump(Z_test, f"{Z_dir}/Z_test-{args.method}.z")

    scatter(
        Z,
        None,  # Z_test
        y_train,
        y_test,
        tree=flat_tree,
        out_name=f"{plot_dir}/{name_suffix}.png",
        show_group="text",
    )

    score_name = f"{score_dir}/score-{name_suffix}.json"
    score_logger = None if args.no_score else ScoreLogger(score_name)
    evaluate_scores(
        X_train, y_train, X_test, y_test, Z, Z_test, args.method, score_logger
    )

    # important: save the logger filer
    if score_logger is not None:
        score_logger.dump()
        score_logger.print()


def run_nca(args):
    nca = NeighborhoodComponentsAnalysis(
        n_components=2, init=args.nca_init, max_iter=100, verbose=2, random_state=42
    )
    nca.fit(X_train, y_train)
    Z = nca.transform(X_train)
    Z_test = nca.transform(X_test)
    return Z, Z_test


def run_umap(args):
    mapper = UMAP(n_neighbors=args.n_neighbors, random_state=args.seed, init="random")
    mapper.fit(X_train, y=y_train)
    Z = mapper.embedding_
    Z_test = mapper.transform(X_test)
    return Z, Z_test


def run_catsne(args):
    random_state = np.random.RandomState(args.seed)
    Z, _ = catsne(X_train, y_train, rand_state=random_state, init="ran")
    return Z, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    argm = parser.add_argument

    argm("--dataset_name", "-d")
    argm("--no-score", action="store_true", help="Do not calculate metric scores")
    argm("--method", "-m", default="nca", help="Run different methods like umap, nca")
    argm("--seed", "-s", default=2020, type=int, help="Random seed")

    argm("--pca", default=0.95, type=float, help="Run PCA on raw data")
    argm("--n_train", default=10000, type=int, help="# datapoints for training set")
    argm("--n_test", default=5000, type=int, help="# datasetpoints fro test set")

    argm("--nca_init", default="auto", help="NCA initialization params: auto, pca, lda")
    argm("--n_neighbors", default=10, type=int, help="UMAP n_neighbors")
    args = parser.parse_args()
    print(args)

    # prepare directories for storing figures and dump embeddings.
    base_dir = ["./", "/content/drive/My Drive/Colab Notebooks/HC-tSNE"][1]
    plot_dir, Z_dir, score_dir = [
        f"{base_dir}/{dir_name}/{args.dataset_name}"
        for dir_name in ["plots", "Z", "scores"]
    ]
    for d in [plot_dir, Z_dir, score_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    # load data (can do PCA)
    (X_train, y_train), (X_test, y_test), label_names = load_dataset(
        args.dataset_name, args.n_train, args.n_test, pca=args.pca, debug=True
    )

    # create flat tree for easily annotation the group name in the visualization
    flat_tree = generate_constraints_flat(y_train, label_names)

    # run a methode, store embedding and calculate some metric
    run(args, flat_tree)
