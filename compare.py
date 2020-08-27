# Run Neighborhood Components Analysis

import os
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from umap import UMAP

from datasets import load_dataset
from hierarchical_constraint import generate_constraints_flat
from plot import scatter
from score import simple_KNN_score


def run(args, flat_tree):
    run_func = {"nca": run_nca, "umap": run_umap}[args.method]
    Z0, Z1 = run_func(args)

    out_name = f"{plot_dir}/{args.method}.png"
    scatter(
        Z0,
        None,
        y_train,
        y_test,
        tree=flat_tree,
        out_name=out_name,
        show_group="text",
    )

    print(args.method, "KNN(5) score")
    simple_KNN_score([Z0], y_train)
    simple_KNN_score([Z1], y_test)


def run_nca(args):
    nca = NeighborhoodComponentsAnalysis(
        n_components=2, init=args.nca_init, max_iter=100, verbose=2, random_state=42
    )
    nca.fit(X_train, y_train)
    Z0 = nca.transform(X_train)
    Z1 = nca.transform(X_test)
    return Z0, Z1


def run_umap(args):
    mapper = UMAP(n_neighbors=args.n_neighbors)
    mapper.fit(X_train, y=y_train)
    Z0 = mapper.embedding_
    Z1 = mapper.transform(X_test)
    return Z0, Z1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    argm = parser.add_argument

    argm("--dataset_name", "-d")
    argm("--method", "-m", default="nca", help="Run different methods like umap, nca")
    argm("--pca", default=0.9, type=float, help="Run PCA on raw data")
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
