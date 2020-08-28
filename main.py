# main script to run experiment with hierarchical triplet constraints for tSNE

import os
import joblib
import numpy as np

from datasets import load_dataset
from hierarchical_constraint import generate_constraints
from hierarchical_constraint import show_tree
from hc_tsne import tsne, hc_tsne
from logger import LossLogger, ScoreLogger
from plot import scatter, plot_loss
from score import evaluate_scores


def run_tsne(config, score_logger, rerun=True):
    Z0_name, Z0_test_name = f"{Z_dir}/Z0.z", f"{Z_dir}/Z0_test.z"

    if rerun or not os.path.exists(Z0_name):
        print("\n[DEBUG]Run original TSNE with ", config)

        Z0 = tsne(X_train, **config)
        Z0_test = Z0.transform(X_test)
        joblib.dump(np.array(Z0), Z0_name)
        joblib.dump(np.array(Z0_test), Z0_test_name)
    else:
        Z0 = joblib.load(Z0_name)
        Z0_test = joblib.load(Z0_test_name)

    scatter(Z0, None, y_train, None, out_name=f"{plot_dir}/Z0.png", show_group=None)

    if score_logger is not None:
        evaluate_scores(
            X_train, y_train, X_test, y_test, Z0, Z0_test, "tsne", score_logger
        )

    return Z0, Z0_test  # Z0 is used an initialization in hc_tsne


def run_hc_tsne(Z_init, tree, alpha, margin, config, score_logger, rerun=False):
    Z1_name, Z1_test_name = f"{Z_dir}/Z1.z", f"{Z_dir}/Z1_test.z"
    loss_name = f"{score_dir}/loss-{name_suffix}.json"
    loss_logger = LossLogger(loss_name)

    if rerun or not os.path.exists(Z1_name):
        print("\n[DEBUG]Run Hierarchical TSNE with ", config["Z_new"])

        Z1 = hc_tsne(
            X_train,
            initialization=Z_init,
            tree=tree,
            alpha=alpha,
            margin=margin,
            loss_logger=loss_logger,
            **config["hc"],
            **config["Z_new"],
        )
        Z1_test = Z1.transform(X_test)
        loss_logger.dump()
        joblib.dump(np.array(Z1), Z1_name)
        joblib.dump(np.array(Z1_test), Z1_test_name)
    else:
        Z1 = joblib.load(Z1_name)
        Z1_test = joblib.load(Z1_test_name)

    fig_name = f"{plot_dir}/HC-{name_suffix}.png"
    scatter(Z1, None, y_train, None, tree=tree, out_name=fig_name)

    loss_logger.load(loss_name)
    plot_loss(loss_logger.loss, out_name=f"{plot_dir}/loss-{name_suffix}.png")

    if score_logger is not None:
        evaluate_scores(
            X_train, y_train, X_test, y_test, Z1, Z1_test, "hc-tsne", score_logger
        )


def main(args):
    # load param config
    config = params_config[args.dataset_name]

    # score logger
    score_name = f"{score_dir}/score-{name_suffix}.json"
    score_logger = None if args.no_score else ScoreLogger(score_name)

    # run original tsne
    Z0, _ = run_tsne(
        config=config["Z_init"], score_logger=score_logger, rerun=args.rerun0
    )

    # build hierarchical constraint in tree form
    tree = generate_constraints(
        args.dataset_name,
        labels=y_train,
        label_names=label_names,
        depth=args.depth,
        label_percent=args.label_percent,
        tree_name=f"{plot_dir}/tree-d{args.depth}.png",
    )
    show_tree(tree)

    run_hc_tsne(
        Z_init=Z0,
        tree=tree,
        alpha=config[f"alpha{int(args.depth)}"],
        margin=args.margin,
        config=config,
        score_logger=score_logger,
        rerun=args.rerun1,
    )

    # important: save the logger filer
    if score_logger is not None:
        score_logger.dump()
        score_logger.print()


params_config = {
    "mnist": {
        "Z_init": dict(
            perplexity=50, n_iter=500, random_state=2020, n_jobs=-1, verbose=2
        ),
        "Z_new": dict(
            perplexity=50,
            n_iter=100,
            random_state=2020,
            n_jobs=-1,
            verbose=2,
            callbacks_every_iters=10,
            early_exaggeration_iter=0,
        ),
        "hc": dict(weights=(0.5, 0.5, 0.0)),
        "alpha0": 5e-4,
        "alpha1": 5e-4,
        "alpha2": 7.5e-4,
    },
    "fmnist": {
        "Z_init": dict(
            perplexity=50, n_iter=500, random_state=2020, n_jobs=-2, verbose=2
        ),
        "Z_new": dict(
            perplexity=50,
            n_iter=100,
            random_state=2020,
            n_jobs=-2,
            verbose=2,
            use_callbacks=True,
            callbacks_every_iters=10,
            early_exaggeration_iter=0,
        ),
        "hc": dict(weights=(0.5, 0.5, 0.0)),
        "alpha0": 1e-3,
        "alpha1": 6e-4,
        "alpha2": 7.5e-4,  # 1e-2
    },
    "cifar10": {
        "Z_init": dict(
            perplexity=50, n_iter=500, random_state=2020, n_jobs=-2, verbose=2
        ),
        "Z_new": dict(
            perplexity=50,
            n_iter=100,
            random_state=2020,
            n_jobs=-2,
            verbose=2,
            callbacks_every_iters=10,
            early_exaggeration_iter=0,
        ),
        "hc": dict(weights=(0.5, 0.5, 0.0)),
        "alpha0": 1.5e-3,
        "alpha1": 1.25e-3,
        "alpha2": 5e-3,
    },
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    argm = parser.add_argument

    argm("--rerun0", action="store_true", help="Rerun original tsne")
    argm("--rerun1", action="store_true", help="Rerun new hc-tsne")
    argm("--no-score", action="store_true", help="Do not calculate metric scores")

    argm("--dataset_name", "-d")
    argm("--pca", default=0.95, type=float, help="Run PCA on raw data")
    argm("--n_train", default=10000, type=int, help="# datapoints for training set")
    argm("--n_test", default=5000, type=int, help="# datasetpoints fro test set")

    argm("--depth", default=2, type=int, help="Depth of tree in the hierarchy.")
    argm("--label_percent", default=1.0, type=float, help="% label used in each group.")

    argm("--margin", "-m", default=0.5, type=float, help="Relative margin tripletloss")

    args = parser.parse_args()
    print(args)

    base_dir = ["./", "/content/drive/My Drive/Colab Notebooks/HC-tSNE"][1]
    plot_dir, Z_dir, score_dir = [
        f"{base_dir}/{dir_name}/{args.dataset_name}"
        for dir_name in ["plots", "Z", "scores"]
    ]
    for d in [plot_dir, Z_dir, score_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    name_suffix = f"d{args.depth}-m{args.margin}"

    (X_train, y_train), (X_test, y_test), label_names = load_dataset(
        args.dataset_name, args.n_train, args.n_test, pca=args.pca, debug=True
    )

    main(args)
