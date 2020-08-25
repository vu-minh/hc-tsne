# main script to run experiment with hierarchical triplet constraints for tSNE


from datasets import load_dataset
from hierarical_constraint import generate_constraints
from hierarical_constraint import show_tree, show_iterating_tree
from hc_tsne import tsne, hc_tsne


params_config = {
    "mnist": {
        "Z_init": dict(perplexity=50, n_iter=500, random_state=4321, verbose=2),
        "Z_new": dict(
            perplexity=100,
            n_iter=100,
            random_state=4321,
            verbose=2,
            early_exaggeration_iter=0,
        ),
        "hc": dict(margin=0.5, weights=(0.5, 0.5, 0.0)),
        "alpha0": 5e-4,
        "alpha1": 2e-4,
        "alpha2": 2e-4,
    },
    "fmnist": {
        "Z_init": dict(perplexity=50, n_iter=500, random_state=4321, verbose=2),
        "Z_new": dict(
            perplexity=100,
            n_iter=50,
            random_state=4321,
            verbose=2,
            early_exaggeration_iter=0,
        ),
        "alpha0": 6e-4,
        "alpha1": 6e-4,
        "alpha2": 7.5e-4,
    },
    "cifar10": {
        "Z_init": dict(perplexity=50, n_iter=500, random_state=4321, verbose=2),
        "Z_new": dict(
            perplexity=100,
            n_iter=50,
            random_state=4321,
            verbose=2,
            early_exaggeration_iter=0,
        ),
        "alpha0": 1.5e-3,
        "alpha1": 1.25e-3,
        "alpha2": 5e-4,
    },
}


def main(args):
    dataset_name = args.dataset_name

    # load dataset
    (X_train, y_train), (X_test, y_test), label_names = load_dataset(
        dataset_name, args.n_train, args.n_test, args.pca, debug=True
    )

    # build hierarchical constraint in tree form
    tree = generate_constraints(
        dataset_name,
        labels=y_train,
        label_names=label_names,
        depth=args.depth,
        label_percent=args.label_percent,
    )
    show_tree(tree)
    show_iterating_tree(tree)

    # load param config
    config = params_config[dataset_name]
    alpha = config[f"alpha{int(args.depth)}"]

    # run original tsne
    Z_init = tsne(X_train, **config["Z_init"])

    # run hc_tsne
    Z_new = hc_tsne(
        X_train,
        initialization=Z_init,
        tree=tree,
        alpha=alpha,
        **config["hc"],
        **config["Z_new"],
    )
    print(Z_new.shape)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    argm = parser.add_argument

    argm("--dataset_name", "-d")
    argm("--pca", default=0.9, type=float, help="Run PCA on raw data")
    argm("--n_train", default=10000, type=int, help="# datapoints for training set")
    argm("--n_test", default=5000, type=int, help="# datasetpoints fro test set")

    argm("--depth", default=2, type=int, help="Depth of tree in the hierarchy.")
    argm("--label_percent", default=1.0, type=float, help="% label used in each group.")

    argm("--margin", "-m", default=0.5, type=float, help="Relative margin tripletloss")

    args = parser.parse_args()
    print(args)

    main(args)
