# main script to run experiment with hierarchical triplet constraints for tSNE


from datasets import load_dataset
from hierarchical_constraint import generate_constraints
from hierarchical_constraint import show_tree, show_iterating_tree
from hc_tsne import tsne, hc_tsne
from loss_logger import LossLogger


params_config = {
    "mnist": {
        "Z_init": dict(
            perplexity=50, n_iter=500, random_state=2020, n_jobs=-2, verbose=2
        ),
        "Z_new": dict(
            perplexity=100,
            n_iter=100,
            random_state=2020,
            n_jobs=-2,
            verbose=2,
            callbacks_every_iters=10,
            early_exaggeration_iter=0,
        ),
        "hc": dict(margin=0.5, weights=(0.5, 0.5, 0.0)),
        "alpha0": 5e-4,
        "alpha1": 2e-4,
        "alpha2": 2e-4,
    },
    "fmnist": {
        "Z_init": dict(
            perplexity=50, n_iter=500, random_state=2020, n_jobs=-2, verbose=2
        ),
        "Z_new": dict(
            perplexity=100,
            n_iter=100,
            random_state=2020,
            n_jobs=-2,
            verbose=2,
            callbacks_every_iters=10,
            early_exaggeration_iter=0,
        ),
        "hc": dict(margin=0.5, weights=(0.5, 0.5, 0.0)),
        "alpha0": 6e-4,
        "alpha1": 6e-4,
        "alpha2": 7.5e-4,
    },
    "cifar10": {
        "Z_init": dict(
            perplexity=50, n_iter=500, random_state=2020, n_jobs=-2, verbose=2
        ),
        "Z_new": dict(
            perplexity=100,
            n_iter=100,
            random_state=2020,
            n_jobs=-2,
            verbose=2,
            callbacks_every_iters=10,
            early_exaggeration_iter=0,
        ),
        "hc": dict(margin=0.5, weights=(0.5, 0.5, 0.0)),
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
    Z0 = tsne(X_train, **config["Z_init"])
    Z0_test = Z0.transform(X_test)

    # loss logger object to store loss value in each iteration
    logger = LossLogger()

    # run hc_tsne
    Z1 = hc_tsne(
        X_train,
        initialization=Z0,
        tree=tree,
        alpha=alpha,
        loss_logger=logger,
        **config["hc"],
        **config["Z_new"],
    )
    Z1_test = Z1.transform(X_test)

    # test knn score
    calculate_KNN_score(y_train, Z0, Z1)
    calculate_KNN_score(y_test, Z0_test, Z1_test)

    print(logger.get_loss("kl_loss"))
    print(logger.get_loss("new_loss"))
    print(logger.get_loss("htriplet_loss"))


def calculate_KNN_score(labels, Z_init, Z_new, K=5):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=K, n_jobs=-1, algorithm="auto")

    old_score = knn.fit(X=Z_init, y=labels).score(X=Z_init, y=labels)
    new_score = knn.fit(X=Z_new, y=labels).score(X=Z_new, y=labels)

    print(f"K={K}, old_score={old_score:.3f}, new_score={new_score:.3f}")


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
