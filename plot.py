import numpy as np
from matplotlib import pyplot as plt
from anytree import LevelOrderIter
from typing import Dict


def plot_loss(losses: Dict, out_name: str = "loss.png"):
    if len(losses.keys()) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
    ax.set_xlabel("Iterations")
    color1, color2 = ["tab:blue", "tab:red"]
    n_iters = len(losses["htriplet_loss"])

    # plot regularization loss in log scale
    if len(losses["htriplet_loss"]) > 0:
        # ax.semilogy(
        ax.plot(
            losses["htriplet_loss"],
            marker=".",
            c=color1,
            label="Regularization Triplet loss",
            markevery=[i * 10 for i in range(n_iters // 10 + 1)],
        )
        ax.tick_params(axis="y", labelcolor=color1)

    # plot new kl_loss
    kl_loss = losses["new_loss"]
    if len(kl_loss) > 0:
        ax2 = ax.twinx()  # share x-axis
        ax2.set_ylim(top=1.1 * max(kl_loss), bottom=0.9 * min(kl_loss))
        ax2.plot(
            [i * 50 for i in range(n_iters // 50 + 1)],
            kl_loss[:-1],
            marker="^",
            c=color2,
            label="New HC-tSNE loss",
        )
        ax2.tick_params(axis="y", labelcolor=color2)

    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.98))
    fig.savefig(out_name, bbox_inches="tight")


def scatter(
    Z_train,
    Z_test,
    y_train=None,
    y_test=None,
    tree=None,
    out_name="Z_tree.png",
    focus_factor=2.5,
    width=600,
    height=600,
    show_group=None,
):
    # size = 5.0 / np.log(len(Z_init) / 1000)
    size = 300.0 / np.sqrt(Z_train.shape[0])
    dpi = plt.rcParams["figure.dpi"]
    fig_width, fig_height = width / dpi, height / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    ax.axis("off")

    # initial embedding with user groups
    if focus_factor:
        _set_limit(ax, Z_train, focus_factor)
    ax.scatter(*Z_train.T, c=y_train, s=size, cmap="tab10", alpha=0.25)

    # annotation for hc-tsne. For other method, pass a flat tree.
    if tree is not None:
        _annotate_groups(tree, Z=Z_train, ax=ax, show_group=show_group)

    if Z_test is not None:
        ax.scatter(
            *Z_test.T, c=y_test, cmap="tab10", alpha=0.8, marker="+", s=(3 * size)
        )

    fig.savefig(out_name, bbox_inches="tight")


def _set_limit(ax, Z, focus_factor=2.5):
    mu = np.mean(Z, axis=0)
    lim = focus_factor * np.std(Z, axis=0)
    ax.set_xlim(mu[0] - lim[0], mu[0] + lim[0])
    ax.set_ylim(mu[1] - lim[1], mu[1] + lim[1])


def _annotate_node(node, Z, ax, show_group="all", **styles):
    centroid = node.centroid

    if show_group == "text":
        props = dict(boxstyle="round", facecolor="none", alpha=0.5)
        ax.text(
            *centroid.T,
            s=node.name,
            fontsize=11,
            horizontalalignment="center",
            verticalalignment="center",
            bbox=props,
        )
    elif show_group == "all":
        ...
        # point_in_node = Z[node.items]
        # ax.scatter(*centroid.T, c="blue", marker="+", s=64)
        # ax.text(*centroid.T, s=node.name, fontsize=16, alpha=styles.get("alpha", 0.5))
        # confidence_ellipse(*point_in_node.T, ax=ax, n_std=2.0, **styles)
    else:
        ax.scatter(*centroid.T, c="blue", marker="+", s=18)


def _annotate_groups(tree, Z, ax, show_group="all"):
    for node in LevelOrderIter(tree):
        node.update_centroid(Z)
        if len(node.children) == 0:  # show the child nodes
            _annotate_node(
                node, Z, ax, show_group, edgecolor="r", linestyle="--", alpha=0.5
            )
        elif show_group == "all":  # show all intermediate nodes
            _annotate_node(
                node, Z, ax, show_group, edgecolor="g", linestyle="--", alpha=0.3
            )
