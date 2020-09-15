import numpy as np
from matplotlib import pyplot as plt
from anytree import LevelOrderIter
from typing import Dict


def plot_loss(losses: Dict, out_name: str = "loss.png"):
    if len(losses.keys()) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.9, 2.1))
    ax.set_xlabel("Iterations")
    color1, color2 = ["tab:blue", "tab:red"]
    n_iters = len(losses["htriplet_loss"])

    # plot regularization loss in log scale
    if len(losses["htriplet_loss"]) > 0:
        ax.semilogy(  # ax.plot(
            losses["htriplet_loss"],
            marker=".",
            c=color1,
            label="Regularization $\mathcal{L}_{intra}+\mathcal{L}_{inter}$",
            markevery=[i * 10 for i in range(n_iters // 10 + 1)],
        )
        ax.tick_params(axis="y", labelcolor=color1)

    # plot new kl_loss
    kl_loss = losses["new_loss"][1:]
    print("[DEBUG] new loss: ", len(kl_loss))
    if len(kl_loss) > 0:
        ax2 = ax.twinx()  # share x-axis
        ax2.set_ylim(top=1.1 * max(kl_loss), bottom=0.9 * min(kl_loss))
        ax2.plot(
            [i * 10 for i in range(n_iters // 10 + 1)],
            kl_loss,
            marker="^",
            c=color2,
            label="New HCt-SNE loss\n$KL_{loss} + \\alpha (\mathcal{L}_{intra}+\mathcal{L}_{inter})$",
            # markevery=[i * 10 for i in range(n_iters // 10 + 1)],
        )
        ax2.tick_params(axis="y", labelcolor=color2)

    fig.legend(loc="upper right", bbox_to_anchor=(0.87, 1.04))
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
    show_group="text",
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


def plot_samples(imgs, labels, class_id, n_samples=100, out_name="sample.png"):
    import math

    n_rows = int(math.sqrt(n_samples))
    plt.figure(figsize=(20, 20))
    n_plot = 1

    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        if lbl == class_id:
            plt.subplot(n_rows, n_rows, n_plot)
            plt.imshow(img.reshape(32, 32, 3))
            plt.gca().set_title(i)
            plt.axis("off")
            n_plot += 1
            if n_plot > n_samples:
                break

    plt.savefig(out_name)


def demo_l2_distance(img1, img2, img3, plot_dir="plots"):
    for img, name in zip([img1, img2, img3], ["airplane1", "airplane2", "bird1"]):
        plt.figure(figsize=(1, 1))
        plt.imshow(img.reshape((32, 32, 3)))
        plt.axis("off")
        plt.savefig(f"{plot_dir}/{name}.png")

    d12 = np.linalg.norm(img1 - img2) ** 2
    d13 = np.linalg.norm(img1 - img3) ** 2

    print(f"d12: {d12:.3f}", f"d13: {d13:.3f}")


def image_grid(imgs, Z, img_size=32, n_img=20, out_name="grid.png"):
    # Ref: h ttps://github.com/prabodhhere/tsne-grid
    from tensorflow.python.keras.preprocessing import image
    from scipy.spatial.distance import cdist
    from lapjv import lapjv

    grid = np.dstack(
        np.meshgrid(np.linspace(0, 1, n_img), np.linspace(0, 1, n_img))
    ).reshape(-1, 2)
    cost_matrix = cdist(grid, Z, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]
    out = np.ones((n_img * img_size, n_img * img_size, 3))

    to_plot = np.square(n_img)
    for pos, img in zip(grid_jv, imgs[0:to_plot]):
        h_range = int(np.floor(pos[0] * (n_img - 1) * img_size))
        w_range = int(np.floor(pos[1] * (n_img - 1) * img_size))
        out[h_range : h_range + img_size, w_range : w_range + img_size] = img.reshape(
            (32, 32, 3)
        )

    im = image.array_to_img(out)
    im.save(out_name, quality=100)


def image_grid2(images, Z, ny=25, nx=40, out_name="grid2.png"):
    import rasterfairy
    from PIL import Image

    # assign to grid
    grid_assignment = rasterfairy.transformPointCloud2D(Z, target=(nx, ny))
    print(grid_assignment)

    tile_width = 32
    tile_height = 32

    full_width = tile_width * nx
    full_height = tile_height * ny
    # aspect_ratio = float(tile_width) / tile_height

    grid_image = Image.new("RGB", (full_width, full_height))

    for img, grid_pos in zip(images, grid_assignment[0]):
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        print(idx_x, idx_y, x, y)

        tile = Image.fromarray(img.reshape((32, 32, 3)).astype("uint8"), "RGB")
        grid_image.paste(tile, box=(int(x), int(y), int(x) + 32, int(y) + 32))

    grid_image.save(out_name)


def plot_rnx_gnn(score_dir, out_name="rnx_gnn.png"):
    import json
    from collections import defaultdict

    score1 = f"{score_dir}/score-d2-m0.5.json"
    score_umap = f"{score_dir}/score-umap.json"
    score_catsne = f"{score_dir}/score-catsne.json"
    all_scores = defaultdict(dict)

    with open(score1, "r") as f1:
        s1 = json.load(f1)
        all_scores.update(s1)

    with open(score_umap, "r") as f2:
        s2 = json.load(f2)
        all_scores.update(s2)

    with open(score_catsne, "r") as f3:
        s3 = json.load(f3)
        all_scores.update(s3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    correct_name = lambda name: {"auc_knn_gain": "auc_knn"}.get(name, name)

    for metric_name, ax in zip(["rnx", "knn_gain"], axes.ravel()):
        for method in ["tsne", "hc-tsne", "umap", "catsne"]:
            scores = all_scores[f"{method}_train"][metric_name]
            auc = all_scores[f"{method}_train"][correct_name(f"auc_{metric_name}")]
            label = f"{method} auc={auc:.3f}"
            ax.semilogx(scores, label=label)
        ax.legend()
        ax.set_title(metric_name.upper())
    fig.savefig(out_name)
