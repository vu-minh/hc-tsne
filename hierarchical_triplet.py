# Hierarchical constraints implemented as tree-like structure.

import itertools
import numpy as np
from anytree import LevelOrderIter, LevelGroupOrderIter


# not used in V1
def _triplet_loss_all(
    anchor_indices, positive_indices, negative_indices, Y, margin=0.0
):
    loss = 0.0
    grad = np.zeros_like(Y, dtype=np.float32)

    n_anchor = len(anchor_indices)
    n_pos = len(positive_indices)
    n_neg = len(negative_indices)
    if n_anchor == 0 or n_pos == 0 or n_neg == 0:
        return loss, grad

    positive_centroid = Y[positive_indices].mean()
    negative_centroid = Y[negative_indices].mean()

    dist_anchor_pos = np.sum((Y[anchor_indices] - positive_centroid) ** 2, axis=1)
    dist_anchor_neg = np.sum((Y[anchor_indices] - negative_centroid) ** 2, axis=1)
    loss_per_item = dist_anchor_pos - (1.0 - margin) * dist_anchor_neg

    violated = (loss_per_item > 0).reshape(-1)

    if np.any(violated):
        loss = loss_per_item[violated].sum()

        # update gradient for the violated anchor points
        violated_anchor_indices = np.array(anchor_indices)[violated]
        change = (
            margin * Y[violated_anchor_indices]
            + (1.0 - margin) * negative_centroid
            - positive_centroid
        )
        grad[violated_anchor_indices] += 2.0 * change

        # TODO Update margin

        # update gradient for points in list `positive_indices`
        for positive_index in positive_indices:
            change1 = np.sum(
                Y[violated_anchor_indices] - Y[positive_index].reshape(1, -1), axis=0
            )
            grad[positive_index] += -2.0 * change1 / n_pos

        # update gradient for poitns in list `negative_indices`
        for negative_index in negative_indices:
            change2 = np.sum(
                Y[violated_anchor_indices] - Y[negative_index].reshape(1, -1), axis=0
            )
            grad[negative_index] += 2.0 * change2 / n_neg

    # TODO ValueError Regularization too large.-> small alpha?
    loss /= n_anchor
    grad /= n_anchor
    # print(loss, np.linalg.norm(grad))
    return loss, grad


# not used in V1
def _triplet_loss_anchor_vs_all(
    anchor_index, positive_indices, negative_indices, Y, margin=0.0
):
    """Triplet loss among an `anchor_index`, all `possitive_indices`
    and all `negative_indices`.
    """
    N1 = len(positive_indices)
    N2 = len(negative_indices)
    assert (N1 > 0) and (N2 > 0)

    anchor = Y[anchor_index]
    positive_centroid = Y[positive_indices].mean()
    negative_centroid = Y[negative_indices].mean()
    loss = (
        np.linalg.norm(anchor - positive_centroid) ** 2
        - np.linalg.norm(anchor - negative_centroid) ** 2
    )

    grad = np.zeros_like(Y, dtype=np.float32)

    if loss <= 0:
        return 0.0, grad

    grad[anchor_index] += 2 * (negative_centroid - positive_centroid)
    grad[positive_indices] += -2 / N1 * (anchor.reshape(1, -1) - Y[positive_indices])
    grad[negative_indices] += 2 / N2 * (anchor.reshape(1, -1) - Y[negative_indices])

    return loss, grad


def _triplet_loss(anchor_indices, positive_point, negative_point, Y, margin=0.0):
    """Triplet loss among (anchor, pos, neg).
    This loss is calculated for all anchors in `anchor_indices`.
    """
    N_anchors = len(anchor_indices)

    dist_anchor_pos = np.sum((Y[anchor_indices] - positive_point) ** 2, axis=1)
    dist_anchor_neg = np.sum((Y[anchor_indices] - negative_point) ** 2, axis=1)
    loss_per_item = dist_anchor_pos - (1.0 - margin) * dist_anchor_neg

    violated = (loss_per_item > 0).reshape(-1)
    loss_per_node = 0.0
    grad_per_node = np.zeros_like(Y, dtype=np.float32)

    if np.any(violated):
        loss_per_node = loss_per_item[violated].sum() / N_anchors

        # update gradient for the violated anchor points
        violated_anchor_indices = np.array(anchor_indices)[violated]
        change = (
            margin * Y[violated_anchor_indices]
            + (1.0 - margin) * negative_point
            - positive_point
        )
        grad_per_node[violated_anchor_indices] += 2.0 * change / N_anchors

    return loss_per_node, grad_per_node


def _kmean_like(point_indices, centroid, Y):
    """KMeans-like loss: sum squared distances
    from all points in `point_indices` to `centroid`.
    """
    loss_per_node = np.einsum("ij,ij->i", Y[point_indices], centroid).mean()

    grad_per_node = np.zeros_like(Y, dtype=np.float32)
    grad_per_node[point_indices] = (
        2.0 * (Y[point_indices] - centroid) / len(point_indices)
    )

    return loss_per_node, grad_per_node


def hierarchical_triplet_loss(
    Y, tree, margin=0.5, weights=(1.0, 1.0, 0.0), weight_by_level=False, vs_all=False
):
    """Hierarchical triplet loss, based on the input `tree`-structure.
    Margin is used as `m = margin * d_{ik}`: distance anchor-negative.
    """
    loss = 0.0
    grad = np.zeros_like(Y, dtype=np.float32)

    # In general, there are 3 terms that contribute for the loss:
    # Term1: makes the points be closer to their centroid than to their parent's.
    # Term2: makes the points be closer to their centroid than to their siblings'.
    # Term3: KMeans-like loss in a group
    w1, w2, w3 = weights

    # iterate the tree and construct the loss term
    # loss(x) = max(0, d(x, positive) - d(x, negative) + m)
    # m = margin * d(x, negative)

    # First, update the centroid for each node
    for node in LevelOrderIter(tree):
        node.update_centroid(Y)

    # Iterate tree to calculate the loss term for all items in each node
    # The 2 rules convert the tree structure into constraints on the distance of points:

    # Term1: a point `y_i` should be closer to its group's centroid
    #   than to the centroid of its parent's centroid
    if w1 > 0:
        for node in LevelOrderIter(tree):
            node_weight = max(1, node.level) if weight_by_level else 1

            # the first node is always root node, ignore it
            if node.parent is None or len(node.items) == 0:
                continue

            if vs_all:
                loss_per_node, grad_per_node = _triplet_loss_all(
                    anchor_indices=node.items,
                    positive_indices=node.items,
                    negative_indices=node.parent.items,
                    Y=Y,
                    margin=margin,
                )
            else:
                loss_per_node, grad_per_node = _triplet_loss(
                    anchor_indices=node.items,
                    positive_point=node.centroid,
                    negative_point=node.parent.centroid,
                    Y=Y,
                    margin=margin,
                )

            loss += w1 * node_weight * loss_per_node
            grad += w1 * node_weight * grad_per_node

    # Term2: a point `y_i` should be closer to its group's centroid
    #   than to its sibling's centroid
    if w2 > 0:
        for sibling_nodes in LevelGroupOrderIter(tree):
            if len(sibling_nodes) == 1:
                continue

            # loop over all pair of sibling nodes
            for node1, node2 in itertools.permutations(sibling_nodes, 2):
                assert node1.level == node2.level
                node_weight = max(1, node.level) if weight_by_level else 1

                if vs_all:
                    loss_per_node, grad_per_node = _triplet_loss_all(
                        anchor_indices=node1.items,
                        positive_indices=node1.items,
                        negative_indices=node2.items,
                        Y=Y,
                        margin=margin,
                    )
                else:
                    loss_per_node, grad_per_node = _triplet_loss(
                        anchor_indices=node1.items,
                        positive_point=node1.centroid,
                        negative_point=node2.centroid,
                        Y=Y,
                        margin=margin,
                    )
                loss += w2 * node_weight * loss_per_node
                grad += w2 * node_weight * grad_per_node

    # Add new K-Means-like loss term to force points in the same cluster close together
    if w3 > 0:
        for node in LevelOrderIter(tree):
            node_weight = max(1, node.level) if weight_by_level else 1
            loss_per_node, grad_per_node = _kmean_like(node.items, node.centroid, Y)
            loss += w3 * node_weight * loss_per_node
            grad += w3 * node_weight * grad_per_node

    if loss > 1e9:
        raise ValueError("Regularization term too large!")

    return loss, grad
