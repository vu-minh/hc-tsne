# openTSNE with constraints

from functools import partial

import numpy as np
from openTSNE import _tsne
from openTSNE import TSNE  # TSNEEmbedding
from openTSNE.quad_tree import QuadTree
from openTSNE.callbacks import ErrorLogger
from hierarchical_triplet import hierarchical_triplet_loss

EPSILON = np.finfo(np.float64).eps


def tsne(X, initialization="pca", **tsne_kwargs):
    """Original openTSNE"""
    tsne = TSNE(
        initialization=initialization, negative_gradient_method="bh", **tsne_kwargs,
    )
    return tsne.fit(X)


def hc_tsne(
    X,
    initialization,
    tree,
    alpha=1e-3,
    weights=(0.5, 0.5, 0.0),
    margin=0.5,
    loss_logger=None,
    **tsne_kwargs,
):
    """Run openTSNE with custom `negative_gradient_method`, in which the
    hierarchical constraints are encoded in a regularization term.

    Args:
        X: ndarray (N, D)
        initialization: initialization embedding in 2D, (N, 2)
        tree: hierarchical constraints represented in tree form (using anytree lib)
        alpha: contribution of regularization term in the new objective function
        weights: weights of different elements in the regularization
        margin: margin in the triplet loss.
            The real margin m is calculated as `margin * dist(anchor, negative)`
        loss_logger: logger object (containing a dict) to store loss at each iter.
        **tsne_kwargs: openTSNE params

    Returns:
        Z: new embedding model, can be used as (N, 2) array,
            or tsne object for embedding new datapoints.
    """
    # from the tree-like constraints, create a regularization term by
    #   using the defined hierarchical triplet loss.
    tree_regularizer = partial(
        hierarchical_triplet_loss, tree=tree, margin=margin, weights=weights
    )

    # run openTSNE with custom negative gradient function
    tsne = TSNE(
        initialization=initialization,
        callbacks=ErrorLogger(),  # use this to evaluate kl_loss at every 10 iterations
        negative_gradient_method=partial(
            my_kl_divergence_bh,
            list_regularizers=[(alpha, tree_regularizer)],
            logger=loss_logger,
        ),
        **tsne_kwargs,
    )

    Z = tsne.fit(X)

    # now clear the regularizers from tsne object so we will not use them for embedding
    # new samples (of test set)
    Z.gradient_descent_params["negative_gradient_method"] = "bh"
    return Z


def my_kl_divergence_bh(
    embedding,
    P,
    dof,
    bh_params,
    reference_embedding=None,
    should_eval_error=False,
    n_jobs=1,
    list_regularizers=[],
    logger=None,
    **_,
):
    gradient = np.zeros_like(embedding, dtype=np.float64, order="C")

    # In the event that we wish to embed new points into an existing embedding
    # using simple optimization, we compute optimize the new embedding points
    # w.r.t. the existing embedding. Otherwise, we want to optimize the
    # embedding w.r.t. itself. We've also got to make sure that the points'
    # interactions don't interfere with each other
    pairwise_normalization = reference_embedding is None
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute negative gradient
    tree = QuadTree(reference_embedding)
    sum_Q = _tsne.estimate_negative_gradient_bh(
        tree,
        embedding,
        gradient,
        **bh_params,
        dof=dof,
        num_threads=n_jobs,
        pairwise_normalization=pairwise_normalization,
    )
    del tree

    # Compute positive gradient
    sum_P, kl_divergence_ = _tsne.estimate_positive_gradient_nn(
        P.indices,
        P.indptr,
        P.data,
        embedding,
        reference_embedding,
        gradient,
        dof,
        num_threads=n_jobs,
        should_eval_error=should_eval_error,
    )

    # Compute loss and gradient of group constraints (regularization)
    # But do not do it in exaggeration state
    regu_loss = 0.0
    if np.sum(P) <= 1.0 + 1e-6:
        regu_loss, regu_grad = loss_and_gradient_of_group_constraints(
            list_regularizers, Y=np.array(embedding, dtype=np.float32), logger=logger
        )
        gradient += regu_grad

    # Computing positive gradients summed up only unnormalized q_ijs, so we
    # have to include normalziation term separately
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)
        logger and logger.log("kl_loss", kl_divergence_)
        kl_divergence_ += regu_loss
        logger and logger.log("new_loss", kl_divergence_)

    return kl_divergence_, gradient


def loss_and_gradient_of_group_constraints(
    list_regularizers, Y, normalized=False, clip_grad=False, logger=None
):
    total_loss = 0.0
    total_grad = np.zeros_like(Y, dtype=np.float64, order="C")

    if normalized:
        norm_Y = np.linalg.norm(Y, ord=2, axis=1, keepdims=True)
        Y = np.divide(Y, norm_Y)

    for alpha, regularizer in list_regularizers:
        loss, grad = regularizer(Y=Y)
        total_loss += alpha * loss
        total_grad += alpha * grad
        logger and logger.log("htriplet_loss", loss)

    # clip too large graident values
    if clip_grad:
        total_grad = clip(total_grad)

    if normalized:
        N, D = Y.shape
        for i in range(N):
            y_i = Y[i].reshape(1, -1)
            norm_y_i = norm_Y[i]
            jacobian = (np.identity(D) / norm_y_i) - (y_i.T @ y_i) / (norm_y_i ** 3)
            dYnorm_dY = np.sum(jacobian, axis=0)
            total_grad[i] *= dYnorm_dY

    return total_loss, total_grad


def clip(data, threshold=3.5):
    # Ref: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    # consider to use openTSNE max_step_norm=5
    coeff = 0.6745
    shape = data.shape
    data = data.reshape(-1)

    diff_median = data - np.median(data)

    # median absolute deviation
    mad = np.median(np.abs(diff_median))

    # modified Z-score
    z_score = coeff * (diff_median) / mad if mad != 0 else 0.0

    outliers = np.abs(z_score) > threshold
    data[outliers] = np.median(data) + (
        np.sign(diff_median[outliers]) * threshold / coeff * mad
    )

    return data.reshape(shape)


class PlotCallback:
    ...
