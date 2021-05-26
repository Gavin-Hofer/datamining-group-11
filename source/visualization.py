# Functions for visualizing the results of the clustering
from typing import  Any
from nptyping import NDArray
from sklearn.metrics import pairwise
from matplotlib import pyplot as plt
import numpy as np


# %% plot_similarity_matrix
def plot_similarity_matrix(
    points: NDArray[(Any, Any), float],
    labels: NDArray[(Any,), int]
) -> plt.Figure:
    """Plot a 2D heatmap of the similarity matrix created from the
    points, sorted by cluster
    """
    S = _similiarity_matrix(points, labels)
    spec = {"width_ratios": [1, 1, .05], "top": .9}
    fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(8, 3), gridspec_kw=spec)
    ax1.imshow(S, cmap='jet', vmin=-1, vmax=1)
    aximg = ax2.imshow(S, cmap='jet', vmin=-1, vmax=1)
    _plot_cluster_outlines(ax2, labels)
    fig.colorbar(aximg, cax=cax)
    return fig


def _similiarity_matrix(points, labels):
    """Create a similarity matrix of to visualize the clusters"""
    points_sorted = _sort_by_label(points, labels)
    return pairwise.cosine_similarity(points_sorted, points_sorted)


def _sort_by_label(points, labels):
    """Sort the points by label"""
    idx = np.argsort(labels)
    return points[idx]


def _plot_cluster_outlines(ax, labels):
    """Plot boxes around each of the clusters"""
    labels = np.sort(labels)
    for lab in set(labels):
        idx1 = np.searchsorted(labels, lab, side='left')
        idx2 = np.searchsorted(labels, lab, side='right') - 1
        _plot_box(ax, idx1, idx2)


def _plot_box(ax, idx1, idx2):
    """Plot a box along a diagonal axis of matrix from idx1 to idx2"""
    x = (idx1, idx2, idx2, idx1, idx1)
    y = (idx1, idx1, idx2, idx2, idx1)
    ax.plot(x, y, 'k-', linewidth=1)
