import random
import argparse
import time
import math
import itertools
import numpy as np
import networkx as nx
import skimage.transform as sktrans
from matplotlib.cm import tab20, tab10
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, ConnectionPatch
from matplotlib.lines import Line2D
import cliquematch


#
from rts_align import construct_graph_2d
from rts_align import KabschEstimate

#


def rigid_form(pts, theta, d):
    rotmat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return np.matmul(pts, rotmat) + d


def show_points(Q, K, Q_corr, K_corr, clq, adjmat, filename=None):
    pform = KabschEstimate(K_corr, Q_corr)

    plt.rcParams.update({"font.size": 16})
    fig, axs = plt.subplots(1, 2, figsize=(6.25, 3))
    axs = axs.ravel()
    ax1 = axs[0]
    ax2 = axs[1]

    ncol = len(Q) + len(K)  # 16
    colors = tab10(range(ncol))

    q_pts0 = ax1.scatter(
        Q[:, 0], Q[:, 1], s=60, c="black", marker="x", alpha=0.5
    )
    q_pts = ax1.scatter(
        Q_corr[:, 0], Q_corr[:, 1], s=80, c="red", marker="o", alpha=0.9
    )
    ax1.set_title("$P$")
    k_pts0 = ax2.scatter(
        K[:, 0], K[:, 1], s=60, c="black", marker="x", alpha=0.5
    )
    k_pts = ax2.scatter(
        K_corr[:, 0], K_corr[:, 1], s=80, c="red", marker="o", alpha=0.9
    )
    ax2.set_title("$Q$")

    x1 = ax1.get_xlim()
    x2 = ax2.get_xlim()
    y1 = ax1.get_ylim()
    y2 = ax2.get_ylim()

    ax1.set_xlim([min(x1[0], x2[0], y1[0], y2[0]), max(x1[1], x2[1], y1[1], y2[1])])
    ax2.set_xlim([min(x1[0], x2[0], y1[0], y2[0]), max(x1[1], x2[1], y1[1], y2[1])])
    ax1.set_ylim([min(x1[0], x2[0], y1[0], y2[0]), max(x1[1], x2[1], y1[1], y2[1])])
    ax2.set_ylim([min(x1[0], x2[0], y1[0], y2[0]), max(x1[1], x2[1], y1[1], y2[1])])

    ax1.set_aspect("equal")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.subplots_adjust(
        hspace=0.15, wspace=0.15, top=0.9, bottom=0.05, left=0.05, right=0.95
    )
    if filename:
        fig.savefig(filename, dpi=200)
    else:
        plt.show()


def find_clique(q_pts, k_pts, delta=5, epsilon=0.1):
    res = (
        construct_graph_2d(q_pts, k_pts, delta, epsilon, min_ratio=0.95, max_ratio=1.05)
        != 0
    )
    G = cliquematch.Graph.from_matrix(res)
    c = np.array(G.get_max_clique(), dtype=np.int32) - 1
    qc = q_pts[c // len(k_pts), :]
    kc = k_pts[c % len(k_pts), :]
    return qc, kc, G.to_matrix(), c


def attempt(num_K, num_extra=0, noise_range=1, output_file=None):
    k_pts = np.random.uniform(-100, 100, (num_K, 2))
    rearr = np.random.choice(list(range(len(k_pts))), len(k_pts), False)

    # randomly select R/T/S
    zoom = 1
    theta = np.pi * np.round(np.random.uniform(-1, 1), 2)
    translation = np.random.randint(-75, 75, 2)
    q_pts = rigid_form(k_pts[rearr] * zoom, theta, translation)

    # add extra points
    if num_extra != 0:
        k_extra = np.random.uniform(-100, 100, (num_extra, 2))
        q_extra = np.random.uniform(-100, 100, (num_extra, 2))
        k_pts = np.row_stack([k_pts, k_extra])
        q_pts = np.row_stack([q_pts, q_extra])

    # add noise
    q_pts = q_pts + noise_range * np.random.uniform(-1, 1, (len(q_pts), 2))
    k_pts = k_pts + noise_range * np.random.uniform(-1, 1, (len(k_pts), 2))

    # find corresponding points and visualize
    qc, kc, g, clq = find_clique(q_pts, k_pts, delta=5, epsilon=0.5)
    show_points(q_pts, k_pts, qc, kc, clq, g, filename=output_file)


def main():
    parser = argparse.ArgumentParser(
        "rts-pts-2d", description="visualize alignment of points", add_help=True
    )
    parser.add_argument("-o", "--output-file", default=None, help="save image to file")

    d = parser.parse_args()
    if d.output_file:
        import mpl_texconfig
    attempt(num_K=5, num_extra=10, noise_range=0.01, output_file=d.output_file)


if __name__ == "__main__":
    main()
