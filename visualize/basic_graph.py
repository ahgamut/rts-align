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
from rts_align import construct_graph
from rts_align import KabschEstimate

#


def rigid_form(pts, theta, d):
    rotmat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return np.matmul(pts, rotmat) + d


def show_points(Q, K, Q_corr, K_corr, adjmat, filename=None):
    pform = KabschEstimate(K_corr, Q_corr)

    plt.rcParams.update({"font.size": 16})
    fig = plt.figure()
    spec = GridSpec(4, 6, figure=fig)

    ax1 = fig.add_subplot(spec[:2, :2])
    ax2 = fig.add_subplot(spec[2:, :2])
    ax3 = fig.add_subplot(spec[:, 2:])

    ncol = len(Q) + len(K)  # 16
    colors = tab10(range(ncol))
    ax3.set_xlim([0, 100])
    ax3.set_ylim([0, 100])
    icon_size = (ax3.get_xlim()[1] - ax3.get_xlim()[0]) * 0.05

    q_pts = ax1.scatter(
        Q[:, 0], Q[:, 1], s=50, c=colors[: len(Q), :], marker="o", alpha=0.5
    )
    ax1.set_title("$P$")
    ax1.set_xticks([])
    ax1.set_yticks([])
    k_pts = ax2.scatter(
        K[:, 0], K[:, 1], s=50, c=colors[len(Q) :, :], marker="o", alpha=0.5
    )
    ax2.set_title("$Q$")
    ax2.set_xticks([])
    ax2.set_yticks([])

    G = nx.Graph(adjmat)
    print(G)
    pos = nx.spring_layout(G, k=20, scale=35, center=(50, 50))
    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax3,
        arrows=True,
        arrowstyle="-",
        min_source_margin=15,
        min_target_margin=15,
    )

    for n in G.nodes:
        ax3.add_patch(Circle(pos[n], icon_size, color="gray", alpha=0.15))
        qcol = colors[n // len(K)]
        kcol = colors[len(Q) + (n % len(K))]
        ax3.add_patch(
            Circle(pos[n] + icon_size / 3, icon_size / 4, color=qcol, alpha=0.5)
        )
        ax3.add_patch(
            Circle(pos[n] - icon_size / 3, icon_size / 4, color=kcol, alpha=0.5)
        )

    ax3.set_title("$G_{P, Q}$")
    ax3.set_aspect("equal")
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.tight_layout()
    if filename:
        fig.savefig(filename, dpi=200)
    else:
        plt.show()


def find_clique(q_pts, k_pts, delta=5, epsilon=0.1):
    res = construct_graph(q_pts, k_pts, delta, epsilon) != 0
    G = cliquematch.Graph.from_matrix(res)
    c = np.array(G.get_max_clique(), dtype=np.int32) - 1
    qc = q_pts[c // len(k_pts), :]
    kc = k_pts[c % len(k_pts), :]
    return qc, kc, G.to_matrix()


def attempt(num_K, num_extra=0, noise_range=1, output_file=None):
    k_pts = np.random.uniform(-100, 100, (num_K, 2))
    rearr = np.random.choice(list(range(len(k_pts))), len(k_pts), False)

    # randomly select R/T/S
    zoom = np.random.uniform(1.8, 3.2)
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
    qc, kc, g = find_clique(q_pts, k_pts, delta=30, epsilon=1.5)
    show_points(q_pts, k_pts, qc, kc, g, filename=output_file)


def main():
    parser = argparse.ArgumentParser(
        "rts-basic-graph", description="visualize alignment of points", add_help=True
    )
    parser.add_argument("-o", "--output-file", default=None, help="save image to file")

    d = parser.parse_args()
    if d.output_file:
        import mpl_texconfig
    attempt(num_K=3, num_extra=1, noise_range=0.01, output_file=d.output_file)


if __name__ == "__main__":
    main()
