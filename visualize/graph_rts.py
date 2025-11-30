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
from matplotlib.patches import Circle, Arc
from matplotlib.lines import Line2D
import cliquematch

#
from rts_align import construct_graph_2d
from rts_align import KabschEstimate

#


def rigid_form(pts, theta, d):
    rotmat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return np.matmul(pts, rotmat) + d


def findpt(l1, l2):
    t1 = np.arctan2(l1[1, 1] - l1[0, 1], l1[1, 0] - l1[0, 0]) * 180 / np.pi
    t2 = np.arctan2(l2[1, 1] - l2[0, 1], l2[1, 0] - l2[0, 0]) * 180 / np.pi
    t1, t2 = np.abs(t1), np.abs(t2)
    a1, a2 = min(t1, t2), max(t1, t2)
    dmat = np.zeros((2, 2))
    dmat[0, 0] = np.linalg.norm(l1[0] - l2[0])
    dmat[0, 1] = np.linalg.norm(l1[0] - l2[1])
    dmat[1, 0] = np.linalg.norm(l1[1] - l2[0])
    dmat[1, 1] = np.linalg.norm(l1[1] - l2[1])
    ptl = np.argmin(dmat) // 2
    col = "red"
    start = 0
    if ptl == 1:
        a1, a2 = -a2, -a1
        col = "blue"
        start = 180
    print(dmat)
    return Arc(
        l1[ptl, :],
        5,
        5,
        angle=start,
        theta1=a1,
        theta2=a2,
        color="blue",
        linewidth=3,
    )


def show_points(Q, K, Q_corr, K_corr, clq, adjmat, filename=None):
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
        Q[:, 0], Q[:, 1], s=80, c=colors[: len(Q), :], marker="o", alpha=0.5
    )
    ax1.set_title("$P$")
    k_pts = ax2.scatter(
        K[:, 0], K[:, 1], s=80, c=colors[len(Q) :, :], marker="o", alpha=0.5
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
    ax2.set_aspect("equal")

    inds = np.ravel(np.array(list(itertools.combinations(range(len(Q_corr)), 2))))
    q_lines, *_ = ax1.plot(
        Q_corr[inds, 0], Q_corr[inds, 1], "k--", alpha=0.7, linewidth=0.7
    )
    k_lines, *_ = ax2.plot(
        K_corr[inds, 0], K_corr[inds, 1], "k--", alpha=0.7, linewidth=0.7
    )

    qld = q_lines.get_xydata()
    kld = q_lines.get_xydata()

    for i in range(0, len(qld), 2):
        lqi = qld[i : i + 2]
        lki = kld[i : i + 2]
        for j in range(i + 2, len(qld), 2):
            lqj = qld[j : j + 2]
            lkj = kld[j : j + 2]
            # ax1.add_patch(findpt(lqj, lqi))

    G = nx.Graph(adjmat)
    ew = lambda e: 1.5 if (e[0] in clq and e[1] in clq) else 1
    ec = lambda e: "#ff0000b0" if (e[0] in clq and e[1] in clq) else "#000000"
    options = {
        "width": [ew(x) for x in G.edges],
        "edge_color": [ec(x) for x in G.edges],
    }
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
        **options,
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
    res = construct_graph_2d(q_pts, k_pts, delta, epsilon, min_ratio=0.2, max_ratio=5) != 0
    G = cliquematch.Graph.from_matrix(res)
    c = np.array(G.get_max_clique(), dtype=np.int32) - 1
    qc = q_pts[c // len(k_pts), :]
    kc = k_pts[c % len(k_pts), :]
    return qc, kc, G.to_matrix(), c


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
    qc, kc, g, clq = find_clique(q_pts, k_pts, delta=5, epsilon=0.5)
    show_points(q_pts, k_pts, qc, kc, clq, g, filename=output_file)


def main():
    parser = argparse.ArgumentParser(
        "rts-graph-2d", description="visualize alignment of points", add_help=True
    )
    parser.add_argument("-o", "--output-file", default=None, help="save image to file")

    d = parser.parse_args()
    if d.output_file:
        import mpl_texconfig
    attempt(num_K=3, num_extra=1, noise_range=0.01, output_file=d.output_file)


if __name__ == "__main__":
    main()
