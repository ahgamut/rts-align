import random
import argparse
import time
import math
import itertools
import numpy as np
import skimage.transform as sktrans
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, ConnectionPatch
from matplotlib.lines import Line2D
import cliquematch

#
from rts_align import construct_graph_2d
from rts_align import KabschEstimate

### for the animation
FRAMES_PER_MOD = 5
START_OFFSET = 3 * FRAMES_PER_MOD
STOP_OFFSET = 3 * FRAMES_PER_MOD


def rigid_form(pts, theta, d):
    rotmat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return np.matmul(pts, rotmat) + d


def show_points(Q, K, Q_corr, K_corr, order=2, num_steps=15, filename=None):
    s = FRAMES_PER_MOD
    anim_steps = (num_steps) * s + 1  # total length after interpolation

    pform = KabschEstimate(K_corr, Q_corr)
    tmp_form = KabschEstimate(K_corr, K_corr)

    forward = np.linspace(tmp_form.coefs, pform.coefs, anim_steps)
    l0 = len(forward)
    frame_map = np.zeros(2 * l0 + 2 * STOP_OFFSET, dtype=np.int32)
    frame_map[:l0] = np.arange(0, l0)
    frame_map[l0 : l0 + STOP_OFFSET] = l0 - 1
    frame_map[l0 + STOP_OFFSET : (2 * l0 + STOP_OFFSET)] = np.arange(0, l0)[::-1]
    frame_map[(2 * l0 + STOP_OFFSET) :] = 0
    l = len(frame_map)

    plt.rcParams.update({"font.size": 16})
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(6, 4.5),
        sharex=True,
        sharey=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    axs = axs.ravel()

    q_pts, *_ = axs[0].plot(Q[:, 0], Q[:, 1], "ko", markersize=6, alpha=0.5)
    k_pts, *_ = axs[1].plot(K[:, 0], K[:, 1], "ko", markersize=6, alpha=0.5)

    inds = np.ravel(np.array(list(itertools.combinations(range(len(Q_corr)), 2))))
    q_lines, *_ = axs[0].plot(
        Q_corr[inds, 0], Q_corr[inds, 1], "b--", alpha=0.7, linewidth=0.7
    )
    k_lines, *_ = axs[1].plot(
        K_corr[inds, 0], K_corr[inds, 1], "b--", alpha=0.7, linewidth=0.7
    )

    pats = []
    for i in range(0, len(Q_corr)):
        con = ConnectionPatch(
            xyA=K_corr[i],
            xyB=Q_corr[i],
            coordsA="data",
            coordsB="data",
            axesA=axs[1],
            axesB=axs[0],
            color="red",
        )
        axs[1].add_artist(con)
        pats.append(con)

    axs[0].set_aspect("equal")
    axs[1].set_aspect("equal")

    new_K = np.zeros((len(K), 2))
    new_Kc0 = np.zeros((len(K_corr), 2))
    K_inds = K_corr[inds]
    new_Kc = np.zeros((len(K_inds), 2))

    def update_frame(n):
        if n >= START_OFFSET and n - START_OFFSET < l:
            frame = n - START_OFFSET
            tmp_form.coefs = forward[frame_map[frame]]
            for i in range(len(K)):
                new_K[i] = tmp_form(K[i])
            for i in range(len(K_corr)):
                new_Kc0[i] = tmp_form(K_corr[i])
            for i in range(len(K_inds)):
                new_Kc[i] = tmp_form(K_inds[i])

            k_pts.set_data(new_K[:, 0], new_K[:, 1])
            k_lines.set_data(new_Kc[:, 0], new_Kc[:, 1])

            for i in range(0, len(Q_corr)):
                pats[i].remove()
                con = ConnectionPatch(
                    xyA=new_Kc0[i],
                    xyB=Q_corr[i],
                    coordsA="data",
                    coordsB="data",
                    axesA=axs[1],
                    axesB=axs[0],
                    color="red",
                )
                axs[1].add_artist(con)
                pats[i] = con

        res = pats + [k_pts, k_lines]
        return res

    ani = animation.FuncAnimation(
        fig,
        update_frame,
        START_OFFSET + l + STOP_OFFSET,
        interval=30,
        blit=False,
        repeat=True,
    )
    plt.show()
    if filename:
        if filename.endswith(".mp4"):
            ffwriter = animation.FFMpegWriter(fps=20)
            ani.save(filename, writer=ffwriter)
        else:
            ani.save(filename, writer="imagemagick", fps=20, dpi=200)


def find_clique(q_pts, k_pts, delta=5, epsilon=0.1):
    res = construct_graph_2d(q_pts, k_pts, delta) != 0
    G = cliquematch.Graph.from_matrix(res)
    c = np.array(G.get_max_clique(), dtype=np.int32) - 1
    qc = q_pts[c // len(k_pts), :]
    kc = k_pts[c % len(k_pts), :]
    return qc, kc


def attempt(num_K, num_extra=0, noise_range=1, gif_file=None):
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
    qc, kc = find_clique(q_pts, k_pts, delta=0.1, epsilon=0.1)
    show_points(q_pts, k_pts, qc, kc, num_steps=12, filename=gif_file)


def main():
    parser = argparse.ArgumentParser(
        "rts-anim-points", description="visualize alignment of points", add_help=True
    )
    parser.add_argument(
        "-k", "--num-K", default=3, help="number of points that correspond", type=int
    )
    parser.add_argument(
        "-z",
        "--num-extra",
        default=0,
        help="additional points for randomness",
        type=int,
    )
    parser.add_argument(
        "-a", "--noise-add", default=0.01, help="add some uniform noise", type=float
    )
    parser.add_argument(
        "-o", "--output-gif", default=None, help="save animation to GIF file"
    )

    d = parser.parse_args()
    attempt(d.num_K, d.num_extra, d.noise_add, d.output_gif)


if __name__ == "__main__":
    main()
