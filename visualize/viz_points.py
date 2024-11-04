import random
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
from rts_align import construct_graph

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

    pform = sktrans.SimilarityTransform()
    pform.estimate(K_corr, Q_corr)

    tmp_form = sktrans.SimilarityTransform()
    tmp_form.estimate(K_corr, K_corr)

    forward = np.linspace(tmp_form.params, pform.params, anim_steps)
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

    fig.suptitle("rotation, translation, AND SCALE?")

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

    def update_frame(n):
        if n >= START_OFFSET and n - START_OFFSET < l:
            frame = n - START_OFFSET
            tmp_form.params = forward[frame_map[frame]]
            new_K = tmp_form(K)
            new_Kc0 = tmp_form(K_corr)
            new_Kc = tmp_form(K_corr[inds])

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
        ani.save(filename, writer="imagemagick", fps=30, dpi=200)


def find_clique(q_pts, k_pts, delta=0.01):
    res = construct_graph(q_pts, k_pts, delta) != 0
    G = cliquematch.Graph.from_matrix(res)
    c = np.array(G.get_max_clique(), dtype=np.int32) - 1
    qc = q_pts[c // len(k_pts), :]
    kc = k_pts[c % len(k_pts), :]
    show_points(q_pts, k_pts, qc, kc, num_steps=12, filename="./align3_anim.gif")


def attempt():
    k_pts = np.random.uniform(-100, 100, (3, 2))
    rearr = np.random.choice(list(range(len(k_pts))), len(k_pts), False)

    zoom = np.random.uniform(1.8, 3.2)
    theta = np.pi * np.round(np.random.uniform(-1, 1), 2)
    translation = np.random.randint(-75, 75, 2)
    error = 1.0 * np.random.uniform(-1, 1, (len(k_pts), 2))
    q_pts = rigid_form(k_pts[rearr] * zoom, theta, translation) + error
    find_clique(q_pts, k_pts, delta=0.2)


def main():
    # np.random.seed(42)
    attempt()


if __name__ == "__main__":
    main()
