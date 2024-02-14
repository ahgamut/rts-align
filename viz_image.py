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
from matplotlib.colors import LinearSegmentedColormap

COLORDICT = {
    "red": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    "blue": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    "green": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    "alpha": [[0.0, 1.0, 1.0], [0.95, 0.0, 0.0], [1.0, 0.0, 0.0]],
}
COLORMAP = LinearSegmentedColormap("zzazz", segmentdata=COLORDICT, N=256)

FRAMES_PER_MOD = 4
START_OFFSET = 5 * FRAMES_PER_MOD
STOP_OFFSET = 5 * FRAMES_PER_MOD


def show_setup(Q_img, K_img, Q, K, filename=None):
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(6, 4.5),
        sharex=True,
        sharey=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    axs = axs.ravel()
    lim = max(max(Q_img.shape), max(K_img.shape))
    axs[0].set_xlim((0, lim))
    axs[0].set_ylim((lim, 0))

    q_pts, *_ = axs[0].plot(Q[:, 0], Q[:, 1], "ro", markersize=3, alpha=0.9)
    axs[0].imshow(Q_img, cmap="Greys_r")
    axs[0].set_title("Q")

    k_pts, *_ = axs[1].plot(K[:, 0], K[:, 1], "ro", markersize=3, alpha=0.9)
    axs[1].imshow(K_img, cmap="Greys_r")
    axs[1].set_title("K")

    fig.suptitle("are we plotting correctly?")
    plt.show()
    if filename:
        fig.savefig(filename, dpi=200)


def show_anim(Q_img, K_img, Q, K, Q_corr, K_corr, order=2, num_steps=15, filename=None):
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
    lim = max(max(Q_img.shape), max(K_img.shape))

    images = []
    for i in range(l0):
        tmp_form.params = forward[i, :]
        warped_k = sktrans.warp(
            K_img,
            inverse_map=tmp_form.inverse,
            output_shape=(lim, lim),
            mode="constant",
            cval=1,
        )
        images.append(warped_k)

    plt.rcParams.update({"font.size": 16})
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(8, 4.5),
        sharex=True,
        sharey=True,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    axs = axs.ravel()
    axs[0].set_xlim((0, lim))
    axs[0].set_ylim((lim, 0))

    q_back = axs[0].imshow(Q_img, cmap="Greys_r")
    k_back = axs[1].imshow(images[0], cmap="Greys_r")
    q_form = axs[2].imshow(Q_img, cmap="Greys_r")
    k_form = axs[2].imshow(images[0], cmap=COLORMAP)

    q_pts, *_ = axs[0].plot(Q[:, 0], Q[:, 1], "ro", markersize=3, alpha=0.7)
    k_pts, *_ = axs[1].plot(K[:, 0], K[:, 1], "ro", markersize=3, alpha=0.7)
    qf_pts, *_ = axs[2].plot(Q_corr[:, 0], Q_corr[:, 1], "ro", markersize=3, alpha=0.5)
    kf_pts, *_ = axs[2].plot(K_corr[:, 0], K_corr[:, 1], "ro", markersize=3, alpha=0.5)

    # fig.suptitle("rotation, translation, and scale")
    axs[0].set_title("Q")
    axs[1].set_title("K")
    axs[2].set_title("overlay")
    corr_mix = []
    for i in range(len(Q_corr)):
        corr_mix.append((Q_corr[i, 0], K_corr[i, 0]))
        corr_mix.append((Q_corr[i, 1], K_corr[i, 1]))
        corr_mix.append("b--")

    inds = np.ravel(np.array(list(itertools.combinations(range(len(Q_corr)), 2))))
    q_lines, *_ = axs[0].plot(
        Q_corr[inds, 0], Q_corr[inds, 1], "b--", alpha=1, linewidth=0.5
    )
    k_lines, *_ = axs[1].plot(
        K_corr[inds, 0], K_corr[inds, 1], "b--", alpha=1, linewidth=0.5
    )
    map_lines = axs[2].plot(
        *corr_mix,
        linewidth=0.5,
        alpha=1,
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

    def update_frame(n):
        if n >= START_OFFSET and n - START_OFFSET < l:
            frame = n - START_OFFSET
            tmp_form.params = forward[frame_map[frame]]
            new_K_img = images[frame_map[frame]]
            new_K = tmp_form(K)
            new_Kc0 = tmp_form(K_corr)
            new_Kc = tmp_form(K_corr[inds])

            k_form.set_data(new_K_img)
            k_form.set_alpha(1)
            k_back.set_data(new_K_img)
            k_pts.set_data(new_K[:, 0], new_K[:, 1])
            kf_pts.set_data(new_Kc0[:, 0], new_Kc0[:, 1])
            kf_pts.set_alpha(1)
            k_lines.set_data(new_Kc[:, 0], new_Kc[:, 1])
            k_lines.set_alpha(1)

            q_lines.set_alpha(1)
            qf_pts.set_alpha(1)

            for i in range(0, len(Q_corr)):
                pats[i].set_alpha(1)
                pats[i].xy1 = new_Kc0[i]
                pats[i].xy2 = Q_corr[i]
                map_lines[i].set_alpha(1)
                map_lines[i].set_data(
                    [Q_corr[i, 0], new_Kc0[i, 0]], [Q_corr[i, 1], new_Kc0[i, 1]]
                )
            fct = frame_map[frame]
            if fct != 0:
                fct += 1
            fct = (100 * fct) / len(forward)
            fig.suptitle("Aligning....: {:.2f}%".format(fct))
        else:
            kf_pts.set_alpha(0)
            qf_pts.set_alpha(0)
            k_form.set_alpha(0)
            k_lines.set_alpha(0)
            q_lines.set_alpha(0)
            for i in range(0, len(Q_corr)):
                pats[i].set_alpha(0)
                map_lines[i].set_alpha(0)
            fig.suptitle("")

        res = pats + [k_pts, k_lines, kf_pts]
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
