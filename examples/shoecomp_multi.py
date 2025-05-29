import sys
import os
import argparse
import zipfile
import numpy as np
import skimage.io as skio
import skimage.filters as skfilt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

NAME = "shoecomp-multi"
DESC = "obtain alignments for shoeprint image pairs annotated using shoecomp"

from rts_align import ImagePair
from rts_align import KabschMapping
from rts_align import find_all_cliques

COLORDICT = {
    "red": [[0.0, 0.612, 0.612], [1.0, 0.561, 0.561]],
    "green": [[0.0, 0.149, 0.149], [1.0, 0.149, 0.149]],
    "blue": [[0.0, 0.561, 0.561], [1.0, 0.561, 0.561]],
    "alpha": [[0.0, 0.55, 0.55], [0.25, 0.55, 0.55], [1.0, 0.0, 0.0]],
}
COLORMAP = LinearSegmentedColormap("k_over", segmentdata=COLORDICT, N=256)


def boundify(msk, bounds):
    if len(bounds) > 0:
        yv, xv = bounds[:, 0], bounds[:, 1]
    else:
        yv, xv = np.nonzero(msk[:, :, 0])
    return {"x": (xv.min(), xv.max()), "y": (yv.min(), yv.max())}


def viz_alignment(count, ipair, corr, title="alignment"):
    m = KabschMapping()

    q = ipair.Q_img
    q_mask = ipair.Q_mask
    k = ipair.K_img
    mapped_k = m.align_K_to_Q(ipair.Q_img, ipair.K_raw, corr)
    mapped_k_mask = m.align_K_to_Q(ipair.Q_img, ipair.K_mask, corr)
    ovr_mask = (q_mask * mapped_k_mask) != 0.0

    q = ipair.Q_img * ovr_mask
    mapped_k = mapped_k * ovr_mask + 1.0 * ~ovr_mask

    k_rect = boundify(ipair.K_mask != 0, ipair.K_bounds)
    q_rect = boundify(ipair.Q_mask != 0, ipair.Q_bounds)
    o_rect = boundify(ovr_mask, [])

    fig, axs = plt.subplots(1, 3)
    axs = axs.ravel()

    thr = lambda x: x > 0.8 * skfilt.threshold_otsu(x)

    Q_corr = corr["Q"]
    K_corr = corr["K"]

    axs[0].imshow(q, cmap="Greys_r")
    axs[0].set_title("Q")
    q_pts, *_ = axs[0].plot(Q_corr[:, 0], Q_corr[:, 1], "ro", markersize=3, alpha=0.7)
    axs[0].set_xlim(q_rect["x"])
    axs[0].set_ylim(q_rect["y"][::-1])
    axs[0].axis("off")

    axs[1].imshow(k, cmap="Greys_r")
    axs[1].set_title("K")
    k_pts, *_ = axs[1].plot(K_corr[:, 0], K_corr[:, 1], "ro", markersize=3, alpha=0.7)
    axs[1].set_xlim(k_rect["x"])
    axs[1].set_ylim(k_rect["y"][::-1])
    axs[1].axis("off")

    axs[2].imshow(q, cmap="Greys_r")
    axs[2].imshow(thr(mapped_k[:, :, 0]), cmap=COLORMAP)
    axs[2].set_title("mapped K on Q")
    axs[2].set_xlim(o_rect["x"])
    axs[2].set_ylim(o_rect["y"][::-1])
    axs[2].axis("off")

    # the lines connecting Q and K
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

    fig.suptitle("{} - {}".format(title, count))
    return fig


def save_outputs(output_dir, count, ipair, corr, prefix="align"):
    m = KabschMapping()

    q = ipair.Q_img
    q_mask = ipair.Q_mask
    k = ipair.K_img
    mapped_k = m.align_K_to_Q(ipair.Q_img, ipair.K_raw, corr)
    mapped_k_mask = m.align_K_to_Q(ipair.Q_img, ipair.K_mask, corr)
    ovr_mask = (q_mask * mapped_k_mask) != 0.0

    q = ipair.Q_img * ovr_mask
    mapped_k = mapped_k * ovr_mask
    o_rect = boundify(ovr_mask)

    q_sub = np.uint8(
        255
        * q[o_rect["y"][0] : o_rect["y"][1] + 1, o_rect["x"][0] : o_rect["x"][1] + 1]
    )
    k_sub = np.uint8(
        255
        * mapped_k[
            o_rect["y"][0] : o_rect["y"][1] + 1, o_rect["x"][0] : o_rect["x"][1] + 1
        ]
    )

    q_path = os.path.join(output_dir, f"{prefix}-q-ALIGNED_{count:03d}.png")
    k_path = os.path.join(output_dir, f"{prefix}-k-ALIGNED_{count:03d}.png")

    skio.imsave(q_path, q_sub)
    skio.imsave(k_path, k_sub)


def runner(
    input_zip,
    delta,
    epsilon,
    lower_bound,
    total,
    rescale,
    heuristic,
    visualize,
    output_dir=None,
):
    zfile = zipfile.ZipFile(input_zip, "r")
    ipair = ImagePair(zfile, rescale=rescale)
    bname = os.path.basename(input_zip)
    bbase = os.path.splitext(bname)[0]
    corr_list = find_all_cliques(
        ipair.Q_pts, ipair.K_pts, delta, epsilon, lower_bound, total, heuristic
    )
    num_corr = len(corr_list)
    if num_corr == 0:
        print("zero alignments found")
    else:
        print(num_corr, "alignments found")

    for i, corr in enumerate(corr_list):
        fig = viz_alignment(i, ipair, corr, bname)
        if visualize:
            plt.show()
        if output_dir is not None:
            fig.subplots_adjust(
                hspace=0.15, wspace=0.05, top=0.95, bottom=0.05, left=0.01, right=0.98
            )
            fpath = os.path.join(output_dir, f"{bbase}-alignment_{i:03d}.png")
            fig.savefig(fpath, dpi=200)
            save_outputs(output_dir, i, ipair, corr, prefix=bbase)


def main():
    parser = argparse.ArgumentParser(NAME, description=DESC)
    parser.add_argument(
        "-i",
        "--input-zip",
        required=True,
        help="ZIP file containing shoecomp comparisons",
    )
    parser.add_argument(
        "--epsilon",
        default=0.05,
        type=float,
        help="maximum allowable scaling distortion",
    )
    parser.add_argument(
        "--delta", default=0.05, type=float, help="maximum allowable angular distortion"
    )
    parser.add_argument(
        "-l",
        "--lower-bound",
        default=5,
        type=int,
        help="allow alignments containing at least these many points",
    )
    parser.add_argument(
        "-t",
        "--total",
        default=10,
        type=int,
        help="stop after finding these many alignments",
    )
    parser.add_argument(
        "-r",
        "--rescale",
        default=1.0,
        type=float,
        help="amount to rescale the images by",
    )
    parser.add_argument(
        "-z",
        "--visualize",
        dest="visualize",
        action="store_true",
        help="visualize alignments",
    )
    parser.add_argument(
        "--heuristic",
        dest="heuristic",
        action="store_true",
        help="do approximate clique search",
    )
    parser.add_argument(
        "-o", "--output-dir", default="./", help="folder to save aligned images"
    )
    parser.add_argument(
        "-x",
        "--no-save-outputs",
        dest="save_output",
        action="store_false",
        help="avoid saving outputs (default)",
    )
    parser.set_defaults(
        allow_all=False, visualize=False, save_output=True, heuristic=False
    )
    d = parser.parse_args()
    runner(
        input_zip=d.input_zip,
        delta=d.delta,
        epsilon=d.epsilon,
        lower_bound=d.lower_bound,
        total=d.total,
        visualize=d.visualize,
        rescale=d.rescale,
        heuristic=d.heuristic,
        output_dir=d.output_dir if d.save_output else None,
    )


if __name__ == "__main__":
    main()
