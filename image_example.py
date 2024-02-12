import numpy as np
import argparse
import cliquematch
import sys, os
import time
import random
import json
from scipy.spatial.distance import pdist, squareform
from skimage.filters import threshold_local, threshold_triangle, threshold_otsu
from skimage.transform import (
    PolynomialTransform,
    AffineTransform,
    SimilarityTransform,
    rescale,
)
import skimage.io as skio

from triples.builder3 import construct_graph
import viz_image


def rigid_form(pts, theta, d):
    rotmat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return np.matmul(pts, rotmat) + d


def kabsch(Q_pts, K_pts, zoom):
    def centerify(x):
        cent = np.mean(x, axis=0)
        return cent, x - cent

    # kabsch algorithm to get rotation and translation
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    Q_cent, Q_norm = centerify(Q_pts)
    K_cent, K_norm = centerify(K_pts * zoom)

    # why are we mapping K to Q?
    # because the original below does that
    H = np.matmul(K_norm.T, Q_norm)
    V, s, U = np.linalg.svd(H, full_matrices=True, compute_uv=True)

    rotmat = np.eye(2, 2)
    d = np.linalg.det(V) * np.linalg.det(U)
    rotmat[1, 1] = 1 if d > 0 else -1
    rotmat = np.matmul(np.matmul(V, rotmat), U.T)

    dx, dy = np.round(-np.matmul(K_cent, rotmat) + Q_cent, 2)
    theta = np.arctan2(rotmat[0, 1], rotmat[0, 0])
    theta = 0 if np.isnan(theta) or np.isinf(theta) else theta

    print(zoom)
    return {
        "zoom": zoom,
        "d": np.array([dx, dy]),
        "theta": theta,
    }


def color_greed(adjmat):
    degrees = np.sum(adjmat, axis=0)
    deg_order = np.argsort(degrees)[::-1]
    ind = np.arange(len(adjmat))
    color = np.zeros(len(adjmat), dtype=np.int32)
    visited = np.zeros(len(adjmat), dtype=np.bool_)
    visited[0] = True
    color[0] = 1
    for i in range(1, len(adjmat)):
        tmp = color[visited & adjmat[i]]
        if len(tmp) == 0:
            color[i] = 1
        else:
            color[i] = np.max(tmp) + 1
        visited[i] = True
    print("coloring number is ", np.max(color))
    # print(np.column_stack([color, degrees]))
    return color


def clean_graph(adjmat, qlen, klen, lower_bound):
    for i in range(len(adjmat)):
        if np.sum(adjmat[i]) < lower_bound:
            adjmat[i, :] = False
            adjmat[:, i] = False
    return adjmat


def find_clique(q_pts, k_pts, delta=0.01, epsilon=1, lower_bound=10, all_max=False):
    res = construct_graph(q_pts, k_pts, delta, epsilon)
    res = res | res.T
    res = res != 0
    clean_graph(res, len(q_pts), len(k_pts), lower_bound)
    color = color_greed(res)
    maxsize = min(len(q_pts), len(k_pts), np.max(color))
    G = cliquematch.Graph.from_matrix(res)
    print(G)

    c = G.get_max_clique(
        use_dfs=True, use_heuristic=True, lower_bound=lower_bound, upper_bound=maxsize
    )
    c = np.array(c, dtype=np.int32) - 1

    qc = q_pts[c // len(k_pts), :]
    kc = k_pts[c % len(k_pts), :]
    indices = np.column_stack([c // len(k_pts), c % len(k_pts)])
    print("clique size", len(c), indices)
    qdist = pdist(qc)
    kdist = pdist(kc)
    qs_by_ks = np.mean(qdist / kdist)
    print("zoom estimate", qs_by_ks)

    tform = kabsch(qc, kc, qs_by_ks)
    k_kabsch = rigid_form(kc * qs_by_ks, tform["theta"], tform["d"])
    pform = SimilarityTransform()
    pform.estimate(kc, qc)
    k_poly2 = pform(kc)

    rmsd = lambda x, y: np.mean(np.sqrt(np.sum((y - x) ** 2, axis=1)))

    print("kabsch distance", rmsd(k_kabsch, qc))
    print("poly2 distance", rmsd(k_poly2, qc))

    if all_max:
        # go through all cliques, find most rigid transform
        beat = rmsd(k_poly2, qc)
        for cs in G.all_cliques(size=len(c)):
            c0 = np.array(cs) - 1
            qc0 = q_pts[c0 // len(k_pts), :]
            kc0 = k_pts[c0 % len(k_pts), :]
            zr = pdist(qc0) / pdist(kc0)
            pform.estimate(kc0, qc0)
            a = rmsd(pform(kc0), qc0)
            indices0 = np.column_stack([c0 // len(k_pts), c0 % len(k_pts)])
            print(len(indices0), a, beat, "zoom", np.mean(zr), np.std(zr))
            if a < beat:
                beat = a
                qc = qc0
                kc = kc0

    return qc, kc


def loader(img_path, points_path, downscale, flip=False):
    img = skio.imread(img_path, as_gray=True)
    a = json.load(open(points_path))
    points = np.array(a["valid"])[1:, ::-1]
    if downscale != 1:
        img = rescale(img, 1 / downscale, anti_aliasing=True)
        points = points / downscale
    if flip:
        print("before", points)
        print(img.shape)
        # img = img[:, ::-1]
        points[:, 0] = img.shape[1] - points[:, 0]
        points[:, 1] = img.shape[0] - points[:, 1]
        print(points)

    return img, points


def main():
    parser = argparse.ArgumentParser("cmp-cubic")
    parser.add_argument("-q", "--q-img", required=True, help="path of Q image")
    parser.add_argument("-qp", "--q-points", required=True, help="path of Q points")
    parser.add_argument("-qd", "--q-down", default=1, type=float, help="downscale Q by")
    parser.add_argument("-k", "--k-img", required=True, help="path of K image")
    parser.add_argument("-kp", "--k-points", required=True, help="path of K points")
    parser.add_argument("-kd", "--k-down", default=1, type=float, help="downscale K by")
    parser.add_argument("--delta", default=0.007, type=float, help="delta error")
    parser.add_argument("--epsilon", default=0.05, type=float, help="epsilon error")
    parser.add_argument(
        "-lb", "--lower-bound", default=10, type=int, help="lower bound for clique size"
    )
    parser.add_argument(
        "--check-ties",
        action="store_true",
        dest="all_max",
        help="look through all maxima",
    )
    parser.add_argument(
        "--first-max",
        action="store_false",
        dest="all_max",
        help="pick only the first maximum",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="output file for saving animation"
    )
    parser.set_defaults(all_max=False)

    d = parser.parse_args()

    q_img, q_pts = loader(d.q_img, d.q_points, d.q_down)
    k_img, k_pts = loader(d.k_img, d.k_points, d.k_down)
    k_img = 1.0 * k_img > threshold_otsu(k_img)

    viz_image.show_setup(q_img, k_img, q_pts, k_pts)
    q_corr, k_corr = find_clique(
        q_pts,
        k_pts,
        delta=d.delta,
        epsilon=d.epsilon,
        lower_bound=d.lower_bound,
        all_max=d.all_max,
    )
    viz_image.show_anim(q_img, k_img, q_pts, k_pts, q_corr, k_corr, filename=d.output)


if __name__ == "__main__":
    main()
