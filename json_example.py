import numpy as np
import cliquematch
import sys, os
import time
import random
import json
from scipy.spatial.distance import pdist, squareform
from skimage.transform import PolynomialTransform, AffineTransform, SimilarityTransform

from triples.builder3 import construct_graph


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


def wcc(adjmat):
    visited = np.zeros(len(adjmat), dtype=np.bool_)
    components = []

    def recwcc(ind):
        if visited[ind]:
            return
        visited[ind] = True
        components[-1].add(ind)
        print(len(components), len(components[-1]))
        for x in adjmat[ind].nonzero()[0]:
            if not visited[x]:
                recwcc(x)
                visited[x] = True

    for i in range(len(adjmat)):
        if not visited[i]:
            components.append(set([i]))
            recwcc(i)
            visited[i] = True

    for i, x in enumerate(components):
        print(i, len(x), x)

    return None + 1


def clean_graph(adjmat, qlen, klen, lower_bound):
    for i in range(len(adjmat)):
        if np.sum(adjmat[i]) < lower_bound:
            adjmat[i, :] = False
            adjmat[:, i] = False
    return adjmat


def find_clique(q_pts, k_pts, delta=0.01, epsilon=1, lower_bound=10):
    res = construct_graph(q_pts, k_pts, delta, epsilon)
    res = res | res.T
    res = res != 0
    print(np.sum(res))
    clean_graph(res, len(q_pts), len(k_pts), lower_bound)
    print(np.sum(res))
    color = color_greed(res)
    print(res.shape, 2 * np.sum(res) / (res.shape[0] * (res.shape[0] - 1)))
    G = cliquematch.Graph.from_matrix(res)
    maxsize = min(len(q_pts), len(k_pts))
    c = (
        np.array(
            G.get_max_clique(
                use_dfs=True, lower_bound=lower_bound, upper_bound=maxsize
            ),
            dtype=np.int32,
        )
        - 1
    )
    print("color", color[c])
    qc = q_pts[c // len(k_pts), :]
    kc = k_pts[c % len(k_pts), :]
    print("clique size", len(c))
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
    import viz_points

    viz_points.show_points(q_pts, k_pts, qc, kc, order=2, num_steps=5)
    beat = rmsd(k_poly2, qc)
    for cs in G.all_cliques(size=len(c)):
        c0 = np.array(cs) - 1
        qc0 = q_pts[c0 // len(k_pts), :]
        kc0 = k_pts[c0 % len(k_pts), :]
        zr = pdist(qc0) / pdist(kc0)
        pform.estimate(kc0, qc0)
        a = rmsd(pform(kc0), qc0)
        print(c0, a, beat, "zoom", np.mean(zr), np.std(zr))
        viz_points.show_points(q_pts, k_pts, qc0, kc0, order=2, num_steps=5)
        if a < beat:
            beat = a
            qc = qc0
            kc = kc0


def attempt(k_pts, q_pts, c_size):
    find_clique(q_pts, k_pts, delta=0.005, epsilon=0.05, lower_bound=c_size)


def ok():
    a = json.load(open("./QK001-KG-minimal.json"))
    b = json.load(open("./QK001-QC-minimal.json"))
    # print(a)
    k_pts = np.array(a["valid"]) / 12 - 200
    q_pts = np.array(b["valid"]) / 12 - 200

    print(len(k_pts), len(q_pts))
    attempt(k_pts / 2, q_pts, c_size=4)


def main():
    a = json.load(open("./QK001-KG.json"))
    b = json.load(open("./QK001-QC.json"))
    # print(a)
    k_pts = np.array(a["valid"])[1:] / 18
    q_pts = np.array(b["valid"])[1:] / 12

    print(len(k_pts), len(q_pts))
    attempt(k_pts, q_pts, c_size=4)


if __name__ == "__main__":
    main()
