import numpy as np
import cliquematch
import sys, os
import time
import random
from scipy.spatial.distance import pdist, squareform
from skimage.transform import PolynomialTransform

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

    print("theta is", theta, np.tan(theta))
    print("dx, dy are", dx, dy)
    return {
        "zoom": zoom,
        "d": np.array([dx, dy]),
        "theta": theta,
    }


def find_clique(q_pts, k_pts, delta=0.01, epsilon=1):
    res = construct_graph(q_pts, k_pts, delta, epsilon) != 0
    G = cliquematch.Graph.from_matrix(res)
    c = np.array(G.get_max_clique(), dtype=np.int32) - 1
    qc = q_pts[c // len(k_pts), :]
    kc = k_pts[c % len(k_pts), :]
    print("clique is", c)
    qdist = pdist(qc)
    kdist = pdist(kc)
    qs_by_ks = np.round(np.mean(qdist / kdist), 3)
    print("scale of Q/K is", qs_by_ks)
    tform = kabsch(qc, kc, qs_by_ks)
    k_kabsch = rigid_form(kc * qs_by_ks, tform["theta"], tform["d"])
    pform = PolynomialTransform()
    pform.estimate(kc, qc)
    k_poly2 = pform(kc)
    for i in range(len(c)):
        print(qc[i], "--", kc[i], k_kabsch[i], k_poly2[i], sep="\t")


def attempt():
    q_pts = np.array([[6, 0], [-10, 0], [0, 8]])
    t = [[3, 0], [-5, 0], [0, 4]]
    random.shuffle(t)
    k_pts = np.array(t)

    translation = np.array([np.random.randint(20, 50), np.random.randint(20, 50)])
    theta = np.pi * np.round(np.random.uniform(-1, 1), 2)
    error = 1.0 * np.random.uniform(-1, 1, (len(q_pts), 2))
    print(translation, theta, np.tan(theta))
    print(np.median(np.abs(error / translation)))
    q_pts = rigid_form(q_pts, theta, translation) + error
    find_clique(q_pts, k_pts, delta=0.2, epsilon=0.5)


def main():
    # np.random.seed(42)
    attempt()


if __name__ == "__main__":
    main()
