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

    return {
        "zoom": zoom,
        "d": np.array([dx, dy]),
        "theta": theta,
    }


def find_clique(q_pts, k_pts, delta=0.01):
    res = construct_graph(q_pts, k_pts, delta) != 0
    print(res.shape, np.sum(res) / (res.shape[0] * (res.shape[0] - 1)))
    G = cliquematch.Graph.from_matrix(res)
    maxsize = min(len(q_pts), len(k_pts))
    c = (
        np.array(
            G.get_max_clique(use_dfs=False, lower_bound=10, upper_bound=maxsize),
            dtype=np.int32,
        )
        - 1
    )
    qc = q_pts[c // len(k_pts), :]
    kc = k_pts[c % len(k_pts), :]
    print("clique size", len(c))
    qdist = pdist(qc)
    kdist = pdist(kc)
    qs_by_ks = np.mean(qdist / kdist)
    print("zoom estimate", qs_by_ks)

    tform = kabsch(qc, kc, qs_by_ks)
    k_kabsch = rigid_form(kc * qs_by_ks, tform["theta"], tform["d"])
    pform = PolynomialTransform()
    pform.estimate(kc, qc, order=4)
    k_poly2 = pform(kc)

    rmsd = lambda x: np.mean(np.sqrt(np.sum((qc - x) ** 2, axis=1)))

    print("kabsch distance", rmsd(k_kabsch))
    print("poly2 distance", rmsd(k_poly2))


def attempt(k_size=30, q_size=30, c_size=10):
    k_pts = 100 * np.float64(np.random.uniform(-1, 1, (k_size, 2)))
    q_pts = np.zeros((q_size, 2), dtype=np.float64)
    before = k_pts[np.random.choice(np.arange(0, k_size), c_size), :]
    q_pts[:c_size] = before
    q_pts[c_size:] = 100 * np.random.uniform(-1, 1, (q_size - c_size, 2))

    ratio = np.round(np.random.uniform(1, 2.5), 2)
    translation = np.array([np.random.randint(-50, 50), np.random.randint(-50, 50)])
    theta = np.pi * np.round(np.random.uniform(-1, 1), 2)

    error = 2.5 * np.random.uniform(-1, 1, (q_size, 2))
    print("error percentage", np.median(np.abs(error / q_pts)))
    q_pts = q_pts * ratio
    q_pts = rigid_form(q_pts, theta, translation) + error

    find_clique(q_pts, k_pts, delta=0.01)
    print("<CHECK> ratio should be", ratio)
    print("<CHECK> theta should be", theta)
    print("<CHECK> translation should be", translation)


def main():
    # np.random.seed(42)
    attempt(q_size=30, k_size=30, c_size=25)


if __name__ == "__main__":
    main()
