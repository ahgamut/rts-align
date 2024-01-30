import numpy as np
import cliquematch
import sys, os
import time
import random

from triples.tr_gimme import gimme_graph3

norm = np.linalg.norm

# from triples.tr_gimme import gimme_graph3, gimme_histo
# from utils import view_ratios

TOLERANCE = 1
MINDIST = 1e-1


def find_clique(q_pts, k_pts, delta=0.01, epsilon=0.01):
    res = gimme_graph3(q_pts, k_pts, delta, epsilon) != 0
    np.fill_diagonal(res, True)
    print(res)
    G = cliquematch.Graph.from_matrix(res)
    c = np.array(G.get_max_clique(), dtype=np.int32) - 1
    qc = c // len(k_pts)
    kc = c % len(k_pts)
    print("clique is", c)
    for i in range(len(c)):
        print(q_pts[qc[i]], "--", k_pts[kc[i]])


def attempt():
    q_pts = np.array([[6, 0], [-10, 0], [0, 8]])
    t = [[3, 0], [-5, 0], [0, 4]]
    random.shuffle(t)
    k_pts = np.array(t)

    translation = np.array([np.random.randint(-50, 50), np.random.randint(-50, 50)])
    theta = np.pi * np.round(np.random.uniform(-1, 1), 2)
    print(translation, theta)

    error = 0.3 * np.random.uniform(-1, 1, (len(q_pts), 2))
    q_pts = q_pts + translation
    rotmat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    # q_pts = np.matmul(q_pts, rotmat)
    # q_pts = q_pts + error
    find_clique(q_pts, k_pts)


def main():
    # np.random.seed(42)
    attempt()


if __name__ == "__main__":
    main()
