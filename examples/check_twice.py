import argparse
import time
import math
import numpy as np
import cliquematch

#
from rts_align import construct_graph
from rts_align import KabschEstimate


def generate_points(n, md=10):
    pts = np.zeros((n, 2))
    pts[0] = np.random.uniform(-100, 100, 2)
    for i in range(1, n):
        pt = [0, 0]
        while True:
            pt = np.random.uniform(-100, 100, 2)
            dist = np.apply_along_axis(lambda x: np.linalg.norm(x - pt), 1, pts[:i, :])
            if np.min(dist) > md:
                pts[i] = pt
                break
    return pts


def rigid_form(pts, theta, d):
    rotmat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return np.matmul(pts, rotmat) + d


def strip_graph(mat, k):
    deg = np.sum(mat | mat.T, axis=0) + 1
    prev_n = len(mat)
    removes = set(np.nonzero(deg < k)[0])
    n = np.sum(deg >= k)
    while n != prev_n:
        # print(n)
        for x in removes:
            mat[x, :] = False
            mat[:, x] = False
        deg = np.sum(mat | mat.T, axis=0) + 1
        prev_n = n
        removes = set(np.nonzero(deg < k)[0]) - removes
        n = np.sum(deg >= k)
    return np.nonzero(deg >= k)[0]


def find_clique(q_pts, k_pts, delta=0.01, epsilon=0.1):
    delta = delta * np.pi / 180.0
    qlen = len(q_pts)
    klen = len(k_pts)

    res0 = construct_graph(
        q_pts, k_pts, delta=delta, epsilon=epsilon, max_ratio=10, min_ratio=0.1
    )
    res1 = res0 != 0
    G1 = cliquematch.Graph.from_matrix(res1)
    c1 = (
        np.array(
            G1.get_max_clique(
                upper_bound=min(qlen, klen), use_dfs=False, use_heuristic=True
            ),
            dtype=np.int32,
        )
        - 1
    )
    l1 = len(c1)

    # G1_deg = np.sum(res1 | res1.T, axis=0) + 1
    # G2_ind = np.nonzero(G1_deg >= l1)[0]
    G2_ind = strip_graph(res1, l1)
    res2 = res1[G2_ind, :]
    res2 = res2[:, G2_ind]
    # print(res1.shape, res2.shape)
    G2 = cliquematch.Graph.from_matrix(res2)

    c2 = (
        np.array(
            G2.get_max_clique(
                lower_bound=max(1, l1 - 1),
                upper_bound=min(qlen, klen),
                use_dfs=True,
                use_heuristic=False,
            ),
            dtype=np.int32,
        )
        - 1
    )
    l2 = len(c2)
    # print(l1, l2)
    c_sub = G2_ind[c2]
    qc = q_pts[c_sub // len(k_pts), :]
    kc = k_pts[c_sub % len(k_pts), :]
    return qc, kc


def attempt(num_K, num_extra=0, noise_range=1, delta=0.1, epsilon=0.1):
    k_pts = generate_points(num_K + num_extra)

    # randomly select R/T/S
    zoom = np.random.uniform(1 / 5.2, 5.2)
    theta = np.pi * np.round(np.random.uniform(-1, 1), 5)
    translation = np.random.randint(-75, 75, 2)
    q_pts = rigid_form(k_pts[:num_K, :] * zoom, theta, translation)

    # add extra points
    if num_extra != 0:
        q_extra = generate_points(num_extra)
        q_pts = np.row_stack([q_pts, q_extra])

    # add noise
    q_pts = q_pts + noise_range * np.random.normal(0, 1, (len(q_pts), 2))
    k_pts = k_pts + noise_range * np.random.normal(0, 1, (len(k_pts), 2))

    # find corresponding points and visualize
    qc, kc = find_clique(q_pts, k_pts, delta=delta, epsilon=epsilon)
    tform = KabschEstimate(kc, qc)

    transl_est = tform.coefs[0, :]
    theta_est = np.arctan2(tform.coefs[1, 1], tform.coefs[1, 0])
    zoom_est = np.sqrt(np.linalg.det(tform.coefs[1:, :]))
    print(zoom_est, theta_est, transl_est)
    print(zoom, theta, translation)


def main():
    parser = argparse.ArgumentParser(
        "rts-check-points", description="check alignment of points", add_help=True
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
        "-a", "--noise-add", default=1, help="add some gaussian noise", type=float
    )
    parser.add_argument(
        "--delta", default=0.1, help="delta tuning parameter", type=float
    )
    parser.add_argument(
        "--epsilon", default=0.1, help="epsilon tuning parameter", type=float
    )

    d = parser.parse_args()
    attempt(d.num_K, d.num_extra, d.noise_add, d.delta, d.epsilon)


if __name__ == "__main__":
    main()
