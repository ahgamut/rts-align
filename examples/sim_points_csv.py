import argparse
import math
import itertools
import numpy as np
import cliquematch
import pandas as pd

#
from rts_align import construct_graph
from rts_align import KabschEstimate


def generate_points(n, md=10):
    pts = np.zeros((n, 2))
    pts[0] = np.random.uniform(-500, 500, 2)
    for i in range(1, n):
        pt = [0, 0]
        while True:
            pt = np.random.uniform(-500, 500, 2)
            dist = np.apply_along_axis(lambda x: np.linalg.norm(x - pt), 1, pts[:i, :])
            if np.min(dist) > md:
                pts[i] = pt
                break
    return pts


def rigid_form(pts, theta, d):
    rotmat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return np.matmul(pts, rotmat) + d


def find_clique(q_pts, k_pts, delta=0.01, epsilon=0.1):
    delta = delta * np.pi / 180.0
    res0 = construct_graph(
        q_pts, k_pts, delta=delta, epsilon=epsilon, max_ratio=10, min_ratio=0.1
    )
    res = res0 != 0
    G = cliquematch.Graph.from_matrix(res)
    qlen = len(q_pts)
    klen = len(k_pts)
    c = np.array(G.get_max_clique(upper_bound=min(qlen, klen)), dtype=np.int32) - 1
    qc = q_pts[c // len(k_pts), :]
    kc = k_pts[c % len(k_pts), :]
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

    matched = 0
    for i in range(len(qc)):
        if np.linalg.norm(tform(kc[i]) - qc[i]) < 0.1:
            matched += 1

    transl_est = tform.coefs[0, :]
    theta_est = np.arctan2(tform.coefs[1, 1], tform.coefs[1, 0])
    zoom_est = np.sqrt(np.linalg.det(tform.coefs[1:, :]))
    res = dict()
    res["num_points"] = num_K
    res["num_outliers"] = num_extra
    res["delta"] = delta
    res["epsilon"] = epsilon
    res["g_noise"] = noise_range
    res["zoom"] = zoom
    res["theta"] = theta
    res["dx"] = translation[0]
    res["dy"] = translation[1]
    res["matched"] = matched
    res["zoom_est"] = zoom_est
    res["theta_est"] = theta_est
    res["dx_est"] = transl_est[0]
    res["dy_est"] = transl_est[1]
    return res


def main():
    parser = argparse.ArgumentParser(
        "rts-sim-points-csv", description="check alignment of points", add_help=True
    )
    parser.add_argument(
        "-n", "--simulations", default=5, type=int, help="number of simulations"
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
        "-a", "--noise-add", default=0.01, help="add some gaussian noise", type=float
    )
    parser.add_argument(
        "--delta", default=0.1, help="delta tuning parameter", type=float
    )
    parser.add_argument(
        "--epsilon", default=0.1, help="epsilon tuning parameter", type=float
    )
    parser.add_argument(
        "-o", "--output-csv", default="./sample.csv", help="output csv file"
    )

    d = parser.parse_args()
    result = []
    for i in range(d.simulations):
        print(i)
        r = attempt(d.num_K, d.num_extra, d.noise_add, d.delta, d.epsilon)
        result.append(r)

    df = pd.DataFrame(result)
    df.to_csv(d.output_csv, index=False, header=True)


if __name__ == "__main__":
    main()
