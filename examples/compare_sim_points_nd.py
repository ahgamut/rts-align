import sys
import argparse
import math
import itertools
import numpy as np
import pandas as pd
import time
import random
from scipy.stats import special_ortho_group
from sklearn.metrics import pairwise_distances

# https://github.com/ahgamut/cliquematch/tree/devel
import cliquematch

#
from rts_align.core import construct_graph
from rts_align import KabschEstimate
from rts_align import find_clique

# https://github.com/ariarobotics/clipperp
# a524943411bf6635219ab510864c81aa1b6a0c7a
# (patch headers in python bindings)
import clipperpluspy


def generate_points(n, k, md=10):
    pts = np.zeros((n, k))
    pts[0] = np.random.uniform(-500, 500, k)
    for i in range(1, n):
        while True:
            pt = np.random.uniform(-500, 500, k)
            dist = np.apply_along_axis(lambda x: np.linalg.norm(x - pt), 1, pts[:i, :])
            if np.min(dist) > md:
                pts[i] = pt
                break
    return pts


def make_rotmat(k):
    return special_ortho_group.rvs(k)


def rigid_form(pts, rotmat, d):
    return np.matmul(pts, rotmat) + d


###


def clipperp_estim(q_pts, k_pts, delta, epsilon):
    delta = delta * np.pi / 180.0
    qlen = len(q_pts)
    klen = len(k_pts)

    # timer
    start_time = time.time()
    q_dist = pairwise_distances(q_pts, metric="euclidean")
    k_dist = pairwise_distances(k_pts, metric="euclidean")
    adjmat = construct_graph(q_pts, k_pts, q_dist, k_dist, epsilon, False)

    # timer
    mid_time = time.time()

    adjmat = np.int32(adjmat != 0)
    adjmat = adjmat | adjmat.T
    np.fill_diagonal(adjmat, 0)
    clique_size, clique, certificate = clipperpluspy.clipperplus_clique(adjmat)

    # timer
    end_time = time.time()

    clique = np.array(clique, dtype=np.int32)
    qc = q_pts[clique // len(k_pts), :]
    kc = k_pts[clique % len(k_pts), :]
    tm = {"start": start_time, "mid": mid_time, "end": end_time}

    tform = KabschEstimate(kc, qc)
    transl_est = tform.coefs[0, :]
    rotmat = tform.coefs[1:, :]
    zoom_est = np.linalg.det(rotmat) ** (1 / len(transl_est))
    rotmat /= zoom_est

    sol = dict()
    sol["clipperp_zoom"] = zoom_est
    sol["clipperp_rotation"] = rotmat.tolist()
    sol["clipperp_translation"] = transl_est.tolist()
    sol["clipperp_time"] = float(tm["end"] - tm["start"])
    sol["clipperp_time-clq"] = float(tm["end"] - tm["mid"])
    sol["clipperp_time-graph"] = float(tm["mid"] - tm["start"])
    return sol


#####


def rts_estim(q_pts, k_pts, delta, epsilon):
    # find corresponding points and visualize
    p = q_pts.shape[1]
    # number of parameters 1 + p + p(p-1)/2
    # each correspondence gives p equations
    lower_bound = int(np.ceil((1 + p + 0.5 * p * (p - 1)) / p))
    sol0 = find_clique(
        q_pts, k_pts, delta=delta, epsilon=epsilon, lower_bound=lower_bound
    )
    qc, kc, tm = sol0["qc"], sol0["kc"], sol0["tm"]
    tform = KabschEstimate(kc, qc)

    transl_est = tform.coefs[0, :]
    rotmat = tform.coefs[1:, :]
    zoom_est = np.linalg.det(rotmat) ** (1 / len(transl_est))
    rotmat /= zoom_est

    sol = dict()
    sol["rts_zoom"] = zoom_est
    sol["rts_rotation"] = rotmat.tolist()
    sol["rts_translation"] = transl_est.tolist()
    sol["rts_time"] = float(tm["end"] - tm["start"])
    sol["rts_time-clq"] = float(tm["end"] - tm["mid"])
    sol["rts_time-graph"] = float(tm["mid"] - tm["start"])
    return sol


#####


def attempt(num_K, num_extra=0, noise_range=1, delta=0.1, epsilon=0.1, dimension=2):
    k_pts = generate_points(num_K + num_extra, dimension)

    # randomly select R/T/S
    zoom = np.random.uniform(1 / 5.2, 5.2)
    rotmat = make_rotmat(dimension)
    translation = np.random.uniform(-75, 75, dimension)
    q_pts = rigid_form(k_pts[:num_K, :] * zoom, rotmat, translation)

    # add extra points
    if num_extra != 0:
        q_extra = generate_points(num_extra, dimension)
        q_pts = np.row_stack([q_pts, q_extra])

    # add noise
    q_pts = q_pts + noise_range * np.random.normal(0, 1, (len(q_pts), dimension))
    k_pts = k_pts

    # shuffle the points
    np.random.shuffle(q_pts)
    np.random.shuffle(k_pts)

    # mappings that don't require correspondence
    sol_clipperp = clipperp_estim(q_pts, k_pts, delta, epsilon)
    sol_rts = rts_estim(q_pts, k_pts, delta, epsilon)

    # add entries
    res = dict()
    res["num-points"] = num_K
    res["num-outliers"] = num_extra
    res["dimension"] = dimension
    res["delta"] = delta
    res["epsilon"] = epsilon
    res["g-noise"] = noise_range
    res["zoom"] = zoom
    res["rotation"] = rotmat.tolist()
    res["translation"] = translation.tolist()

    res.update(sol_clipperp)
    res.update(sol_rts)
    print(res, file=sys.stderr)
    return res


def main():
    parser = argparse.ArgumentParser(
        "rts-compare-csv", description="compare rts", add_help=True
    )
    parser.add_argument(
        "-n", "--simulations", default=5, type=int, help="number of simulations"
    )
    parser.add_argument(
        "--min-dimension", default=2, type=int, help="minimum dimension"
    )
    parser.add_argument(
        "--max-dimension", default=2, type=int, help="maximum dimension"
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
    i = 0
    while i < d.simulations:
        try:
            print(i, file=sys.stderr)
            dimension = random.randint(d.min_dimension, d.max_dimension)
            r = attempt(
                d.num_K, d.num_extra, d.noise_add, d.delta, d.epsilon, dimension
            )
            result.append(r)
        except Exception as e:
            print("attempt failure", i, e, file=sys.stderr)
        i += 1

    df = pd.DataFrame(result)
    df.to_csv(d.output_csv, index=False, header=True)


if __name__ == "__main__":
    main()
