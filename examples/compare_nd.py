import sys
import argparse
import math
import itertools
import numpy as np
import pandas as pd
import time
import random
import base64
from scipy.stats import special_ortho_group
from sklearn.metrics import pairwise_distances

##
from estim_nd import RTS, RTSHeuristic, CLIPPER


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


#####
def make_estims(d):
    result = dict()

    outlier_frac = (1 + d.num_extra) / (1 + d.num_K + d.num_extra)

    estim_objs = [
        CLIPPER(delta=d.delta, epsilon=d.epsilon),
        RTS(delta=d.delta, epsilon=d.epsilon),
        RTSHeuristic(delta=d.delta, epsilon=d.epsilon),
    ]

    for e in estim_objs:
        result[e.__estim_name__] = e
    return result


#####


def attempt(
    num_K,
    estims,
    num_extra=0,
    noise_range=1,
    delta=0.1,
    epsilon=0.1,
    dimension=2,
):
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
    sol_rts1 = estims["rts"](q_pts, k_pts, delta, epsilon)
    sol_rts2 = estims["rts-heuristic"](q_pts, k_pts, delta, epsilon)
    sol_clipperp = estims["clipperp"](q_pts, k_pts, delta, epsilon)

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
    res.update(sol_rts1)
    res.update(sol_rts2)
    print(res, file=sys.stderr)
    return res


def encodemat(mlist):
    return base64.b32encode(str(mlist).encode("ascii")).hex()


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
        "--delta", default=0.001, help="delta tuning parameter", type=float
    )
    parser.add_argument(
        "--epsilon", default=0.001, help="epsilon tuning parameter", type=float
    )
    parser.add_argument(
        "-o", "--output-csv", default="./sample.csv", help="output csv file"
    )

    d = parser.parse_args()
    estims = make_estims(d)
    result = []
    i = 0
    while i < d.simulations:
        try:
            print(i, file=sys.stderr)
            dimension = random.randint(d.min_dimension, d.max_dimension)
            r = attempt(
                d.num_K,
                estims,
                d.num_extra,
                d.noise_add,
                d.delta,
                d.epsilon,
                dimension,
            )
            result.append(r)
        except Exception as e:
            print("attempt failure", i, e, file=sys.stderr)
        i += 1

    df = pd.DataFrame(result)
    for cname in df.columns:
        if "translation" in cname or "rotation" in cname:
            df[cname] = df[cname].apply(encodemat)

    df.to_csv(d.output_csv, index=False, header=True)


if __name__ == "__main__":
    main()
