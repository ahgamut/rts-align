import sys
import argparse
import math
import itertools
import numpy as np
import pandas as pd
import time

# https://github.com/ahgamut/cliquematch/tree/devel
import cliquematch

#
from rts_align import construct_graph_2d
from rts_align import KabschEstimate
from rts_align import find_clique
from rts_align.clq import get_clique

# https://github.com/ariarobotics/clipperp
# a524943411bf6635219ab510864c81aa1b6a0c7a
# (patch headers in python bindings)
import clipperpluspy


def generate_points(n, md=10):
    pts = np.zeros((n, 3))
    pts[0] = np.random.uniform(-500, 500, 3)
    for i in range(1, n):
        pt = [0, 0, 0]
        while True:
            pt = np.random.uniform(-500, 500, 3)
            dist = np.apply_along_axis(lambda x: np.linalg.norm(x - pt), 1, pts[:i, :])
            if np.min(dist) > md:
                pts[i] = pt
                break
    return pts


def make_rotmat():
    v = np.random.normal(0, 1, 4)
    q = v / np.linalg.norm(v)
    q *= np.sign(q[0])
    q0, q1, q2, q3 = q

    R = [
        [2 * (q0**2 + q1**2) - 1, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 2 * (q0**2 + q2**2) - 1, 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 2 * (q0**2 + q3**2) - 1],
    ]
    R = np.array(R)
    return q, R


def rotmat_to_angles(rotmat):
    # https://learnopencv.com/rotation-matrix-to-euler-angles/
    sy = np.sqrt(rotmat[0, 0] * rotmat[0, 0] + rotmat[1, 0] * rotmat[1, 0])
    singular = sy < 1e-6

    if singular:
        roll = np.arctan2(-rotmat[1, 2], rotmat[1, 1])
        pitch = np.arctan2(-rotmat[2, 0], sy)
        yaw = 0
    else:
        roll = np.arctan2(rotmat[2, 1], rotmat[2, 2])
        pitch = np.arctan2(-rotmat[2, 0], sy)
        yaw = np.arctan2(rotmat[1, 0], rotmat[0, 0])

    return roll, pitch, yaw


def rigid_form(pts, rotmat, d):
    return np.matmul(pts, rotmat) + d


###


def clipperp_estim(q_pts, k_pts, delta, epsilon):
    delta = delta * np.pi / 180.0
    qlen = len(q_pts)
    klen = len(k_pts)

    # timer
    start_time = time.time()
    adjmat = construct_graph_3d(
        q_pts, k_pts, delta=delta, epsilon=epsilon, max_ratio=10, min_ratio=0.1
    )

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
    roll, pitch, yaw = rotmat_to_angles(rotmat)
    zoom_est = np.linalg.det(tform.coefs[1:, :]) ** (1 / 3)

    sol = dict()
    sol["clipperp_zoom"] = zoom_est
    sol["clipperp_roll"] = roll
    sol["clipperp_pitch"] = pitch
    sol["clipperp_yaw"] = yaw
    sol["clipperp_dx"] = transl_est[0]
    sol["clipperp_dy"] = transl_est[1]
    sol["clipperp_dz"] = transl_est[2]
    sol["clipperp_time"] = float(tm["end"] - tm["start"])
    sol["clipperp_time-clq"] = float(tm["end"] - tm["mid"])
    sol["clipperp_time-graph"] = float(tm["mid"] - tm["start"])
    return sol


#####


def rts_estim(q_pts, k_pts, delta, epsilon):
    # find corresponding points and visualize
    sol0 = find_clique(q_pts, k_pts, delta=delta, epsilon=epsilon)
    qc, kc, tm = sol0["qc"], sol0["kc"], sol0["tm"]
    tform = KabschEstimate(kc, qc)

    transl_est = tform.coefs[0, :]
    rotmat = tform.coefs[1:, :]
    roll, pitch, yaw = rotmat_to_angles(rotmat)
    zoom_est = np.linalg.det(tform.coefs[1:, :]) ** (1 / 3)

    sol = dict()
    sol["rts_zoom"] = zoom_est
    sol["rts_roll"] = roll
    sol["rts_pitch"] = pitch
    sol["rts_yaw"] = yaw
    sol["rts_dx"] = transl_est[0]
    sol["rts_dy"] = transl_est[1]
    sol["rts_dz"] = transl_est[2]
    sol["rts_time"] = float(tm["end"] - tm["start"])
    sol["rts_time-clq"] = float(tm["end"] - tm["mid"])
    sol["rts_time-graph"] = float(tm["mid"] - tm["start"])
    return sol


#####


def attempt(num_K, num_extra=0, noise_range=1, delta=0.1, epsilon=0.1):
    k_pts = generate_points(num_K + num_extra)

    # randomly select R/T/S
    zoom = np.random.uniform(1 / 5.2, 5.2)
    qua, rotmat = make_rotmat()
    roll, pitch, yaw = rotmat_to_angles(rotmat)
    translation = np.random.uniform(-75, 75, 3)
    q_pts = rigid_form(k_pts[:num_K, :] * zoom, rotmat, translation)

    # add extra points
    if num_extra != 0:
        q_extra = generate_points(num_extra)
        q_pts = np.row_stack([q_pts, q_extra])

    # add noise
    q_pts = q_pts + noise_range * np.random.normal(0, 1, (len(q_pts), 3))
    k_pts = k_pts

    # shuffle the points
    np.random.shuffle(q_pts)
    np.random.shuffle(k_pts)

    # mappings that don't require correspondence
    sol_rts = rts_estim(q_pts, k_pts, delta, epsilon)
    sol_clipperp = clipperp_estim(q_pts, k_pts, delta, epsilon)

    # add entries
    res = dict()
    res["num-points"] = num_K
    res["num-outliers"] = num_extra
    res["delta"] = delta
    res["epsilon"] = epsilon
    res["g-noise"] = noise_range
    res["zoom"] = zoom
    res["roll"] = roll
    res["pitch"] = pitch
    res["yaw"] = yaw
    res["dx"] = translation[0]
    res["dy"] = translation[1]
    res["dz"] = translation[2]

    res.update(sol_rts)
    res.update(sol_clipperp)
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
            r = attempt(d.num_K, d.num_extra, d.noise_add, d.delta, d.epsilon)
            result.append(r)
        except Exception as e:
            print("attempt failure", i, e, file=sys.stderr)
        i += 1

    df = pd.DataFrame(result)
    df.to_csv(d.output_csv, index=False, header=True)


if __name__ == "__main__":
    main()
