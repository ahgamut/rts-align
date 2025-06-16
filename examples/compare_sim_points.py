import sys
import argparse
import math
import itertools
import numpy as np
import cliquematch
import pandas as pd
import time

#
from rts_align import construct_graph
from rts_align import KabschEstimate
from rts_align import find_clique
from rts_align.clq import get_clique

# https://github.com/yangjiaolong/Go-ICP
# via https://github.com/aalavandhaann/go-icp_cython
# 4568dd976fc5a63246835edbc748f35bc483f409
import py_goicp

# https://github.com/ariarobotics/clipperp
# a524943411bf6635219ab510864c81aa1b6a0c7a
# (patch headers in python bindings)
import clipperpluspy

# https://github.com/MIT-SPARK/TEASER-plusplus
# f91cfdb7baed951a3607257bd31f3f6694773497
import teaserpp_python


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


###


def make_p3d_list(arr, msize):
    N = len(arr)
    res = []
    for i in range(N):
        x, y = (arr[i, :]) / (1.2 * msize)  # all values inside [-1, 1]
        res.append(py_goicp.POINT3D(x, y, 0.0))
    return N, res


def goicp_estim(q_pts, k_pts, outlier_frac):
    msize = max(np.max(np.abs(q_pts)), np.max(np.abs(k_pts)))
    Nq, qp = make_p3d_list(q_pts, msize)
    Nk, kp = make_p3d_list(k_pts, msize)

    mod = py_goicp.GoICP()
    mod.MSEThresh = 0.0001
    mod.trimFraction = outlier_frac
    start_time = time.time()

    mod.loadModelAndData(Nk, kp, Nq, qp)
    mod.setDTSizeAndFactor(300, 2.0)
    mod.BuildDT()
    mod.Register()

    end_time = time.time()
    scale = np.linalg.det(mod.optimalRotation())
    rotmat = np.array(mod.optimalRotation())[:2, :2]
    trans0 = np.array(mod.optimalTranslation())[:2] * 1.2 * msize
    transl = -np.matmul(rotmat.T, trans0)

    sol = dict()
    sol["goicp_zoom"] = scale
    sol["goicp_theta"] = np.arctan2(rotmat[0, 1], rotmat[0, 0])
    sol["goicp_theta2"] = np.arctan2(rotmat.T[0, 1], rotmat.T[0, 0])
    sol["goicp_dx"] = transl[0]
    sol["goicp_dy"] = transl[1]
    sol["goicp_time"] = float(end_time - start_time)
    return sol


def goicp_estim2(q_pts, k_pts, outlier_frac, scale):
    msize = max(np.max(np.abs(q_pts)), np.max(np.abs(k_pts * scale)))
    Nq, qp = make_p3d_list(q_pts, msize)
    Nk, kp = make_p3d_list(k_pts * scale, msize)

    mod = py_goicp.GoICP()
    mod.MSEThresh = 0.0001
    mod.trimFraction = 0.5
    start_time = time.time()

    mod.loadModelAndData(Nk, kp, Nq, qp)
    mod.setDTSizeAndFactor(300, 2.0)
    mod.BuildDT()
    mod.Register()

    end_time = time.time()
    rotmat = np.array(mod.optimalRotation())[:2, :2]
    trans0 = np.array(mod.optimalTranslation())[:2] * 1.2 * msize
    transl = -np.matmul(rotmat.T, trans0)

    sol = dict()
    sol["goicp-scaled_zoom"] = scale
    sol["goicp-scaled_theta"] = np.arctan2(rotmat[0, 1], rotmat[0, 0])
    sol["goicp-scaled_theta2"] = np.arctan2(rotmat.T[0, 1], rotmat.T[0, 0])
    sol["goicp-scaled_dx"] = transl[0]
    sol["goicp-scaled_dy"] = transl[1]
    sol["goicp-scaled_time"] = float(end_time - start_time)
    return sol


####


def clipperp_estim(q_pts, k_pts, delta, epsilon):
    delta = delta * np.pi / 180.0
    qlen = len(q_pts)
    klen = len(k_pts)

    # timer
    start_time = time.time()
    adjmat = construct_graph(
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
    theta_est = np.arctan2(tform.coefs[1, 1], tform.coefs[1, 0])
    zoom_est = np.sqrt(np.linalg.det(tform.coefs[1:, :]))

    sol = dict()
    sol["clipperp_zoom"] = zoom_est
    sol["clipperp_theta"] = theta_est
    sol["clipperp_dx"] = transl_est[0]
    sol["clipperp_dy"] = transl_est[1]
    sol["clipperp_time"] = float(tm["end"] - tm["start"])
    sol["clipperp_time-clq"] = float(tm["end"] - tm["mid"])
    sol["clipperp_time-graph"] = float(tm["mid"] - tm["start"])
    return sol


#####


def teaser_estim(q_pts, k_pts, noise_range):
    dst = np.column_stack([q_pts, 0 * np.ones(len(q_pts))]).T
    src = np.column_stack([k_pts, 0 * np.ones(len(k_pts))]).T

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 5 * noise_range
    solver_params.estimate_scaling = True
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_tim_graph = (
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.COMPLETE
    )
    solver_params.inlier_selection_mode = (
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 1000
    solver_params.rotation_cost_threshold = 1e-12
    solver_params.kcore_heuristic_threshold = 1.0

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()
    solver.solve(src, dst)
    end = time.time()

    solution = solver.getSolution()

    sol = dict()
    sol["teaser_zoom"] = solution.scale
    rotmat = solution.rotation[:2, :2]
    sol["teaser_theta"] = np.arctan2(rotmat[1, 0], rotmat[1, 1])
    sol["teaser_theta2"] = np.arctan2(rotmat.T[1, 0], rotmat.T[1, 1])
    sol["teaser_dx"] = solution.translation[0]
    sol["teaser_dy"] = solution.translation[1]
    sol["teaser_time"] = float(end - start)
    return sol


def teaser2_estim(q_pts, k_pts, noise_range):
    sol0 = teaser_estim(q_pts, k_pts, noise_range)
    sol = dict()
    for k, v in sol0.items():
        k2 = k.replace("teaser", "teaser-noshuf")
        sol[k2] = v
    return sol


#####


def rts_estim(q_pts, k_pts, delta, epsilon):
    # find corresponding points and visualize
    sol0 = find_clique(q_pts, k_pts, delta=delta, epsilon=epsilon)
    qc, kc, tm = sol0["qc"], sol0["kc"], sol0["tm"]
    tform = KabschEstimate(kc, qc)

    transl_est = tform.coefs[0, :]
    theta_est = np.arctan2(tform.coefs[1, 1], tform.coefs[1, 0])
    zoom_est = np.sqrt(np.linalg.det(tform.coefs[1:, :]))

    sol = dict()
    sol["rts_zoom"] = zoom_est
    sol["rts_theta"] = theta_est
    sol["rts_dx"] = transl_est[0]
    sol["rts_dy"] = transl_est[1]
    sol["rts_time"] = float(tm["end"] - tm["start"])
    sol["rts_time-clq"] = float(tm["end"] - tm["mid"])
    sol["rts_time-graph"] = float(tm["mid"] - tm["start"])
    return sol


#####


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
    k_pts = k_pts

    # mappings that require correspondence
    sol_teaser2 = teaser2_estim(q_pts, k_pts, noise_range)

    # shuffle the points
    np.random.shuffle(q_pts)
    np.random.shuffle(k_pts)

    # mappings that don't require correspondence
    sol_rts = rts_estim(q_pts, k_pts, delta, epsilon)
    sol_clipperp = clipperp_estim(q_pts, k_pts, delta, epsilon)
    sol_teaser1 = teaser_estim(q_pts, k_pts, noise_range)
    outlier_frac = (1 + num_extra) / (1 + len(k_pts))
    sol_goicp1 = goicp_estim(q_pts, k_pts, outlier_frac)
    sol_goicp2 = goicp_estim2(q_pts, k_pts, outlier_frac, zoom)

    # add entries
    res = dict()
    res["num-points"] = num_K
    res["num-outliers"] = num_extra
    res["delta"] = delta
    res["epsilon"] = epsilon
    res["g-noise"] = noise_range
    res["zoom"] = zoom
    res["theta"] = theta
    res["dx"] = translation[0]
    res["dy"] = translation[1]

    res.update(sol_teaser1)
    res.update(sol_teaser2)
    res.update(sol_rts)
    res.update(sol_clipperp)
    res.update(sol_goicp1)
    res.update(sol_goicp2)
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
            print("attempt failure", e)
            i -= 1
        i += 1

    df = pd.DataFrame(result)
    df.to_csv(d.output_csv, index=False, header=True)


if __name__ == "__main__":
    main()
