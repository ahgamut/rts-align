import sys
import argparse
import math
import itertools
import numpy as np
import pandas as pd
import time
import traceback

# https://github.com/ahgamut/cliquematch/tree/devel
import cliquematch

#
from rts_align import construct_graph_3d
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


class Base3DEstimator:
    __estim_name__ = None
    __default_params__ = set(["zoom", "roll", "pitch", "yaw", "dx", "dy", "dz"])
    __aux_params__ = set(["success", "time"])

    def __init__(self, *args, **kwargs):
        params = set(kwargs.get("params", [])) | self.__default_params__
        params = params - self.__aux_params__
        e = self.__estim_name__
        self._dummy_result = dict()
        self._dummy_params = set()
        for k in params:
            self._dummy_result[f"{e}_{k}"] = 0.0
            self._dummy_params.add(f"{e}_{k}")
        self._dummy_result[f"{e}_success"] = False

    def _call_impl(self, q_pts, k_pts, *args, **kwargs):
        raise NotImplementedError("subclass Base3DEstimator")

    def __call__(self, q_pts, k_pts, *args, **kwargs):
        result = dict(**self._dummy_result)
        e = self.__estim_name__
        try:
            start = time.time()
            subres = self._call_impl(q_pts, k_pts, *args, **kwargs)
            end = time.time()
            for k in self._dummy_params:
                result[k] = subres[k]
            result[f"{e}_success"] = True
            result[f"{e}_time"] = end - start
        except Exception as err:
            msg = traceback.format_exc()
            print(f"estimation with {self.__estim_name__} failed with", msg)
            result[f"{e}_success"] = False
            result[f"{e}_time"] = 0.0
        return result

    def rotmat_to_angles(self, rotmat):
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


######
def make_p3d_list(arr, msize):
    N = len(arr)
    res = []
    for i in range(N):
        x, y, z = (arr[i, :]) / (1.2 * msize)  # all values inside [-1, 1]
        res.append(py_goicp.POINT3D(x, y, z))
    return N, res


class GoICP(Base3DEstimator):
    __estim_name__ = "goicp"

    def __init__(self, outlier_frac, mse_thresh=0.0001):
        super().__init__()
        self.outlier_frac = outlier_frac
        self.mse_thresh = mse_thresh

    def _call_impl(self, q_pts, k_pts, *args, **kwargs):
        msize = max(np.max(np.abs(q_pts)), np.max(np.abs(k_pts)))
        Nq, qp = make_p3d_list(q_pts, msize)
        Nk, kp = make_p3d_list(k_pts, msize)

        mod = py_goicp.GoICP()
        mod.MSEThresh = self.mse_thresh
        mod.trimFraction = self.outlier_frac

        mod.loadModelAndData(Nk, kp, Nq, qp)
        mod.setDTSizeAndFactor(300, 2.0)
        mod.BuildDT()
        mod.Register()

        scale = np.linalg.det(mod.optimalRotation())
        rotmat = np.array(mod.optimalRotation())
        roll, pitch, yaw = self.rotmat_to_angles(rotmat)
        trans0 = np.array(mod.optimalTranslation()) * 1.2 * msize
        transl = -np.matmul(rotmat.T, trans0)

        sol = dict()
        e = self.__estim_name__
        sol[f"{e}_zoom"] = scale
        sol[f"{e}_roll"] = roll
        sol[f"{e}_pitch"] = pitch
        sol[f"{e}_yaw"] = yaw
        sol[f"{e}_dx"] = transl[0]
        sol[f"{e}_dy"] = transl[1]
        sol[f"{e}_dz"] = transl[2]
        return sol


class GoICPScaled(Base3DEstimator):
    __estim_name__ = "goicp-scaled"

    def __init__(self, outlier_frac, mse_thresh=0.0001):
        super().__init__()
        self.outlier_frac = outlier_frac
        self.mse_thresh = mse_thresh

    def _call_impl(self, q_pts, k_pts, *args, **kwargs):
        scale = kwargs["scale"]
        msize = max(np.max(np.abs(q_pts)), np.max(np.abs(k_pts * scale)))
        Nq, qp = make_p3d_list(q_pts, msize)
        Nk, kp = make_p3d_list(k_pts * scale, msize)

        mod = py_goicp.GoICP()
        mod.MSEThresh = self.mse_thresh
        mod.trimFraction = self.outlier_frac

        mod.loadModelAndData(Nk, kp, Nq, qp)
        mod.setDTSizeAndFactor(300, 2.0)
        mod.BuildDT()
        mod.Register()

        scale = np.linalg.det(mod.optimalRotation())
        rotmat = np.array(mod.optimalRotation())
        roll, pitch, yaw = self.rotmat_to_angles(rotmat)
        trans0 = np.array(mod.optimalTranslation()) * 1.2 * msize
        transl = -np.matmul(rotmat.T, trans0)

        sol = dict()
        e = self.__estim_name__
        sol[f"{e}_zoom"] = kwargs["scale"]
        sol[f"{e}_roll"] = roll
        sol[f"{e}_pitch"] = pitch
        sol[f"{e}_yaw"] = yaw
        sol[f"{e}_dx"] = transl[0]
        sol[f"{e}_dy"] = transl[1]
        sol[f"{e}_dz"] = transl[2]
        return sol


class TEASER(Base3DEstimator):
    __estim_name__ = "teaser"

    def __init__(self, noise_range):
        super().__init__()
        self.noise_range = noise_range

    def _call_impl(self, q_pts, k_pts, *args, **kwargs):
        dst = q_pts.T
        src = k_pts.T

        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = 5 * self.noise_range
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
        solver.solve(src, dst)
        solution = solver.getSolution()
        rotmat = solution.rotation
        roll, pitch, yaw = self.rotmat_to_angles(rotmat)

        e = self.__estim_name__
        sol = dict()
        sol[f"{e}_zoom"] = solution.scale
        sol[f"{e}_roll"] = roll
        sol[f"{e}_pitch"] = pitch
        sol[f"{e}_yaw"] = yaw
        sol[f"{e}_dx"] = solution.translation[0]
        sol[f"{e}_dy"] = solution.translation[1]
        sol[f"{e}_dz"] = solution.translation[2]
        return sol


class TEASER_NoShuf(TEASER):
    __estim_name__ = "teaser-noshuf"


class CLIPPER(Base3DEstimator):
    __estim_name__ = "clipperp"

    def __init__(self, delta, epsilon, min_ratio=0.1, max_ratio=10.0):
        super().__init__(params=["time-clq", "time-graph"])
        self.delta = delta
        self.epsilon = epsilon
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def _call_impl(self, q_pts, k_pts, *args, **kwargs):
        qlen = len(q_pts)
        klen = len(k_pts)

        # timer
        start_time = time.time()
        adjmat = construct_graph_3d(
            q_pts,
            k_pts,
            delta=self.delta,
            epsilon=self.epsilon,
            min_ratio=self.min_ratio,
            max_ratio=self.max_ratio,
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
        roll, pitch, yaw = self.rotmat_to_angles(rotmat)
        zoom_est = np.linalg.det(tform.coefs[1:, :]) ** (1 / 3)

        e = self.__estim_name__
        sol = dict()
        sol[f"{e}_zoom"] = zoom_est
        sol[f"{e}_roll"] = roll
        sol[f"{e}_pitch"] = pitch
        sol[f"{e}_yaw"] = yaw
        sol[f"{e}_dx"] = transl_est[0]
        sol[f"{e}_dy"] = transl_est[1]
        sol[f"{e}_dz"] = transl_est[2]
        sol[f"{e}_time-clq"] = float(end_time - mid_time)
        sol[f"{e}_time-graph"] = float(mid_time - start_time)
        return sol


class RTS(Base3DEstimator):
    __estim_name__ = "rts"

    def __init__(self, delta, epsilon):
        super().__init__(params=["time-clq", "time-graph"])
        self.delta = delta
        self.epsilon = epsilon
        self.use_heuristic = False

    def _call_impl(self, q_pts, k_pts, *args, **kwargs):
        sol0 = find_clique(
            q_pts,
            k_pts,
            delta=self.delta,
            epsilon=self.epsilon,
            heuristic=self.use_heuristic,
        )
        qc, kc, tm = sol0["qc"], sol0["kc"], sol0["tm"]
        tform = KabschEstimate(kc, qc)

        transl_est = tform.coefs[0, :]
        rotmat = tform.coefs[1:, :]
        roll, pitch, yaw = self.rotmat_to_angles(rotmat)
        zoom_est = np.linalg.det(tform.coefs[1:, :]) ** (1 / 3)

        e = self.__estim_name__
        sol = dict()
        sol[f"{e}_zoom"] = zoom_est
        sol[f"{e}_roll"] = roll
        sol[f"{e}_pitch"] = pitch
        sol[f"{e}_yaw"] = yaw
        sol[f"{e}_dx"] = transl_est[0]
        sol[f"{e}_dy"] = transl_est[1]
        sol[f"{e}_dz"] = transl_est[2]
        sol[f"{e}_time-clq"] = float(tm["end"] - tm["mid"])
        sol[f"{e}_time-graph"] = float(tm["mid"] - tm["start"])
        return sol


class RTSHeuristic(Base3DEstimator):
    __estim_name__ = "rts-heuristic"

    def __init__(self, delta, epsilon):
        super().__init__(params=["time-clq", "time-graph"])
        self.delta = delta
        self.epsilon = epsilon
        self.use_heuristic = True

    def _call_impl(self, q_pts, k_pts, *args, **kwargs):
        sol0 = find_clique(
            q_pts,
            k_pts,
            delta=self.delta,
            epsilon=self.epsilon,
            heuristic=self.use_heuristic,
        )
        qc, kc, tm = sol0["qc"], sol0["kc"], sol0["tm"]
        tform = KabschEstimate(kc, qc)

        transl_est = tform.coefs[0, :]
        rotmat = tform.coefs[1:, :]
        roll, pitch, yaw = self.rotmat_to_angles(rotmat)
        zoom_est = np.linalg.det(tform.coefs[1:, :]) ** (1 / 3)

        e = self.__estim_name__
        sol = dict()
        sol[f"{e}_zoom"] = zoom_est
        sol[f"{e}_roll"] = roll
        sol[f"{e}_pitch"] = pitch
        sol[f"{e}_yaw"] = yaw
        sol[f"{e}_dx"] = transl_est[0]
        sol[f"{e}_dy"] = transl_est[1]
        sol[f"{e}_dz"] = transl_est[2]
        sol[f"{e}_time-clq"] = float(tm["end"] - tm["mid"])
        sol[f"{e}_time-graph"] = float(tm["mid"] - tm["start"])
        return sol
