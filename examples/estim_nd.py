import sys
import argparse
import math
import itertools
import numpy as np
import pandas as pd
import time
import traceback
from sklearn.metrics import pairwise_distances

# https://github.com/ahgamut/cliquematch/tree/devel
import cliquematch

#
from rts_align.core import construct_graph
from rts_align import KabschEstimate
from rts_align import find_clique
from rts_align.clq import get_clique

# https://github.com/ariarobotics/clipperp
# a524943411bf6635219ab510864c81aa1b6a0c7a
# (patch headers in python bindings)
import clipperpluspy


class BaseNDEstimator:
    __estim_name__ = None
    __default_params__ = set(["zoom", "rotation", "translation"])
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
        raise NotImplementedError("subclass BaseNDEstimator")

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
            print(
                f"estimation with {self.__estim_name__} failed with",
                msg,
                file=sys.stderr,
            )
            result[f"{e}_success"] = False
            result[f"{e}_time"] = 0.0
        return result


class CLIPPER(BaseNDEstimator):
    __estim_name__ = "clipperp"

    def __init__(self, delta, epsilon, min_ratio=0.1, max_ratio=10.0):
        super().__init__(params=["time-clq", "time-graph"])
        self.delta = delta * np.pi / 180
        self.epsilon = epsilon
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def _call_impl(self, q_pts, k_pts, *args, **kwargs):
        qlen = len(q_pts)
        klen = len(k_pts)

        # timer
        start_time = time.time()
        q_dist = pairwise_distances(q_pts, metric="euclidean")
        k_dist = pairwise_distances(k_pts, metric="euclidean")
        adjmat = construct_graph(
            q_pts,
            k_pts,
            q_dist,
            k_dist,
            self.epsilon,
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
        zoom_est = np.linalg.det(rotmat) ** (1 / len(transl_est))
        rotmat /= zoom_est

        sol = dict()
        e = self.__estim_name__
        sol[f"{e}_success"] = clique_size
        sol[f"{e}_zoom"] = zoom_est
        sol[f"{e}_rotation"] = rotmat.tolist()
        sol[f"{e}_translation"] = transl_est.tolist()
        sol[f"{e}_time-clq"] = float(tm["end"] - tm["mid"])
        sol[f"{e}_time-graph"] = float(tm["mid"] - tm["start"])
        return sol


class RTS(BaseNDEstimator):
    __estim_name__ = "rts"

    def __init__(self, delta, epsilon):
        super().__init__(params=["time-clq", "time-graph"])
        self.delta = delta * np.pi / 180
        self.epsilon = epsilon
        self.use_heuristic = False

    def _call_impl(self, q_pts, k_pts, *args, **kwargs):
        sol0 = find_clique(
            q_pts,
            k_pts,
            delta=self.delta,
            epsilon=self.epsilon,
            lower_bound=2,
            heuristic=self.use_heuristic,
        )
        qc, kc, tm = sol0["qc"], sol0["kc"], sol0["tm"]
        tform = KabschEstimate(kc, qc)

        transl_est = tform.coefs[0, :]
        rotmat = tform.coefs[1:, :]
        zoom_est = np.linalg.det(rotmat) ** (1 / len(transl_est))
        rotmat /= zoom_est

        sol = dict()
        e = self.__estim_name__
        sol[f"{e}_success"] = len(qc)
        sol[f"{e}_zoom"] = zoom_est
        sol[f"{e}_rotation"] = rotmat.tolist()
        sol[f"{e}_translation"] = transl_est.tolist()
        sol[f"{e}_time-clq"] = float(tm["end"] - tm["mid"])
        sol[f"{e}_time-graph"] = float(tm["mid"] - tm["start"])
        return sol


class RTSHeuristic(BaseNDEstimator):
    __estim_name__ = "rts-heuristic"

    def __init__(self, delta, epsilon):
        super().__init__(params=["time-clq", "time-graph"])
        self.delta = delta * np.pi / 180
        self.epsilon = epsilon
        self.use_heuristic = True

    def _call_impl(self, q_pts, k_pts, *args, **kwargs):
        sol0 = find_clique(
            q_pts,
            k_pts,
            delta=self.delta,
            epsilon=self.epsilon,
            lower_bound=2,
            heuristic=self.use_heuristic,
        )
        qc, kc, tm = sol0["qc"], sol0["kc"], sol0["tm"]
        tform = KabschEstimate(kc, qc)

        transl_est = tform.coefs[0, :]
        rotmat = tform.coefs[1:, :]
        zoom_est = np.linalg.det(rotmat) ** (1 / len(transl_est))
        rotmat /= zoom_est

        sol = dict()
        e = self.__estim_name__
        sol[f"{e}_success"] = len(qc)
        sol[f"{e}_zoom"] = zoom_est
        sol[f"{e}_rotation"] = rotmat.tolist()
        sol[f"{e}_translation"] = transl_est.tolist()
        sol[f"{e}_time-clq"] = float(tm["end"] - tm["mid"])
        sol[f"{e}_time-graph"] = float(tm["mid"] - tm["start"])
        return sol
