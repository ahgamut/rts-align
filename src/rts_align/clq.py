import numpy as np
import cliquematch
from rts_align.core import construct_graph


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


def get_clique(adjmat, lower_bound, upper_bound):
    # first use the lower bound given
    G1_ind = strip_graph(adjmat, lower_bound)
    res1 = adjmat[G1_ind, :]
    res1 = res1[:, G1_ind]
    G1 = cliquematch.Graph.from_matrix(res1)
    c1 = (
        np.array(
            G1.get_max_clique(
                upper_bound=upper_bound,
                lower_bound=lower_bound,
                use_dfs=False,
                use_heuristic=True,
            ),
            dtype=np.int32,
        )
        - 1
    )
    l1 = len(c1)
    lower_bound = max(lower_bound, l1 - 1)

    # now hopefully the lower bound has increased,
    # so res2 will have fewer vertices than res1
    G2_ind = strip_graph(adjmat, lower_bound)
    res2 = adjmat[G2_ind, :]
    res2 = res2[:, G2_ind]
    # print(res1.shape, res2.shape, lower_bound)
    G2 = cliquematch.Graph.from_matrix(res2)

    c2 = (
        np.array(
            G2.get_max_clique(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
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
    return c_sub


def find_clique(q_pts, k_pts, delta=0.01, epsilon=0.1, lower_bound=3):
    delta = delta * np.pi / 180.0
    qlen = len(q_pts)
    klen = len(k_pts)

    res_basic = construct_graph(
        q_pts, k_pts, delta=delta, epsilon=epsilon, max_ratio=10, min_ratio=0.1
    )
    res_basic = res_basic != 0

    c = get_clique(res_basic, lower_bound, upper_bound=min(qlen, klen))
    qc = q_pts[c // len(k_pts), :]
    kc = k_pts[c % len(k_pts), :]
    return qc, kc


def find_all_cliques(q_pts, k_pts, delta=0.01, epsilon=0.1, lower_bound=3, total=10):
    delta = delta * np.pi / 180.0
    qlen = len(q_pts)
    klen = len(k_pts)

    res_basic = construct_graph(
        q_pts, k_pts, delta=delta, epsilon=epsilon, max_ratio=10, min_ratio=0.1
    )
    res_basic = res_basic != 0

    c_set = []
    while len(c_set) < total:
        try:
            c = get_clique(res_basic, lower_bound, upper_bound=min(qlen, klen))
            c_set.append(c)
            # print(np.sum(res_basic), len(c))
            res_basic[c, :] = False
            res_basic[:, c] = False
        except Exception as e:
            break

    res = []
    for c in c_set:
        qc = q_pts[c // len(k_pts), :]
        kc = k_pts[c % len(k_pts), :]
        res.append({"Q": qc, "K": kc})

    return res
