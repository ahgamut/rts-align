import numpy as np
import numba as nb

@nb.jit(nopython=True)
def _affine_numbaized_actual(pts, res, M, coefsT):
    a0 = np.zeros(3, dtype=np.float32)
    for i in range(M):
        pt = pts[i]
        a0[-3] = 1
        a0[-2:] = pt
        res[i, 0] = np.dot(a0, coefsT[0])
        res[i, 1] = np.dot(a0, coefsT[1])


def affine_numbaized(pts, coefs):
    res = np.zeros(pts.shape, dtype=np.float32)
    coefsT = np.ascontiguousarray(np.transpose(coefs))
    _affine_numbaized_actual(pts, res, len(pts), coefsT)
    return res


@nb.jit(nopython=True)
def _tp_numbaized_actual(pts, res, M, src, coefsT, N):
    a0 = np.zeros(N, dtype=np.float32)
    for i in range(M):
        pt = pts[i]
        a0[:-3] = 1e-10 + np.sum((src - pt) ** 2, axis=1)
        a0[:-3] = 0.5 * (a0[:-3]) * np.log(a0[:-3])
        a0[-3] = 1
        a0[-2:] = pt
        res[i, 0] = np.dot(a0, coefsT[0])
        res[i, 1] = np.dot(a0, coefsT[1])


def tp_numbaized(pts, src, coefs):
    res = np.zeros(pts.shape, dtype=np.float32)
    coefsT = np.ascontiguousarray(np.transpose(coefs))
    _tp_numbaized_actual(pts, res, len(pts), src, coefsT, len(coefs))
    return res
