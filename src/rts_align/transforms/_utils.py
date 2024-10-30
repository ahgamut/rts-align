import numpy as np


def affine_simple(pt, coefs):
    a0 = np.zeros(len(coefs))
    a0[-3] = 1
    a0[-2:] = pt
    return np.matmul(a0, coefs)


def tp_simple(pt, src, coefs):
    a0 = np.zeros(len(coefs))
    a0[-3] = 1
    a0[-2:] = pt
    a0[:-3] = 1e-10 + np.sum((src - pt) ** 2, axis=1)
    a0[:-3] = 0.5 * (a0[:-3]) * np.log(a0[:-3])
    return np.matmul(a0, coefs)
