import numpy as np

try:
    from ._utils_opt import affine_numbaized as _affine
    from ._utils_opt import s_affine_numbaized as s_affine
except ImportError:
    from ._utils import affine_simple as _affine
    from ._utils_opt import s_affine_simple as s_affine


class AffineTransform:
    def __init__(self, coefs):
        self.coefs = coefs  # 3 rows, 2 columns

    def __call__(self, pt):
        arr = np.zeros(len(self.coefs), dtype=np.float32)
        arr[-3] = 1
        arr[-2:] = pt
        return np.matmul(arr, self.coefs)

    def return_call_pieces(self):
        return lambda x: _affine(x, self.coefs)
