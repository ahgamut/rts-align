import numpy as np

try:
    from ._utils_opt import affine_numbaized as _affine
except ImportError:
    from ._utils import affine_simple as _affine


def centerify(x):
    cent = np.mean(x, axis=0)
    return cent, x - cent


class KabschEstimate:
    def __init__(self, src, dst):
        self.src = np.array(src)
        self.dst = np.array(dst)
        self.coefs = KabschEstimate.find_coefs(src, dst)

    @staticmethod
    def find_coefs(src, dst):
        coefs = np.zeros((3, 2), dtype=np.float32)
        # kabsch algorithm to get rotation and translation
        # https://en.wikipedia.org/wiki/Kabsch_algorithm
        # following Umeyama's variant to calculate scale
        src_cent, src_norm = centerify(src)
        dst_cent, dst_norm = centerify(dst)
        n = src.shape[0]
        m = src.shape[1]

        src_var = np.mean(np.linalg.norm(src_norm, axis=1) ** 2)
        dst_var = np.mean(np.linalg.norm(dst_norm, axis=1) ** 2)

        H = ((dst_norm.T) @ src_norm) / n
        U, d, Vt = np.linalg.svd(H, full_matrices=True, compute_uv=True)

        S = np.eye(m, m)
        det = np.linalg.det(Vt) * np.linalg.det(U)
        S[m-1, m-1] = 1 if det >= 0 else -1

        # print(src_var, d)
        scale = np.trace(np.diag(d) @ S) / (src_var + 1e-8)
        rotmat = (scale * (U @ S @ Vt)).T
        shift =  (-src_cent @ rotmat + dst_cent)
        # print(scale, rotmat, shift)

        coefs[0, :] = shift
        coefs[1, :] = rotmat[0, :]
        coefs[2, :] = rotmat[1, :]
        return coefs

    def __call__(self, pt):
        arr = np.zeros(len(self.coefs), dtype=np.float32)
        arr[-3] = 1
        arr[-2:] = pt
        return np.matmul(arr, self.coefs)

    def return_call_pieces(self):
        return lambda x: _affine(x, self.coefs)
