import numpy as np

from .affine import AffineTransform

try:
    from ._utils_opt import tp_numbaized as tpfunc
except ImportError:
    from ._utils import tp_simple as tpfunc

U = lambda x: x * x * np.log(x + 1e-10)
Ur = lambda x: 0.5 * x * np.log(x + 1e-10)


class TPSTransform:
    def __init__(self, src, coefs):
        self.src = src
        self.coefs = coefs

    def __call__(self, pt):
        arr = np.zeros(len(self.coefs), dtype=np.float32)
        arr[-3] = 1
        arr[-2:] = pt
        arr[:-3] = Ur(np.sum((self.src - pt) ** 2, axis=1))
        return np.matmul(arr, self.coefs)

    def return_call_pieces(self):
        return lambda x: tpfunc(x, self.src, self.coefs)

    @staticmethod
    def from_decomp(src, subx, suby):
        coefs = np.zeros((len(subx) + 3, 2), dtype=np.float32)
        coefs[:-3, 0] = subx
        coefs[:-3, 1] = suby
        coefs[-3] = 0
        coefs[-2] = [0, 0]
        coefs[-1] = [0, 0]

        return TPSTransform(src, coefs)


class ThinPlateSpline(object):
    def __init__(self, src, dst):
        self.src = np.array(src)
        self.dst = np.array(dst)
        self.Linv, self.K = ThinPlateSpline._calculate_Linv(self.src)
        self.coefs = ThinPlateSpline._calculate_coefs(self.dst, self.Linv)

    @staticmethod
    def _calculate_Linv(pts):
        n = len(pts)
        # R = squareform(pdist(pts, metric="euclidean"))
        R = np.zeros((n,n), dtype=np.float64)
        for i in range(n):
            pt = pts[i]
            fd = lambda x: np.linalg.norm(x-pt)
            R[i, :] = np.apply_along_axis(fd, 0, pts)
        np.fill_diagonal(R, 1)
        K = U(R)
        P = np.column_stack([np.ones((len(pts), 1)), pts])
        L = np.block([[K, P], [P.T, np.zeros((3, 3))]])
        Linv = np.linalg.inv(L)

        return Linv, K

    @staticmethod
    def _calculate_coefs(dst, Linv):
        coefs = np.zeros((len(Linv), 2), dtype=np.float32)
        arr = np.zeros(len(Linv), dtype=np.float32)
        arr[:-3] = dst[:, 0]
        coefs[:, 0] = np.matmul(Linv, arr)
        arr[:-3] = dst[:, 1]
        coefs[:, 1] = np.matmul(Linv, arr)
        return coefs

    def __call__(self, pt):
        arr = np.zeros(len(self.coefs), dtype=np.float32)
        arr[-3] = 1
        arr[-2:] = pt
        arr[:-3] = Ur(np.sum((self.src - pt) ** 2, axis=1))
        return np.matmul(arr, self.coefs)

    def return_call_pieces(self):
        return lambda x: tpfunc(x, self.src, self.coefs)

    def get_affine(self):
        return AffineTransform(self.coefs[-3:])

    def get_sequence(self, epsilon=1e-10):
        Lninv = self.Linv[: len(self.K), : len(self.K)]
        LiKLi = np.matmul(np.matmul(Lninv, self.K), Lninv)
        eigs, eigv = np.linalg.eig(LiKLi)
        eigs = np.real(eigs)
        eigv = np.real(eigv)

        subco = self.coefs[:-3]
        resco = np.zeros(subco.shape)
        resco[:, 0] = np.linalg.solve(eigv, subco[:, 0])
        resco[:, 1] = np.linalg.solve(eigv, subco[:, 1])

        sig = abs(eigs) > epsilon
        eigs = eigs[sig]
        eigv = eigv[:, sig]
        resco = resco[sig]

        # sort into increasing order
        order = np.argsort(eigs)
        print(eigs[order])
        eigv = eigv[:, order]
        resco = resco[order]

        forms = []
        forms.append(AffineTransform(self.coefs[-3:]))
        # the decomposition is orthogonal so
        # the order of i doesn't matter, as
        # long as you're getting all of them
        for i in range(len(resco)):
            subx = 0
            suby = 0
            for j in range(i+1):
                subx += resco[j, 0] * eigv[:, j]
                suby += resco[j, 1] * eigv[:, j]

            new_form = TPSTransform.from_decomp(self.src, subx, suby)
            new_form.coefs[-3:] = self.coefs[-3:]
            forms.append(new_form)

        return forms



    def get_decomposition(self, epsilon=1e-10):
        Lninv = self.Linv[: len(self.K), : len(self.K)]
        LiKLi = np.matmul(np.matmul(Lninv, self.K), Lninv)
        eigs, eigv = np.linalg.eig(LiKLi)
        eigs = np.real(eigs)
        eigv = np.real(eigv)

        subco = self.coefs[:-3]
        resco = np.zeros(subco.shape)
        resco[:, 0] = np.linalg.solve(eigv, subco[:, 0])
        resco[:, 1] = np.linalg.solve(eigv, subco[:, 1])

        sig = abs(eigs) > epsilon
        eigs = eigs[sig]
        eigv = eigv[:, sig]
        resco = resco[sig]

        forms = []
        forms.append(AffineTransform(self.coefs[-3:]))
        # the decomposition is orthogonal so
        # the order of i doesn't matter, as
        # long as you're getting all of them
        for i in range(len(resco)):
            subx = resco[i, 0] * eigv[:, i]
            suby = resco[i, 1] * eigv[:, i]
            forms.append(TPSTransform.from_decomp(self.src, subx, suby))

        return forms
